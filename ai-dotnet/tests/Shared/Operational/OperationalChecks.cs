using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Starlings.Acquisition;
using Xunit;

namespace Starlings.TestSupport;

/// <summary>
/// Gate G4 operational checks (plan 10.2) shared by the model test projects: malformed locks and snapshots, queue
/// saturation, timed-out requests with cancellation latency, and disposal during in-flight work. The long soak is
/// ai-dotnet/tools/Starlings.Soak.Tool.
/// </summary>
internal static class OperationalChecks
{
    /// <summary>
    /// Every corruption of the model's own lock JSON fails to parse with <see cref="FormatException"/>: truncated text,
    /// a branch name instead of the commit, a short or upper-case digest, a negative size, a path leaving the snapshot,
    /// an unknown schema version, and no files.
    /// </summary>
    public static void MalformedLocksAreRejected(ModelAssetLock assetLock, ITestOutputHelper output)
    {
        string json = assetLock.ToJson();
        Assert.Equal(assetLock.Files, ModelAssetLock.Parse(json).Files);
        var mutations = new Dictionary<string, Func<JsonObject, string>>
        {
            ["truncated"] = _ => json[..(json.Length / 2)],
            ["branch revision"] = o => Set(o, n => n["revision"] = "main"),
            ["short digest"] = o => Set(o, n => n["files"]![0]!["sha256"] = assetLock.Files[0].Sha256[..32]),
            ["upper-case digest"] = o => Set(o, n => n["files"]![0]!["sha256"] = assetLock.Files[0].Sha256.ToUpperInvariant()),
            ["negative size"] = o => Set(o, n => n["files"]![0]!["size"] = -1),
            ["escaping path"] = o => Set(o, n => n["files"]![0]!["path"] = "../" + assetLock.Files[0].Path),
            ["unknown schema"] = o => Set(o, n => n["schema_version"] = 99),
            ["no files"] = o => Set(o, n => n["files"] = new JsonArray()),
            ["duplicate file"] = o => Set(o, n => n["files"]!.AsArray().Add(n["files"]![0]!.DeepClone())),
        };
        foreach (var (name, mutate) in mutations)
        {
            string bad = mutate(JsonNode.Parse(json)!.AsObject());
            var ex = Assert.Throws<FormatException>(() => ModelAssetLock.Parse(bad));
            output.WriteLine($"{name}: {ex.Message}");
        }

        static string Set(JsonObject o, Action<JsonObject> change)
        {
            change(o);
            return o.ToJsonString();
        }
    }

    /// <summary>
    /// A snapshot directory holding every locked file at the wrong size (a partial copy) is rejected with
    /// <see cref="ModelAssetIntegrityException"/> before any model loads; an empty one with <see cref="ModelAssetMissingException"/>.
    /// </summary>
    public static void WrongSnapshotsAreRejected(ModelAssetLock assetLock, Action<string> openDirectory)
    {
        string dir = Directory.CreateTempSubdirectory("starlings-g4-snapshot-").FullName;
        try
        {
            Assert.Throws<ModelAssetMissingException>(() => openDirectory(dir));
            foreach (var file in assetLock.Files)
            {
                string path = Path.Combine(dir, file.Path);
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                File.WriteAllText(path, "{}");
            }

            var ex = Assert.Throws<ModelAssetIntegrityException>(() => openDirectory(dir));
            Assert.Contains(assetLock.Files[0].Path, ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// Queue saturation. The libraries have no request queue or admission limit (plan 5.3: added only when a benchmark
    /// justifies it), so saturation means more concurrent callers than cores on one shared instance: every call must
    /// complete (no deadlock, no rejection) with the result it gives alone. The latency spread is reported; under
    /// oversubscription it grows with the number of callers, which is the expected behaviour without a queue.
    /// </summary>
    public static void SaturatedCallersComplete<T>(
        int inputs, Func<int, CancellationToken, T> call, Action<T, T, int> assertSame, ITestOutputHelper output,
        int? callers = null, int callsPerCaller = 4, TimeSpan? timeout = null)
    {
        int n = callers ?? Environment.ProcessorCount * 4;
        var reference = Enumerable.Range(0, inputs).Select(i => call(i, CancellationToken.None)).ToArray();
        var results = new T[n, callsPerCaller];
        var latencies = new double[n * callsPerCaller];
        using var start = new Barrier(n);
        var wall = Stopwatch.StartNew();
        var threads = Enumerable.Range(0, n).Select(c => new Thread(() =>
        {
            start.SignalAndWait();
            for (int k = 0; k < callsPerCaller; k++)
            {
                long t = Stopwatch.GetTimestamp();
                results[c, k] = call(((c * callsPerCaller) + k) % inputs, CancellationToken.None);
                latencies[(c * callsPerCaller) + k] = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            }
        })).ToList();
        threads.ForEach(t => t.Start());
        var limit = timeout ?? TimeSpan.FromMinutes(5);
        foreach (var t in threads)
        {
            Assert.True(t.Join(limit - wall.Elapsed > TimeSpan.Zero ? limit - wall.Elapsed : TimeSpan.Zero), $"{n} saturated callers did not finish within {limit}");
        }

        for (int c = 0; c < n; c++)
        {
            for (int k = 0; k < callsPerCaller; k++)
            {
                int i = ((c * callsPerCaller) + k) % inputs;
                assertSame(reference[i], results[c, k], i);
            }
        }

        Array.Sort(latencies);
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{n} callers x {callsPerCaller} calls on {Environment.ProcessorCount} logical CPUs: wall {wall.Elapsed.TotalMilliseconds:F0} ms; call latency p50 {latencies[latencies.Length / 2]:F1} ms, p95 {latencies[(int)(latencies.Length * 0.95)]:F1} ms, max {latencies[^1]:F1} ms; every call completed with its single-call result (no queue, no admission limit)"));
    }

    /// <summary>
    /// Timed-out requests: <paramref name="longCall"/> runs under a deadline (<see cref="CancellationTokenSource.CancelAfter(TimeSpan)"/>)
    /// several times; each must throw <see cref="OperationCanceledException"/> carrying the token, its cancellation
    /// latency (deadline to throw) is reported, and <paramref name="reuse"/> must then succeed on the same instance.
    /// Returns the latencies in milliseconds.
    /// </summary>
    public static double[] TimedOutRequestsThenReuse(Action<CancellationToken> longCall, Action reuse, TimeSpan deadline, TimeSpan latencyBound, ITestOutputHelper output, int repeats = 3)
    {
        var latencies = new double[repeats];
        for (int r = 0; r < repeats; r++)
        {
            using var cts = new CancellationTokenSource();
            var sw = Stopwatch.StartNew();

            // The deadline's time is read before Cancel(), on the cancelling thread, and read here after joining it. A
            // callback registered on the token would race the call: callbacks run newest first, so the call's own
            // cancellation callback (registered later) can stop the call before a recorder registered here has run.
            var canceller = CancelAfter(cts, deadline);
            OperationCanceledException ex;
            long now;
            try
            {
                ex = Assert.ThrowsAny<OperationCanceledException>(() => longCall(cts.Token));
            }
            finally
            {
                now = Stopwatch.GetTimestamp();
                canceller.Join();
            }

            long fired = Volatile.Read(ref canceller.CancelledAt);
            Assert.True(fired != 0 && fired <= now, "the call threw before its deadline");
            Assert.Equal(cts.Token, ex.CancellationToken);
            latencies[r] = (now - fired) * 1000.0 / Stopwatch.Frequency;
            output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"deadline {deadline.TotalMilliseconds:F0} ms: threw after {sw.Elapsed.TotalMilliseconds:F0} ms, cancellation latency {latencies[r]:F2} ms"));
            reuse();
        }

        Assert.All(latencies, l => Assert.True(l < latencyBound.TotalMilliseconds, $"cancellation latency {l:F0} ms exceeds {latencyBound.TotalMilliseconds:F0} ms"));
        return latencies;
    }

    /// <summary>
    /// Cancels <paramref name="cts"/> <paramref name="delay"/> from now on a dedicated thread, recording the clock
    /// (<see cref="Stopwatch.GetTimestamp"/>) immediately before <see cref="CancellationTokenSource.Cancel()"/>.
    /// <see cref="Canceller.Join"/> it before reading <see cref="Canceller.CancelledAt"/> or disposing the source.
    /// </summary>
    public static Canceller CancelAfter(CancellationTokenSource cts, TimeSpan delay)
    {
        var canceller = new Canceller();
        long due = Stopwatch.GetTimestamp() + (long)(delay.TotalSeconds * Stopwatch.Frequency);
        canceller.Thread = new Thread(() =>
        {
            while (Stopwatch.GetTimestamp() < due)
            {
                Thread.Sleep(1);
            }

            Volatile.Write(ref canceller.CancelledAt, Stopwatch.GetTimestamp());
            cts.Cancel();
        })
        { IsBackground = true, Name = "canceller" };
        canceller.Thread.Start();
        return canceller;
    }

    /// <summary>A pending cancellation started by <see cref="CancelAfter"/>.</summary>
    public sealed class Canceller
    {
        /// <summary>Clock reading taken just before cancelling; 0 until then.</summary>
        public long CancelledAt;

        internal Thread Thread = null!;

        /// <summary>Waits until the cancellation has been requested and its callbacks have run.</summary>
        public void Join() => Thread.Join();

        /// <summary>Milliseconds from <paramref name="from"/> (a <see cref="Stopwatch.GetTimestamp"/> reading) to the cancellation.</summary>
        public double MillisecondsSince(long from) => (Volatile.Read(ref CancelledAt) - from) * 1000.0 / Stopwatch.Frequency;
    }

    /// <summary>
    /// Disposal during in-flight work (shutdown): <paramref name="work"/> runs repeatedly on another thread while
    /// <paramref name="dispose"/> is called. The in-flight call must either complete or throw
    /// <see cref="ObjectDisposedException"/> (work not yet started when disposal happened); any other exception, or a
    /// hang, fails. After disposal <paramref name="afterDispose"/> must throw <see cref="ObjectDisposedException"/>.
    /// </summary>
    public static void DisposeDuringInFlightWork(Action work, Action dispose, Action afterDispose, ITestOutputHelper output, TimeSpan? settle = null)
    {
        using var started = new ManualResetEventSlim();
        int completed = 0;
        Exception? failure = null;
        bool sawDisposed = false;
        var worker = new Thread(() =>
        {
            started.Set();
            try
            {
                while (true)
                {
                    work();
                    Interlocked.Increment(ref completed);
                }
            }
            catch (ObjectDisposedException)
            {
                sawDisposed = true;
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        });
        worker.Start();
        started.Wait();
        Thread.Sleep(settle ?? TimeSpan.FromMilliseconds(50));
        var sw = Stopwatch.StartNew();
        dispose();
        double disposeMs = sw.Elapsed.TotalMilliseconds;
        Assert.True(worker.Join(TimeSpan.FromMinutes(2)), "the in-flight call did not finish after disposal");
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"dispose returned in {disposeMs:F1} ms; worker finished {sw.Elapsed.TotalMilliseconds:F0} ms after dispose; {completed} calls completed; ended with ObjectDisposedException: {sawDisposed}"));
        Assert.Null(failure);
        Assert.True(sawDisposed, "the worker never observed the disposal");
        Assert.Throws<ObjectDisposedException>(afterDispose);
    }
}
