using System.Diagnostics;
using Starlings.TestSupport;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Runtime.Onnx.Tests;

/// <summary>Lifecycle, concurrency and cancellation (gate G4 checklist) on weight-free graphs; runs in CI.</summary>
public class ResourceTests(ITestOutputHelper output)
{
    private static readonly OnnxSessionOptions SingleThread = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };

    [Fact]
    public void RepeatedCreateRunDisposeKeepsMemoryBounded()
    {
        const int Warmup = 50;
        const int Iterations = 500;
        for (int i = 0; i < Warmup; i++)
        {
            CreateRunDispose();
        }

        long before = Settle();
        for (int i = 0; i < Iterations; i++)
        {
            CreateRunDispose();
        }

        long after = Settle();
        double growthMb = (after - before) / (1024.0 * 1024.0);
        output.WriteLine($"private memory after {Warmup} warmup cycles: {before / 1048576.0:F1} MB; after {Iterations} more: {after / 1048576.0:F1} MB; growth {growthMb:F2} MB");

        // A leaked session (graph, allocator arena, thread pool) costs far more than 64 KB per cycle.
        Assert.True(growthMb < 32, $"private memory grew {growthMb:F1} MB over {Iterations} create/run/dispose cycles");

        static void CreateRunDispose()
        {
            using var session = OnnxModelSession.Create(TestModels.Add, SingleThread);
            SessionTests.Run(session, 1);
        }
    }

    [Fact]
    public void NativeHeapTrimIsOnByDefaultOnLinuxGlibcAndCanBeTurnedOff()
    {
        // On by default where glibc exists (Linux in CI); a no-op, reported as off, elsewhere.
        Assert.True(new OnnxSessionOptions().TrimNativeHeapOnRelease);
        using (var session = OnnxModelSession.Create(TestModels.Add, SingleThread))
        {
            Assert.Equal(OperatingSystem.IsLinux() && NativeHeap.Enabled, session.Diagnostics.TrimNativeHeapOnRelease);
            SessionTests.Run(session, 1);
        }

        // Releasing the session above ran the trim without error; a direct call reports whether it ran.
        Assert.Equal(OperatingSystem.IsLinux() && NativeHeap.Enabled, NativeHeap.Trim());

        using var off = OnnxModelSession.Create(TestModels.Add, SingleThread with { TrimNativeHeapOnRelease = false });
        Assert.False(off.Diagnostics.TrimNativeHeapOnRelease);
        SessionTests.Run(off, 1);
    }

    [Fact]
    public void CpuMemoryArenaFollowsTheOptionAndDefaultsToOnnxRuntimes()
    {
        Assert.Null(new OnnxSessionOptions().CpuMemoryArena);
        using (var session = OnnxModelSession.Create(TestModels.Add, SingleThread))
        {
            Assert.True(session.Diagnostics.CpuMemoryArena);
        }

        foreach (bool arena in new[] { false, true })
        {
            using var session = OnnxModelSession.Create(TestModels.Add, SingleThread with { CpuMemoryArena = arena });
            Assert.Equal(arena, session.Diagnostics.CpuMemoryArena);
            SessionTests.Run(session, 1);
        }
    }

    [Fact]
    public void DisposeIsIdempotentAndLaterCallsThrow()
    {
        var session = OnnxModelSession.Create(TestModels.Add, SingleThread);
        SessionTests.Run(session, 1);

        session.Dispose();
        session.Dispose();

        Assert.True(session.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => SessionTests.Run(session, 1));
        Assert.NotNull(session.Diagnostics);
    }

    [Fact]
    public void ConcurrentRunsOnSharedSessionAreIsolated()
    {
        const int Threads = 8;
        const int RunsPerThread = 250;
        using var session = OnnxModelSession.Create(TestModels.Add, SingleThread);
        int failures = 0;

        Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, t =>
        {
            for (int i = 0; i < RunsPerThread; i++)
            {
                float v = (t * 1000) + i;
                float[] x = [v, v + 1, v + 2, v + 3];
                float[] y = [v, v, v, v];
                using var xv = OrtValue.CreateTensorValueFromMemory(x, [1, 4]);
                using var yv = OrtValue.CreateTensorValueFromMemory(y, [1, 4]);
                using var outputs = session.Run(["x", "y"], [xv, yv], ["z"]);
                var z = outputs[0].GetTensorDataAsSpan<float>();
                if (z[0] != 2 * v || z[3] != (2 * v) + 3)
                {
                    Interlocked.Increment(ref failures);
                }
            }
        });

        output.WriteLine($"{Threads} threads x {RunsPerThread} runs on one session: {failures} wrong outputs");
        Assert.Equal(0, failures);
    }

    [Fact]
    public async Task DisposeDuringRunLetsTheRunFinishThenReleases()
    {
        var session = OnnxModelSession.Create(TestModels.Slow, SingleThread);
        using var started = new ManualResetEventSlim();
        var inFlight = Task.Run(() =>
        {
            started.Set();
            return RunSlow(session, 384, CancellationToken.None);
        });

        started.Wait(TestContext.Current.CancellationToken);
        Thread.Sleep(50);
        session.Dispose();

        Assert.True(session.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => RunSlow(session, 4, CancellationToken.None));
        float[] result = await inFlight;
        Assert.All(result, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void CancelledTokenPreventsTheRun()
    {
        using var session = OnnxModelSession.Create(TestModels.Slow, SingleThread);
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var sw = Stopwatch.StartNew();
        Assert.ThrowsAny<OperationCanceledException>(() => RunSlow(session, 512, cts.Token));
        output.WriteLine($"pre-cancelled run returned after {sw.Elapsed.TotalMilliseconds:F2} ms");
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void CancellationStopsAtNodeBoundaryAndSessionStaysUsable()
    {
        // What this checks: a cancellation requested while the run is in flight stops it at a node boundary (ONNX
        // Runtime's executor throws for the terminate flag, and the run returns long before the remaining nodes would
        // have run), the exception carries the caller's token, and the session stays usable.
        //
        // How it is timed: a canceller thread waits until the run has been entered plus a tenth of an uncancelled run,
        // reads the clock and only then calls Cancel(), which sets the terminate flag before it returns. The test
        // thread reads the cancellation time after joining the canceller. Earlier versions read it from a callback
        // registered on the token; the session's own terminate callback, registered later, runs first, so on Windows
        // CI the run threw and the test read the clock before its callback had run (PR #37, first Windows run:
        // "cancelled at -256193 ms"). Before that, the whole cancelled run was compared with an uncancelled one, which
        // failed when the timer fired late (PR #30, first Windows run).
        //
        // A cycle whose canceller was scheduled after half an uncancelled run cannot tell a node-boundary stop from a
        // run to completion, so it is reported as inconclusive and repeated; the token and reuse checks still apply.
        // Three conclusive cycles are required.
        const int N = 384; // about 3.4x the work of n = 256, so scheduling jitter is small against the run
        const int ConclusiveCycles = 3;
        const int MaxAttempts = 12;
        using var session = OnnxModelSession.Create(TestModels.Slow, SingleThread);

        // Uncancelled reference duration for the same input: the median of three runs (the first includes warm-up).
        var runs = new double[3];
        for (int i = 0; i < runs.Length; i++)
        {
            var w = Stopwatch.StartNew();
            RunSlow(session, N, CancellationToken.None);
            runs[i] = w.Elapsed.TotalMilliseconds;
        }

        Array.Sort(runs);
        double full = runs[1];
        long cancelOffset = (long)(full / 10 * Stopwatch.Frequency / 1000);
        int conclusive = 0;
        for (int attempt = 0; attempt < MaxAttempts && conclusive < ConclusiveCycles; attempt++)
        {
            using var cts = new CancellationTokenSource();
            using var entering = new ManualResetEventSlim();
            long start = 0;
            long cancelled = 0;
            var canceller = new Thread(() =>
            {
                entering.Wait();
                long due = Volatile.Read(ref start) + cancelOffset;
                while (Stopwatch.GetTimestamp() < due)
                {
                    Thread.Sleep(1);
                }

                cancelled = Stopwatch.GetTimestamp();
                cts.Cancel(); // runs the session's callback synchronously: the terminate flag is set when this returns
            })
            { IsBackground = true, Name = "canceller" };
            canceller.Start();

            OperationCanceledException? ex = null;
            long returned;
            try
            {
                RunSlow(session, N, cts.Token, () =>
                {
                    Volatile.Write(ref start, Stopwatch.GetTimestamp());
                    entering.Set();
                });
            }
            catch (OperationCanceledException e)
            {
                ex = e;
            }
            finally
            {
                returned = Stopwatch.GetTimestamp();

                // Whatever happened, let the canceller finish before cts and entering are disposed.
                entering.Set();
                canceller.Join();
            }

            double cancelledAt = Ms(start, cancelled);
            double latency = Ms(cancelled, returned);
            bool judged = cancelledAt < full / 2;
            output.WriteLine($"attempt {attempt}: full run {full:F0} ms; cancelled at {cancelledAt:F1} ms; returned {latency:F2} ms later{(judged ? string.Empty : " (inconclusive: cancelled after half a run)")}; inner: {ex?.InnerException?.GetType().Name}: {ex?.InnerException?.Message}");
            if (judged)
            {
                // Cancelled with more than half a run left: the run must have been stopped, by the executor itself.
                Assert.NotNull(ex);
                Assert.Equal(cts.Token, ex.CancellationToken);
                Assert.IsType<OnnxRuntimeException>(ex.InnerException);

                // Ignoring the cancellation would leave more than half a run (full - cancelledAt > full / 2).
                Assert.True(latency < full / 4, $"returned {latency:F0} ms after the cancellation, not at a node boundary (an uncancelled run takes {full:F0} ms)");
                conclusive++;
            }
            else if (ex is not null)
            {
                Assert.Equal(cts.Token, ex.CancellationToken);
            }

            // Reuse after cancellation: a fresh run completes and is correct.
            float[] small = RunSlow(session, 4, CancellationToken.None);
            Assert.All(small, v => Assert.True(float.IsFinite(v)));
        }

        Assert.True(conclusive == ConclusiveCycles, $"only {conclusive} of {MaxAttempts} cancellations happened within the first half of a {full:F0} ms run");

        static double Ms(long from, long to) => (to - from) * 1000.0 / Stopwatch.Frequency;
    }

    private static float[] RunSlow(OnnxModelSession session, int n, CancellationToken cancellationToken, Action? entering = null)
    {
        var x = new float[n * n];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = ((i % 7) - 3) * 0.01f;
        }

        using var xv = OrtValue.CreateTensorValueFromMemory(x, [n, n]);
        entering?.Invoke();
        using var outputs = session.Run(["x"], [xv], ["y"], cancellationToken);
        return outputs[0].GetTensorDataAsSpan<float>().ToArray();
    }

    private static long Settle()
    {
        for (int i = 0; i < 3; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        using var process = Process.GetCurrentProcess();
        process.Refresh();
        return process.PrivateMemorySize64;
    }
}
