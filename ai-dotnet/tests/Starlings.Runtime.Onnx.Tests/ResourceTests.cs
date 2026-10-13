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
        using var session = OnnxModelSession.Create(TestModels.Slow, SingleThread);

        // Uncancelled reference duration for the same input.
        var full = Stopwatch.StartNew();
        RunSlow(session, 256, CancellationToken.None);
        full.Stop();

        for (int cycle = 0; cycle < 3; cycle++)
        {
            using var cts = new CancellationTokenSource();
            var sw = Stopwatch.StartNew();
            cts.CancelAfter(TimeSpan.FromMilliseconds(Math.Max(10, full.Elapsed.TotalMilliseconds / 10)));
            TimeSpan cancelledAt = TimeSpan.Zero;
            cts.Token.Register(() => cancelledAt = sw.Elapsed);

            var ex = Assert.ThrowsAny<OperationCanceledException>(() => RunSlow(session, 256, cts.Token));
            var latency = sw.Elapsed - cancelledAt;
            output.WriteLine($"cycle {cycle}: full run {full.Elapsed.TotalMilliseconds:F0} ms; cancelled at {cancelledAt.TotalMilliseconds:F0} ms; returned {latency.TotalMilliseconds:F2} ms later; inner: {ex.InnerException?.GetType().Name}: {ex.InnerException?.Message}");
            Assert.Equal(cts.Token, ex.CancellationToken);
            Assert.True(sw.Elapsed < full.Elapsed, "cancelled run took as long as an uncancelled one");

            // Reuse after cancellation: a fresh run completes and is correct.
            float[] small = RunSlow(session, 4, CancellationToken.None);
            Assert.All(small, v => Assert.True(float.IsFinite(v)));
        }
    }

    private static float[] RunSlow(OnnxModelSession session, int n, CancellationToken cancellationToken)
    {
        var x = new float[n * n];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = ((i % 7) - 3) * 0.01f;
        }

        using var xv = OrtValue.CreateTensorValueFromMemory(x, [n, n]);
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
