using Microsoft.ML.OnnxRuntime;
using Starlings.TestSupport;

namespace Starlings.Runtime.Onnx.Tests;

/// <summary>Dedicated run threads (<see cref="OnnxSessionOptions.DedicatedRunThreads"/>) on weight-free graphs.</summary>
public class RunThreadsTests
{
    private static readonly OnnxSessionOptions SingleThread = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };

    [Fact]
    public void OptionIsOffByDefaultAndReported()
    {
        Assert.Null(new OnnxSessionOptions().DedicatedRunThreads);
        using (var session = OnnxModelSession.Create(TestModels.Add, SingleThread))
        {
            Assert.False(session.Diagnostics.DedicatedRunThreads);
            SessionTests.Run(session, 1);
        }

        foreach (bool dedicated in new[] { false, true })
        {
            using var session = OnnxModelSession.Create(TestModels.Add, SingleThread with { DedicatedRunThreads = dedicated });
            Assert.Equal(dedicated, session.Diagnostics.DedicatedRunThreads);
            SessionTests.Run(session, 3);
        }
    }

    [Fact]
    public void InvokeRunsOnANamedLongLivedThreadAndInlineFromIt()
    {
        string? name = null;
        bool nestedOnRunThread = false;
        int outer = -1, nested = -2;
        RunThreads.Invoke(() =>
        {
            name = Thread.CurrentThread.Name;
            outer = Environment.CurrentManagedThreadId;
            RunThreads.Invoke(() =>
            {
                nestedOnRunThread = RunThreads.IsRunThread;
                nested = Environment.CurrentManagedThreadId;
            });
        });

        Assert.Equal("starlings-onnx-run", name);
        Assert.True(nestedOnRunThread);
        Assert.Equal(outer, nested);
        Assert.False(RunThreads.IsRunThread);
    }

    [Fact]
    public void InvokeRethrowsTheRunsException()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => RunThreads.Invoke(() => throw new InvalidOperationException("boom")));
        Assert.Equal("boom", ex.Message);
    }

    [Fact]
    public void ConcurrentCallersShareABoundedSetOfThreads()
    {
        var threads = new System.Collections.Concurrent.ConcurrentDictionary<int, bool>();
        Parallel.For(0, 200, new ParallelOptions { MaxDegreeOfParallelism = 4 * Environment.ProcessorCount }, _ =>
            RunThreads.Invoke(() =>
            {
                threads[Environment.CurrentManagedThreadId] = true;
                Thread.SpinWait(10_000);
            }));

        Assert.InRange(threads.Count, 1, RunThreads.MaxThreads);
        Assert.InRange(RunThreads.ThreadCount, threads.Count, RunThreads.MaxThreads);
    }

    [Fact]
    public void ConcurrentSessionRunsGiveCorrectResults()
    {
        using var session = OnnxModelSession.Create(TestModels.Add, SingleThread with { DedicatedRunThreads = true });
        Parallel.For(0, 64, i => SessionTests.Run(session, i));
    }

    [Fact]
    public void CancellationStillStopsARunOnADedicatedThread()
    {
        using var session = OnnxModelSession.Create(TestModels.Slow, SingleThread with { DedicatedRunThreads = true });
        using (var cancelled = new CancellationTokenSource())
        {
            cancelled.Cancel();
            Assert.ThrowsAny<OperationCanceledException>(() => RunSlow(session, 64, cancelled.Token));
        }

        // A token cancelled while the run executes sets the terminate flag; the run then throws the caller's token.
        using var cts = new CancellationTokenSource();
        cts.CancelAfter(TimeSpan.FromMilliseconds(5));
        try
        {
            RunSlow(session, 512, cts.Token);
        }
        catch (OperationCanceledException ex)
        {
            Assert.Equal(cts.Token, ex.CancellationToken);
        }

        // The session stays usable either way.
        Assert.Equal(64 * 64, RunSlow(session, 64, CancellationToken.None).Length);
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
}
