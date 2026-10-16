using System.Diagnostics;
using System.Globalization;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// Tests that measure process-wide memory. They run in a collection with parallelization disabled, after the parallel
/// ones, so other classes cannot move the process's private memory during the measurement.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ProcessMemoryTestGroup
{
    public const string Name = "process memory (not parallel)";
}

/// <summary>Profile <c>embeddinggemma-2/onnx-fp32-cpu</c>: repeated load and dispose of the real 1.08 GB graph.</summary>
[Collection(ProcessMemoryTestGroup.Name)]
public class ModelMemoryTests(ITestOutputHelper output)
{
    [Fact]
    public void RepeatedLoadAndDisposeReleasesTheModel()
    {
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        var options = new EmbeddingGemma2EncoderOptions { Session = new OnnxSessionOptions { ComputeModelSha256 = false } };
        var input = new EmbeddingGemma2Input(EmbeddingGemma2Fixtures.LoadRows()[6].InputIds);
        var samples = new List<double>();
        for (int i = 0; i < 5; i++)
        {
            using (var encoder = EmbeddingGemma2Encoder.Create(files!.ModelPath, options))
            {
                encoder.Embed(input, TestContext.Current.CancellationToken);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            using var process = Process.GetCurrentProcess();
            samples.Add(process.PrivateMemorySize64 / 1048576.0);
        }

        output.WriteLine("private memory (MB) after each load/embed/dispose cycle: " + string.Join(", ", samples.Select(s => s.ToString("F0", CultureInfo.InvariantCulture))));

        // Each load holds a 1.08 GB graph; a leaked session would add about that much per cycle.
        Assert.True(samples[^1] - samples[1] < 400, "private memory grew across load/dispose cycles");
    }
}

/// <summary>
/// The library's session defaults (docs/evidence/eg2-hosted-soak/, docs/evidence/eg2-soak-reliability/): CPU sessions run
/// without ONNX Runtime's CPU arena, on the runtime's dedicated run threads.
/// </summary>
public class SessionDefaultTests
{
    [Fact]
    public void CpuSessionsTurnTheArenaOffUnlessTheCallerChose()
    {
        Assert.False(EmbeddingGemma2Encoder.SessionDefaults(new OnnxSessionOptions()).CpuMemoryArena);
        Assert.True(EmbeddingGemma2Encoder.SessionDefaults(new OnnxSessionOptions { CpuMemoryArena = true }).CpuMemoryArena);
        Assert.False(EmbeddingGemma2Encoder.SessionDefaults(new OnnxSessionOptions { CpuMemoryArena = false }).CpuMemoryArena);

        // CUDA sessions keep ONNX Runtime's default.
        Assert.Null(EmbeddingGemma2Encoder.SessionDefaults(new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda }).CpuMemoryArena);
    }

    [Fact]
    public void CpuSessionsUseDedicatedRunThreadsUnlessTheCallerChose()
    {
        Assert.True(EmbeddingGemma2Encoder.SessionDefaults(new OnnxSessionOptions()).DedicatedRunThreads);
        Assert.False(EmbeddingGemma2Encoder.SessionDefaults(new OnnxSessionOptions { DedicatedRunThreads = false }).DedicatedRunThreads);
        Assert.True(EmbeddingGemma2Encoder.SessionDefaults(new OnnxSessionOptions { DedicatedRunThreads = true }).DedicatedRunThreads);

        // The caller's arena choice is kept alongside.
        var both = EmbeddingGemma2Encoder.SessionDefaults(new OnnxSessionOptions { CpuMemoryArena = true });
        Assert.True(both.CpuMemoryArena);
        Assert.True(both.DedicatedRunThreads);

        // CUDA sessions run on the calling thread.
        Assert.Null(EmbeddingGemma2Encoder.SessionDefaults(new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda }).DedicatedRunThreads);
    }

    [Fact]
    public void TheTextEncoderSessionReportsTheArenaOff()
    {
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        using var encoder = EmbeddingGemma2Encoder.Create(files!.ModelPath, new EmbeddingGemma2EncoderOptions { Session = new OnnxSessionOptions { ComputeModelSha256 = false } });
        Assert.False(encoder.Session.Diagnostics.CpuMemoryArena);
        Assert.True(encoder.Session.Diagnostics.DedicatedRunThreads);
    }
}
