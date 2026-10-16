using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// Dedicated run threads (this library's CPU default, docs/evidence/e5-soak-reliability/) change which thread calls
/// ONNX Runtime, not what it computes.
/// </summary>
public class RunThreadsTests(ITestOutputHelper output)
{
    [Fact]
    public void CpuSessionsUseDedicatedRunThreadsUnlessTheCallerChose()
    {
        Assert.True(E5Encoder.SessionDefaults(new OnnxSessionOptions()).DedicatedRunThreads);
        Assert.False(E5Encoder.SessionDefaults(new OnnxSessionOptions { DedicatedRunThreads = false }).DedicatedRunThreads);
        Assert.True(E5Encoder.SessionDefaults(new OnnxSessionOptions { DedicatedRunThreads = true }).DedicatedRunThreads);

        // The arena keeps ONNX Runtime's default, or the caller's choice.
        Assert.Null(E5Encoder.SessionDefaults(new OnnxSessionOptions()).CpuMemoryArena);
        Assert.False(E5Encoder.SessionDefaults(new OnnxSessionOptions { CpuMemoryArena = false }).CpuMemoryArena);

        // CUDA sessions run on the calling thread.
        Assert.Null(E5Encoder.SessionDefaults(new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda }).DedicatedRunThreads);
    }

    [Fact]
    public void EmbeddingsAreBitIdenticalOnDedicatedRunThreadsAndOnTheCaller()
    {
        var (path, reason) = E5TestAssets.FindModel();
        Assert.SkipWhen(path is null, reason);
        var inputs = E5Fixtures.Cases.Select(c => new EncoderInput(c.InputIds)).ToArray();

        float[][] Embed(bool? dedicated)
        {
            using var encoder = E5Encoder.Create(path!, new E5EncoderOptions
            {
                Session = new OnnxSessionOptions { ComputeModelSha256 = false, DedicatedRunThreads = dedicated },
            });
            Assert.Equal(dedicated ?? true, encoder.Session.Diagnostics.DedicatedRunThreads);
            var single = inputs.Select(i => encoder.Embed(i, TestContext.Current.CancellationToken));
            var batch = encoder.EmbedBatch(inputs, TestContext.Current.CancellationToken);
            return [.. single, .. batch];
        }

        var byDefault = Embed(null);
        var on = Embed(true);
        var off = Embed(false);
        for (int i = 0; i < on.Length; i++)
        {
            Assert.Equal(off[i], on[i]);
            Assert.Equal(off[i], byDefault[i]);
        }

        output.WriteLine($"{inputs.Length} cases, single and batched, bit-identical on dedicated run threads (the default) and on the calling thread");
    }
}
