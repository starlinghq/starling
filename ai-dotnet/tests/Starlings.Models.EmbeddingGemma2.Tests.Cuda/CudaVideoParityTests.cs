using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests.Cuda;

/// <summary>CUDA fixture with the GPU-safe default of 2 frames per vision call (the GPU is shared).</summary>
public sealed class EmbeddingGemma2VideoCudaFixture() : EmbeddingGemma2VideoModelFixture(
    new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 },
    new EmbeddingGemma2MultimodalOptions(),
    CudaGate.Check);

/// <summary>
/// Profile <c>embeddinggemma-2/video-fp32-cuda</c>: the vision and multimodal text graphs on the CUDA execution provider
/// (<c>use_tf32=0</c>, forced by Runtime.Onnx) from adapter-decoded fixture videos, against the CPU PyTorch reference,
/// with the profile's protected tolerances.yaml.
/// </summary>
public sealed class CudaVideoParityTests(EmbeddingGemma2VideoCudaFixture fixture, ITestOutputHelper output) : IClassFixture<EmbeddingGemma2VideoCudaFixture>
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("video-fp32-cuda");

    [Fact]
    public void SessionsUseCudaWithTf32Disabled()
    {
        Assert.Equal("{ use_tf32: 0 }", Tolerances["provider_options"]);
        foreach (var session in new[] { fixture.Embedder.VisionSession, fixture.Embedder.MultimodalSession })
        {
            var d = session.Diagnostics;
            output.WriteLine($"{d.ModelSha256?[..12]}: {string.Join(", ", d.SessionProviders)}; use_tf32={d.ProviderOptions["use_tf32"]}; CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")}");
            Assert.Equal(OnnxRuntimeInfo.CudaProviderName, d.SessionProviders[0]);
            Assert.Equal("0", d.ProviderOptions["use_tf32"]);
        }
    }

    [Fact]
    public void RowsMatchTheReference() => EmbeddingGemma2VideoParityChecks.RowsAgainstReference(fixture, Tolerances, output);

    [Fact]
    public void BatchesMatchSingleRows() =>
        EmbeddingGemma2VideoParityChecks.BatchesAgainstSingles(fixture, new EmbeddingGemma2MultimodalOptions { MaxBatchSize = 4, MaxVideoFramesPerVisionCall = 4, MaxVisionBatchPatches = 4 * 1_260 }, Tolerances, output, "frame calls of 4, text batch 4");

    [Fact]
    public void FramesWithoutTimingNeedTheOptIn() => EmbeddingGemma2VideoParityChecks.FramesWithoutTimingOnOptIn(fixture, Tolerances, output);

    [Fact]
    public void QueriesRetrieveVideosOfTheirScene() => EmbeddingGemma2VideoParityChecks.RetrievalSanity(fixture, output);

    [Fact]
    public void TextOnlyInputsUseTheReleasedTextGraph() => EmbeddingGemma2VideoParityChecks.TextOnlyEqualsTheTextEmbedder(fixture);
}
