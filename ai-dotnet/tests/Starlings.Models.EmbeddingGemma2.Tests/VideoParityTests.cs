using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>CPU fixture: frame batches of 8 (on ONNX Runtime CPU frame batches equal single frames bit for bit).</summary>
public sealed class EmbeddingGemma2VideoCpuFixture() : EmbeddingGemma2VideoModelFixture(
    new OnnxSessionOptions(),
    new EmbeddingGemma2MultimodalOptions { MaxVideoFramesPerVisionCall = 8, MaxVisionBatchPatches = 8 * 1_260 });

/// <summary>
/// Profile <c>embeddinggemma-2/video-fp32-cpu</c>: the vision graph and the multimodal text graph on ONNX Runtime CPU,
/// from adapter-decoded fixture videos through the C# sampling and preprocessing, against the PyTorch reference, with the
/// profile's protected tolerances.yaml. Skips without the graphs, the video set or FFmpeg 8.
/// </summary>
public sealed class VideoParityTests(EmbeddingGemma2VideoCpuFixture fixture, ITestOutputHelper output) : IClassFixture<EmbeddingGemma2VideoCpuFixture>
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("video-fp32-cpu");

    [Fact]
    public void SessionsUseTheLockedGraphs()
    {
        var embedder = fixture.Embedder;
        Assert.Equal("{}", Tolerances["provider_options"]);
        Assert.Equal(EmbeddingGemma2ModelAssets.VisionModelSha256, embedder.VisionSession.Diagnostics.ModelSha256);
        Assert.Equal(EmbeddingGemma2ModelAssets.MultimodalModelSha256, embedder.MultimodalSession.Diagnostics.ModelSha256);
        Assert.Equal(OnnxRuntimeInfo.CpuProviderName, embedder.VisionSession.Diagnostics.SessionProviders[0]);
    }

    [Fact]
    public void RowsMatchTheReference() => EmbeddingGemma2VideoParityChecks.RowsAgainstReference(fixture, Tolerances, output);

    [Fact]
    public void BatchesMatchSingleRows()
    {
        // Text batches of 4, frame calls of the GPU-safe default (2), and upstream's padded layout (1,260 patches per frame).
        EmbeddingGemma2VideoParityChecks.BatchesAgainstSingles(fixture, new EmbeddingGemma2MultimodalOptions { MaxBatchSize = 4 }, Tolerances, output, "frame calls of 2 (default), text batch 4");
        EmbeddingGemma2VideoParityChecks.BatchesAgainstSingles(fixture, new EmbeddingGemma2MultimodalOptions { PadImagesToMaxPatches = true, MaxVideoFramesPerVisionCall = 3, MaxVisionBatchPatches = 3 * 2_520 }, Tolerances, output, "padded to 1,260 per frame, frame calls of 3");
    }

    [Fact]
    public void FramesWithoutTimingNeedTheOptIn() => EmbeddingGemma2VideoParityChecks.FramesWithoutTimingOnOptIn(fixture, Tolerances, output);

    [Fact]
    public void QueriesRetrieveVideosOfTheirScene() => EmbeddingGemma2VideoParityChecks.RetrievalSanity(fixture, output);

    [Fact]
    public void TextOnlyInputsUseTheReleasedTextGraph() => EmbeddingGemma2VideoParityChecks.TextOnlyEqualsTheTextEmbedder(fixture);

    [Fact]
    public void MatryoshkaAppliesToVideoEmbeddings() => EmbeddingGemma2VideoParityChecks.Matryoshka(fixture, Tolerances);

    [Fact]
    public void CancellationStopsBeforeTheVisionGraph()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var video = EmbeddingGemma2VideoTestAssets.Decode("h264-1s-320x240-30fps").Video;
        Assert.ThrowsAny<OperationCanceledException>(() => fixture.Embedder.Embed(video, cts.Token));
    }
}
