using Starlings.Models.EmbeddingGemma2;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

public sealed class EmbeddingGemma2ImageCpuFixture() : EmbeddingGemma2ImageModelFixture(new OnnxSessionOptions());

/// <summary>
/// Profile <c>embeddinggemma-2/image-fp32-cpu</c>: the vision graph and the multimodal text graph on ONNX Runtime CPU,
/// from SkiaSharp-decoded fixture images through the C# preprocessing, against the PyTorch reference, with the
/// profile's protected tolerances.yaml. Skips without the graphs or the image set.
/// </summary>
public sealed class ImageParityTests(EmbeddingGemma2ImageCpuFixture fixture, ITestOutputHelper output) : IClassFixture<EmbeddingGemma2ImageCpuFixture>
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("image-fp32-cpu");

    [Fact]
    public void SessionsUseTheLockedGraphs()
    {
        var embedder = fixture.Embedder;
        Assert.Equal(EmbeddingGemma2ModelAssets.VisionModelSha256, embedder.VisionSession.Diagnostics.ModelSha256);
        Assert.Equal(EmbeddingGemma2ModelAssets.MultimodalModelSha256, embedder.MultimodalSession.Diagnostics.ModelSha256);
        Assert.Equal(OnnxRuntimeInfo.CpuProviderName, embedder.VisionSession.Diagnostics.SessionProviders[0]);
    }

    [Fact]
    public void RowsMatchTheReference() => EmbeddingGemma2ImageParityChecks.RowsAgainstReference(fixture, Tolerances, output);

    [Fact]
    public void BatchesMatchSingleRows()
    {
        // Vision batches of up to 8 images, padded to the longest image of the call.
        EmbeddingGemma2ImageParityChecks.BatchesAgainstSingles(fixture, new EmbeddingGemma2MultimodalOptions { MaxVisionBatchSize = 8, MaxVisionBatchPatches = 8 * 2_520, MaxBatchSize = 4 }, Tolerances, output, "vision batch 8, text batch 4");

        // Upstream's layout: every image padded to 2,520 patches.
        EmbeddingGemma2ImageParityChecks.BatchesAgainstSingles(fixture, new EmbeddingGemma2MultimodalOptions { MaxVisionBatchSize = 3, MaxVisionBatchPatches = 3 * 2_520, PadImagesToMaxPatches = true }, Tolerances, output, "padded to 2,520, vision batch 3");
    }

    [Fact]
    public void QueriesRetrieveTheirImages() => EmbeddingGemma2ImageParityChecks.RetrievalSanity(fixture, output);

    [Fact]
    public void TextOnlyInputsUseTheReleasedTextGraph() => EmbeddingGemma2ImageParityChecks.TextOnlyEqualsTheTextEmbedder(fixture);

    [Fact]
    public void MatryoshkaAppliesToImageEmbeddings() => EmbeddingGemma2ImageParityChecks.Matryoshka(fixture, Tolerances);

    [Fact]
    public void CancellationStopsBeforeTheVisionGraph()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var image = EmbeddingGemma2ImageTestAssets.Decode("synth-shapes");
        Assert.ThrowsAny<OperationCanceledException>(() => fixture.Embedder.Embed(image, cts.Token));
    }
}
