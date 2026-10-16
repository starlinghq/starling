using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests.Cuda;

public sealed class EmbeddingGemma2ImageCudaFixture() : EmbeddingGemma2ImageModelFixture(
    new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 },
    CudaGate.Check);

/// <summary>
/// Profile <c>embeddinggemma-2/image-fp32-cuda</c>: the vision and multimodal text graphs on the CUDA execution provider
/// (<c>use_tf32=0</c>, forced by Runtime.Onnx) from SkiaSharp-decoded fixture images, against the CPU PyTorch reference,
/// with the profile's protected tolerances.yaml. The GPU is shared: one image per vision call (the default).
/// </summary>
[Collection("cuda")]
public sealed class CudaImageParityTests(EmbeddingGemma2ImageCudaFixture fixture, ITestOutputHelper output) : IClassFixture<EmbeddingGemma2ImageCudaFixture>
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("image-fp32-cuda");

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
    public void RowsMatchTheReference() => EmbeddingGemma2ImageParityChecks.RowsAgainstReference(fixture, Tolerances, output);

    [Fact]
    public void BatchesMatchSingleRows() =>
        EmbeddingGemma2ImageParityChecks.BatchesAgainstSingles(fixture, new EmbeddingGemma2MultimodalOptions { MaxBatchSize = 4 }, Tolerances, output, "vision batch 1 (GPU default), text batch 4");

    [Fact]
    public void QueriesRetrieveTheirImages() => EmbeddingGemma2ImageParityChecks.RetrievalSanity(fixture, output);

    [Fact]
    public void TextOnlyInputsUseTheReleasedTextGraph() => EmbeddingGemma2ImageParityChecks.TextOnlyEqualsTheTextEmbedder(fixture);
}

/// <summary>
/// Capability <c>image_budget_other</c> on CUDA (proposed rows, protected bounds of <c>image-fp32-cuda</c>), in a class of
/// its own with no class fixture: the 1,120-token budget (10,080 patches) needs about 4.6 GB for one attention buffer,
/// which ONNX Runtime's CUDA arena could not get next to the image fixture's sessions. In the serial "cuda" collection no
/// other class's sessions are loaded either, and one embedder per budget is loaded at a time.
/// </summary>
[Collection("cuda")]
public sealed class CudaImageBudgetTests(ITestOutputHelper output)
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("image-fp32-cuda");

    [Fact]
    public void OtherBudgetsMatchTheReference()
    {
        string? gate = CudaGate.Check();
        Assert.SkipWhen(gate is not null, gate ?? string.Empty);
        Assert.SkipWhen(EmbeddingGemma2ImageTestAssets.ImagesSkipReason is not null, EmbeddingGemma2ImageTestAssets.ImagesSkipReason ?? string.Empty);
        var (files, reason) = EmbeddingGemma2ImageTestAssets.FindImageModel();
        Assert.SkipWhen(files is null, reason);
        EmbeddingGemma2ImageParityChecks.OtherBudgets(files!, new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 }, Tolerances, output);
    }
}
