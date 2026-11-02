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

    /// <summary>
    /// Capability <c>concurrent_calls</c> over CUDA with batches (issue #257): four threads released together each run
    /// <c>EmbedBatch</c> on the fixture's embedder with a different batch of 4 image or interleaved rows (different image
    /// sizes and token counts), five rounds. Every vector must equal the sequential result of the same batch bit for bit.
    /// Four threads by default, not the text test's eight: on a 16 GB RTX 4060 Ti, eight concurrent image calls (4 or 2
    /// rows each) failed to get device memory from ONNX Runtime's CUDA arena (docs/evidence/cuda-concurrency-2026-11-01/).
    /// <see cref="ConcurrentThreadsVariable"/> sets another count (2 to 16), for example 8 on a 24 GB RTX 3090.
    /// </summary>
    [Fact]
    public void ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit()
    {
        const int BatchRows = 4, Rounds = 5;
        var embedder = fixture.Embedder; // skips when the model is absent
        int threads = ConcurrentThreads();
        var inputs = EmbeddingGemma2ImageTestAssets.Rows
            .Where(r => r.InContract && r.Images.Count > 0 && r.Prompt == "none")
            .Select(r => (r.Id, EmbeddingGemma2ImageTestAssets.BuildInput(r, EmbeddingGemma2ImageTestAssets.Decode)))
            .ToList();
        var batches = ConcurrentMultimodalBatches.Rotations(inputs, threads, BatchRows);
        var (vectors, max) = ConcurrentMultimodalBatches.Check(embedder, batches, Rounds, TestContext.Current.CancellationToken);
        output.WriteLine($"{threads} concurrent EmbedBatch calls x {BatchRows} rows x {Rounds} rounds on one CUDA embedder ({inputs.Count} image and interleaved rows): {vectors} vectors bit-for-bit equal to the sequential run (max abs {max:E3})");
    }

    /// <summary>Thread count of the image concurrency test (unset: 4).</summary>
    public const string ConcurrentThreadsVariable = "STARLING_EG2_CUDA_IMAGE_CONCURRENT_THREADS";

    private static int ConcurrentThreads()
    {
        string? value = Environment.GetEnvironmentVariable(ConcurrentThreadsVariable);
        if (string.IsNullOrEmpty(value))
        {
            return 4;
        }

        Assert.True(
            int.TryParse(value, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int threads) && threads is >= 2 and <= 16,
            $"{ConcurrentThreadsVariable}={value}: expected an integer from 2 to 16");
        return threads;
    }
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
