using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests.Cuda;

/// <summary>
/// CUDA fixture with the GPU-safe default of 2 frames per vision call (the GPU is shared). The frame-call measurement (W1,
/// <c>tools/measure-eg2-cuda-frame-calls.py</c>) sets <see cref="FramesPerCallVariable"/> to 2, 3 or 4; the package
/// default stays 2 until the maintainer decides (D7).
/// </summary>
public sealed class EmbeddingGemma2VideoCudaFixture() : EmbeddingGemma2VideoModelFixture(
    new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 },
    CreateOptions(),
    CudaGate.Check)
{
    /// <summary>Frames per vision call for this run (unset: the package default).</summary>
    public const string FramesPerCallVariable = "STARLING_EG2_CUDA_FRAMES_PER_CALL";

    private static EmbeddingGemma2MultimodalOptions CreateOptions() =>
        int.TryParse(Environment.GetEnvironmentVariable(FramesPerCallVariable), System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int frames) && frames is >= 1 and <= 8
            ? new EmbeddingGemma2MultimodalOptions { MaxVideoFramesPerVisionCall = frames, MaxVisionBatchPatches = Math.Max(2_520, frames * 1_260) }
            : new EmbeddingGemma2MultimodalOptions();
}

/// <summary>
/// Profile <c>embeddinggemma-2/video-fp32-cuda</c>: the vision and multimodal text graphs on the CUDA execution provider
/// (<c>use_tf32=0</c>, forced by Runtime.Onnx) from adapter-decoded fixture videos, against the CPU PyTorch reference,
/// with the profile's protected tolerances.yaml.
/// </summary>
[Collection("cuda")]
public sealed class CudaVideoParityTests(EmbeddingGemma2VideoCudaFixture fixture, ITestOutputHelper output) : IClassFixture<EmbeddingGemma2VideoCudaFixture>
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("video-fp32-cuda");

    [Fact]
    public void SessionsUseCudaWithTf32Disabled()
    {
        output.WriteLine($"frames per vision call: {fixture.Options.MaxVideoFramesPerVisionCall} (max vision batch patches {fixture.Options.MaxVisionBatchPatches})");
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

    /// <summary>
    /// Capability <c>concurrent_calls</c> over CUDA with batches (issue #257): eight threads released together each run
    /// <c>EmbedBatch</c> on the fixture's embedder with a different batch of 2 video or multimodal rows (different
    /// codecs, sizes, frame counts and token counts), five rounds. Every vector must equal the sequential result of the
    /// same batch bit for bit. The videos are decoded before any thread starts.
    /// </summary>
    [Fact]
    public void ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit()
    {
        const int Threads = 8, BatchRows = 2, Rounds = 5;
        var embedder = fixture.Embedder; // skips when the model, the videos or FFmpeg are absent
        var inputs = EmbeddingGemma2VideoTestAssets.Rows
            .Where(r => r.InDefaultContract && r.Videos.Count > 0 && r.Prompt == "none")
            .Select(r => (r.Id, EmbeddingGemma2VideoTestAssets.BuildInput(r)))
            .ToList();
        var batches = ConcurrentMultimodalBatches.Rotations(inputs, Threads, BatchRows);
        var (vectors, max) = ConcurrentMultimodalBatches.Check(embedder, batches, Rounds, TestContext.Current.CancellationToken);
        output.WriteLine($"{Threads} concurrent EmbedBatch calls x {BatchRows} rows x {Rounds} rounds on one CUDA embedder ({inputs.Count} video and multimodal rows, {fixture.Options.MaxVideoFramesPerVisionCall} frames per vision call): {vectors} vectors bit-for-bit equal to the sequential run (max abs {max:E3})");
    }
}
