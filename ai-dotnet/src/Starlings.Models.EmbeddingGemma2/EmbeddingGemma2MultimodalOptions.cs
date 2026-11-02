using Starlings.Runtime.Onnx;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// Immutable configuration of <see cref="EmbeddingGemma2MultimodalEmbedder"/>. Defaults are the
/// <c>embeddinggemma-2/image-fp32-cpu</c> and <c>image-fp32-cuda</c> profile values and a GPU-safe vision batch.
/// </summary>
public sealed record EmbeddingGemma2MultimodalOptions
{
    /// <summary>
    /// Session settings for all three graphs (provider, threads). CUDA always runs with TF32 disabled
    /// (<c>use_tf32=0</c>, enforced by <c>Starlings.Runtime.Onnx</c>): with TF32 the image embeddings drift by
    /// up to 1.4e-3 and fail the profile's bound. <see cref="OnnxSessionOptions.ExpectedModelSha256"/> is ignored here;
    /// see <see cref="VerifyModelSha256"/>. <see cref="OnnxSessionOptions.KnownModelSha256"/> is ignored too: each graph
    /// takes its own digest from the files when their snapshot hashed them.
    /// </summary>
    public OnnxSessionOptions Session { get; init; } = new();

    /// <summary>
    /// When <see langword="true"/> (default), <see cref="EmbeddingGemma2MultimodalEmbedder.Create(EmbeddingGemma2ImageModelFiles, EmbeddingGemma2MultimodalOptions?)"/>
    /// checks each graph's SHA-256 against the asset lock while loading it (reads each file once more).
    /// </summary>
    public bool VerifyModelSha256 { get; init; } = true;

    /// <summary>Output dimension: 768 (native) or a Matryoshka size, 512, 256 or 128. Default 768.</summary>
    public int Dimension { get; init; } = EmbeddingGemma2Encoder.NativeDimension;

    /// <summary>
    /// Maximum tokens per input after image expansion (each image adds its soft tokens plus two boundary tokens),
    /// <c>&lt;bos&gt;</c>, <c>&lt;eos&gt;</c> and prompt included. Longer inputs are rejected with
    /// <see cref="EmbeddingGemma2InputTooLongException"/> before any inference. At most 8,192 (the card's context: about
    /// 29 images at the default budget). Default 8,192.
    /// </summary>
    public int MaxSequenceLength { get; init; } = EmbeddingGemma2Encoder.MaxSupportedSequenceLength;

    /// <summary>Rows per text-graph call at most (text-only and image inputs are batched separately). Default 16.</summary>
    public int MaxBatchSize { get; init; } = 16;

    /// <summary>Padded cells (rows x longest row) per text-graph call at most. Default 8,192.</summary>
    public int MaxBatchTokens { get; init; } = 8_192;

    /// <summary>
    /// For text-only inputs (the released text graph): order rows by ascending token length before chunking and
    /// scatter results back to input order, as <see cref="EmbeddingGemma2EncoderOptions.SortByLength"/>. Inputs with
    /// images or videos keep input order. Applies to the text encoder that
    /// <see cref="EmbeddingGemma2MultimodalEmbedder.Create(EmbeddingGemma2ImageModelFiles, EmbeddingGemma2MultimodalOptions?)"/>
    /// creates; an encoder passed to the constructor keeps its own options. Default <see langword="false"/>.
    /// Experimental (capability <c>batch.sort_by_length</c>).
    /// </summary>
    public bool SortByLength { get; init; }

    /// <summary>
    /// Soft tokens per image at most (upstream <c>max_soft_tokens</c>): 280 is the checkpoint's default; 70, 140, 560 and
    /// 1,120 are accepted by upstream and experimental here (proposed rows, within the protected bounds). Default 280.
    /// </summary>
    public int MaxImageSoftTokens { get; init; } = EmbeddingGemma2ImageProcessor.DefaultMaxSoftTokens;

    /// <summary>
    /// Decoded pixels per image at most, checked before preprocessing (memory bound; upstream has none). Larger images
    /// throw <see cref="EmbeddingGemma2ImageTooLargeException"/>. Default 50,000,000 (50 megapixels: 150 MB of RGB).
    /// </summary>
    public long MaxImagePixels { get; init; } = 50_000_000;

    /// <summary>
    /// Images per vision-graph call at most. Default 1, which is safe on a GPU with about 5.5 GB free: one image of
    /// 2,520 patches needs a 610 MB attention buffer per layer on ONNX Runtime CUDA, and a batch of two failed with
    /// 5.5 GB free. On CPU, vision batches equal single images bit for bit, so raising this only trades memory for
    /// throughput.
    /// </summary>
    public int MaxVisionBatchSize { get; init; } = 1;

    /// <summary>
    /// Padded patch rows (images x longest image in the call) per vision-graph call at most; a call closes before it
    /// would exceed this, and an image larger than the budget runs alone. Default 2,520 (one full-budget image).
    /// </summary>
    public int MaxVisionBatchPatches { get; init; } = 2_520;

    /// <summary>
    /// When set, the embedder chooses <see cref="MaxVisionBatchSize"/>, <see cref="MaxVisionBatchPatches"/> and
    /// <see cref="MaxVideoFramesPerVisionCall"/> from the free memory of the vision graph's device when it loads (host RAM
    /// on CPU, device memory on CUDA), with the safety margin of <see cref="EmbeddingGemma2AutoBatchLimits"/>. The values
    /// configured here are floors; the limits chosen are in <see cref="EmbeddingGemma2MultimodalEmbedder.Options"/>.
    /// Default <see langword="null"/>: off, the configured limits apply unchanged. Experimental (capability
    /// <c>image.auto_batch_limits</c>).
    /// </summary>
    public EmbeddingGemma2AutoBatchLimits? AutoBatchLimits { get; init; }

    /// <summary>
    /// When <see langword="true"/>, every image is padded to the full patch budget (2,520 rows at 280 soft tokens) as
    /// upstream's processor does. Default <see langword="false"/>: each vision call is padded only to its longest image
    /// (on ONNX Runtime CPU the soft tokens are bit-identical; on CUDA within 3.4e-5, inside the soft-token bound),
    /// which saves vision time for wide, tall and small batches.
    /// </summary>
    public bool PadImagesToMaxPatches { get; init; }

    /// <summary>
    /// Video frames per vision-graph call at most (frames of one or more videos of the same text chunk; images run in
    /// calls of their own). Each frame is at most 1,260 patches (140 soft tokens); a call also stays within
    /// <see cref="MaxVisionBatchPatches"/>. Default 2 (2 x 1,260 = 2,520 patches, the padded size of one full-budget
    /// image): safe on a GPU with about 5 GB free. On ONNX Runtime CPU frame batches equal single frames bit for bit, so
    /// raising this (with <see cref="MaxVisionBatchPatches"/>, for example 8 and 8 x 1,260) only trades memory for
    /// throughput. A 32-frame video is 16 calls at the default.
    /// </summary>
    public int MaxVideoFramesPerVisionCall { get; init; } = 2;

    /// <summary>
    /// When <see langword="true"/>, a video built with <see cref="EmbeddingGemma2Video.FromFramesWithoutTiming"/> is embedded
    /// with upstream's behaviour for a decoded array without metadata: every frame is a candidate and more than 32 are
    /// thinned to 32. That gives a different embedding than the same file, which upstream samples at 1 frame per second
    /// (cosine 0.965 on the 5 s fixture clip). Default <see langword="false"/>: such videos throw
    /// <see cref="EmbeddingGemma2VideoTimingRequiredException"/> before any inference.
    /// </summary>
    public bool AllowVideosWithoutTiming { get; init; }

    /// <summary>Encoder options of the released text graph for text-only inputs.</summary>
    internal EmbeddingGemma2EncoderOptions ToTextEncoderOptions(string? expectedSha256, string? knownSha256 = null) => new()
    {
        Session = Session with { ExpectedModelSha256 = expectedSha256, KnownModelSha256 = knownSha256 },
        Dimension = Dimension,
        MaxSequenceLength = MaxSequenceLength,
        MaxBatchSize = MaxBatchSize,
        MaxBatchTokens = MaxBatchTokens,
        SortByLength = SortByLength,
    };

    /// <summary>
    /// Reuse the large per-request buffers across requests (the default). False allocates them per request and keeps
    /// none; tests compare the two.
    /// </summary>
    internal bool PoolBuffers { get; init; } = true;

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Session);
        if (!EmbeddingGemma2Encoder.IsSupportedDimension(Dimension))
        {
            throw new ArgumentOutOfRangeException(nameof(Dimension), Dimension, "Supported dimensions are 768, 512, 256 and 128 (Matryoshka truncation).");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(MaxSequenceLength, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(MaxSequenceLength, EmbeddingGemma2Encoder.MaxSupportedSequenceLength);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxBatchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxBatchTokens, 1);
        EmbeddingGemma2ImageProcessor.ValidateBudget(MaxImageSoftTokens);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxImagePixels, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxVisionBatchSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxVisionBatchPatches, 9);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxVideoFramesPerVisionCall, 1);
        AutoBatchLimits?.Validate();
    }
}

/// <summary>
/// An image has more decoded pixels than allowed (<see cref="EmbeddingGemma2MultimodalOptions.MaxImagePixels"/>, or
/// the decoder's limit). Nothing was embedded.
/// </summary>
public sealed class EmbeddingGemma2ImageTooLargeException : ArgumentException
{
    /// <summary>Creates the exception.</summary>
    public EmbeddingGemma2ImageTooLargeException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public EmbeddingGemma2ImageTooLargeException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public EmbeddingGemma2ImageTooLargeException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Creates the exception for an image of <paramref name="width"/> x <paramref name="height"/>.</summary>
    public EmbeddingGemma2ImageTooLargeException(int width, int height, long maximumPixels)
        : base($"The image is {width}x{height} ({(long)width * height} pixels); the limit is {maximumPixels} decoded pixels. Downscale it first or raise the limit.")
    {
        Width = width;
        Height = height;
        MaximumPixels = maximumPixels;
    }

    /// <summary>Width of the rejected image.</summary>
    public int Width { get; init; }

    /// <summary>Height of the rejected image.</summary>
    public int Height { get; init; }

    /// <summary>The limit in effect.</summary>
    public long MaximumPixels { get; init; }
}
