using Starlings.Runtime.Onnx;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// Immutable encoder configuration. Defaults are the <c>embeddinggemma-2</c> profile values
/// (<c>model-pipeline/registry/embeddinggemma-2/*/manifest.yaml</c>).
/// </summary>
public sealed record EmbeddingGemma2EncoderOptions
{
    /// <summary>
    /// Session settings (provider, threads, digest check). CUDA always runs with TF32 disabled (<c>use_tf32=0</c>,
    /// enforced by <c>Starlings.Runtime.Onnx</c>): with TF32 the embedding moves by up to 1.9e-4 and fails the
    /// profile's bound.
    /// </summary>
    public OnnxSessionOptions Session { get; init; } = new();

    /// <summary>
    /// Output dimension: 768 (native) or a Matryoshka size, 512, 256 or 128. A smaller size keeps the leading
    /// dimensions of the pooled embedding and L2-normalizes them again, as sentence-transformers <c>truncate_dim</c>
    /// does. Any other value is rejected. Default 768.
    /// </summary>
    public int Dimension { get; init; } = EmbeddingGemma2Encoder.NativeDimension;

    /// <summary>
    /// Maximum tokens per sequence, <c>&lt;bos&gt;</c>, <c>&lt;eos&gt;</c> and prompt included. Longer inputs are
    /// rejected with <see cref="EmbeddingGemma2InputTooLongException"/>, never truncated. May be lowered, not raised
    /// above <see cref="EmbeddingGemma2Encoder.MaxSupportedSequenceLength"/> (8,192, the card's validated context).
    /// Default 8,192.
    /// </summary>
    public int MaxSequenceLength { get; init; } = EmbeddingGemma2Encoder.MaxSupportedSequenceLength;

    /// <summary>
    /// Rows per ONNX Runtime call at most. Batches run as consecutive chunks in input order, each right-padded to its
    /// own longest row. Default 16.
    /// </summary>
    public int MaxBatchSize { get; init; } = 16;

    /// <summary>
    /// Padded cells (rows x longest row) per ONNX Runtime call at most; a chunk closes before it would exceed this.
    /// A single row longer than the budget runs alone. Bounds temporary memory, which grows with rows x width for the
    /// token embeddings (768 floats per cell) and with rows x width x width for the full-attention layers' masks and
    /// scores. Default 8,192.
    /// </summary>
    public int MaxBatchTokens { get; init; } = 8_192;

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
    }
}

/// <summary>
/// An input has more tokens than the encoder accepts (<see cref="EmbeddingGemma2EncoderOptions.MaxSequenceLength"/>,
/// at most 8,192 including <c>&lt;bos&gt;</c>, <c>&lt;eos&gt;</c> and the prompt). The input is rejected, not
/// truncated; nothing was embedded.
/// </summary>
public sealed class EmbeddingGemma2InputTooLongException : ArgumentException
{
    /// <summary>Creates the exception.</summary>
    public EmbeddingGemma2InputTooLongException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public EmbeddingGemma2InputTooLongException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public EmbeddingGemma2InputTooLongException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Creates the exception for input <paramref name="inputIndex"/> with a descriptive message.</summary>
    public EmbeddingGemma2InputTooLongException(int inputIndex, int tokenCount, int maximumTokens)
        : base($"Input {inputIndex} has {tokenCount} tokens; the maximum is {maximumTokens} (special tokens and prompt included). EmbeddingGemma 2 rejects longer input instead of truncating it; split or shorten the text.", "inputs")
    {
        InputIndex = inputIndex;
        TokenCount = tokenCount;
        MaximumTokens = maximumTokens;
    }

    /// <summary>Position of the rejected input in the batch.</summary>
    public int InputIndex { get; init; }

    /// <summary>Tokens in the rejected input.</summary>
    public int TokenCount { get; init; }

    /// <summary>The limit in effect.</summary>
    public int MaximumTokens { get; init; }
}
