using Starlings.Runtime.Onnx;

namespace Starlings.Models.E5;

/// <summary>
/// Immutable encoder configuration. Defaults are the <c>multilingual-e5-small</c> profile values
/// (<c>model-pipeline/registry/multilingual-e5-small/*/manifest.yaml</c>); changing them leaves the qualified profile.
/// </summary>
public sealed record E5EncoderOptions
{
    /// <summary>Session settings (provider, threads, digest check).</summary>
    public OnnxSessionOptions Session { get; init; } = new();

    /// <summary>
    /// Maximum tokens per sequence, special tokens included. The encoder rejects longer id sequences with
    /// <see cref="ArgumentException"/> and never truncates them; text is truncated on the right to this length by
    /// <see cref="E5Tokenizer"/> (the profile's <c>truncate-right</c>, as the reference does). Default
    /// <see cref="E5Encoder.MaxSupportedSequenceLength"/> (512).
    /// </summary>
    public int MaxSequenceLength { get; init; } = E5Encoder.MaxSupportedSequenceLength;

    /// <summary>Id used to right-pad shorter rows in a batch (XLM-R <c>&lt;pad&gt;</c> = 1). Default 1.</summary>
    public long PadTokenId { get; init; } = 1;

    /// <summary>Exclusive upper bound for token ids (e5 vocabulary size 250037).</summary>
    public long VocabularySize { get; init; } = 250_037;

    /// <summary>
    /// Rows per ONNX Runtime call at most. Larger batches run as consecutive chunks of this size (see also
    /// <see cref="MaxBatchTokens"/> and <see cref="SortByLength"/>), each padded to its own longest row.
    /// Bounds temporary memory: hidden states take rows x width x 384 x 4 bytes. Default 32.
    /// </summary>
    public int MaxBatchSize { get; init; } = 32;

    /// <summary>
    /// Padded cells (rows x longest row) per ONNX Runtime call at most, or <see langword="null"/> (default) for no
    /// token budget. A chunk closes before it would exceed the budget; a single row longer than the budget runs alone.
    /// Bounds temporary memory and attention work for long inputs. Chunks are never larger than
    /// <see cref="MaxBatchSize"/> rows. Experimental (capability <c>batch.token_budget</c>).
    /// </summary>
    public int? MaxBatchTokens { get; init; }

    /// <summary>
    /// When <see langword="true"/>, rows are ordered by ascending token length (ties in input order) before they are
    /// chunked, so each call pads less, and results are scattered back to input order. Outputs then depend on which
    /// rows share a call and differ from the default within the profile's padded-batch-vs-single bound, not bit for
    /// bit. Default <see langword="false"/>: chunks are consecutive rows in input order, as qualified. Experimental
    /// (capability <c>batch.sort_by_length</c>).
    /// </summary>
    public bool SortByLength { get; init; }

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Session);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxSequenceLength, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxBatchSize, 1);
        if (MaxBatchTokens is { } maxBatchTokens)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchTokens, 1, nameof(MaxBatchTokens));
        }
        ArgumentOutOfRangeException.ThrowIfNegative(PadTokenId);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(VocabularySize, PadTokenId);
    }
}
