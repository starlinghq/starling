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
    /// Maximum tokens per sequence, special tokens included. Longer inputs are rejected, not truncated: truncation
    /// (right, per the profile) is the tokenizer's job. Default 512.
    /// </summary>
    public int MaxSequenceLength { get; init; } = 512;

    /// <summary>Id used to right-pad shorter rows in a batch (XLM-R <c>&lt;pad&gt;</c> = 1). Default 1.</summary>
    public long PadTokenId { get; init; } = 1;

    /// <summary>Exclusive upper bound for token ids (e5 vocabulary size 250037).</summary>
    public long VocabularySize { get; init; } = 250_037;

    /// <summary>
    /// Rows per ONNX Runtime call. Larger batches run as consecutive chunks of this size, each padded to its own
    /// longest row. Bounds temporary memory: hidden states take rows x width x 384 x 4 bytes. Default 32.
    /// </summary>
    public int MaxBatchSize { get; init; } = 32;

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Session);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxSequenceLength, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxBatchSize, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(PadTokenId);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(VocabularySize, PadTokenId);
    }
}
