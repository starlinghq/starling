using Starlings.Runtime.Onnx;
using Starlings.Runtime.Onnx.Batching;

namespace Starlings.Models.Gliner2;

/// <summary>
/// Immutable extractor configuration. Defaults are the <c>gliner2-base-v1/onnx-fp32-cpu</c> profile values
/// (<c>model-pipeline/registry/gliner2-base-v1/onnx-fp32-cpu/manifest.yaml</c>).
/// </summary>
public sealed record Gliner2ExtractorOptions
{
    /// <summary>The profile's input limit: GLiNER2 tokens per text (schema prefix and words).</summary>
    public const int ProfileMaxSequenceLength = 512;

    /// <summary>
    /// Session settings (provider, threads, digest check). <see cref="OnnxExecutionProvider.Cuda"/> always runs with
    /// TF32 disabled (<c>use_tf32=0</c>), which the profile requires.
    /// </summary>
    public OnnxSessionOptions Session { get; init; } = new();

    /// <summary>Default score threshold (<c>score &gt;= threshold</c>, compared in float32). Default 0.5, GLiNER2's default.</summary>
    public float Threshold { get; init; } = 0.5f;

    /// <summary>
    /// Maximum tokens per text, schema prefix included. Longer texts are rejected with
    /// <see cref="Gliner2InputTooLongException"/>, never truncated (GLiNER2 does not truncate either). Default and
    /// maximum 512, the backbone's <c>max_position_embeddings</c>; inputs above it are outside the profile.
    /// </summary>
    public int MaxSequenceLength { get; init; } = ProfileMaxSequenceLength;

    /// <summary>
    /// Texts per ONNX Runtime call at most. Larger batches run as consecutive padded chunks of this size (see also
    /// <see cref="MaxBatchTokens"/> and <see cref="SortByLength"/>). Default 8,
    /// GLiNER2's default <c>batch_size</c>.
    /// </summary>
    public int MaxBatchSize { get; init; } = 8;

    /// <summary>
    /// Padded token cells (texts x longest text, schema prefix included) per ONNX Runtime call at most, or
    /// <see langword="null"/> (default) for no token budget. A chunk closes before it would exceed the budget; a single
    /// text longer than the budget runs alone. Chunks are never larger than <see cref="MaxBatchSize"/> texts.
    /// Experimental (capability <c>batch.token_budget</c>).
    /// </summary>
    public int? MaxBatchTokens { get; init; }

    /// <summary>
    /// When <see langword="true"/>, texts are ordered by ascending token count (ties in input order) before they are
    /// chunked, so each call pads less, and results are returned in input order. Logits then depend on which texts
    /// share a call and differ from the default within the profile's padded-batch-vs-single bound, not bit for bit.
    /// Default <see langword="false"/>: chunks are consecutive texts in input order, as qualified. Experimental
    /// (capability <c>batch.sort_by_length</c>).
    /// </summary>
    public bool SortByLength { get; init; }

    /// <summary>
    /// Whether graph calls from concurrent requests run one at a time. Null (default) serializes them on CUDA, where
    /// concurrent runs on one session returned wrong entities (capability <c>concurrent_calls.cuda.unserialized</c>:
    /// unsupported), and lets them run concurrently on CPU (tested equal to sequential runs).
    /// </summary>
    public bool? SerializeNativeCalls { get; init; }

    /// <summary>
    /// Cross-request micro-batching for <see cref="Gliner2EntityExtractor.ExtractAsync(string, IReadOnlyList{string}, IReadOnlyDictionary{string, string}?, float?, CancellationToken)"/>
    /// (Experimental, capability <c>micro_batching</c>), or null (default): off, <c>ExtractAsync</c> runs on the calling
    /// thread like <see cref="Gliner2EntityExtractor.Extract"/>. When set, <c>ExtractAsync</c> prepares and validates the
    /// text on the calling thread, then queues it; one dispatcher thread runs the queued texts of concurrent callers
    /// that share a schema (the same labels in the same order and the same descriptions) in padded batches
    /// (<see cref="MicroBatcher{TItem, TResult}"/>; <see cref="MaxBatchSize"/> still splits each batch into graph
    /// calls), and each caller decodes its own spans with its own threshold. The synchronous methods never queue. A
    /// queued text's logits then depend on the other texts in its batch: equal to a single call within the profile's
    /// padded-batch tolerance, not bit for bit reproducible.
    /// </summary>
    public MicroBatchingOptions? MicroBatching { get; init; }

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Session);
        ValidateThreshold(Threshold, nameof(Threshold));
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxSequenceLength, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(MaxSequenceLength, ProfileMaxSequenceLength);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxBatchSize, 1);
        if (MaxBatchTokens is { } maxBatchTokens)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchTokens, 1, nameof(MaxBatchTokens));
        }

        MicroBatching?.Validate();
    }

    internal static void ValidateThreshold(float threshold, string name)
    {
        if (float.IsNaN(threshold) || threshold < 0f || threshold > 1f)
        {
            throw new ArgumentOutOfRangeException(name, threshold, "The threshold must be in [0, 1].");
        }
    }
}

/// <summary>
/// A text needs more tokens than the profile allows. Nothing is truncated; split the text or shorten it. An
/// <see cref="InputTooLongException"/>, and through it an <see cref="ArgumentException"/>: <see cref="InputTooLongException.InputIndex"/>
/// is the offending text's index in the call (0 for single-text calls), <see cref="InputTooLongException.TokenCount"/>
/// the tokens it needs, schema prefix included, and <see cref="InputTooLongException.MaxTokens"/> the limit in effect
/// (<see cref="Gliner2ExtractorOptions.MaxSequenceLength"/>).
/// </summary>
public sealed class Gliner2InputTooLongException : InputTooLongException
{
    /// <summary>Creates the exception.</summary>
    public Gliner2InputTooLongException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public Gliner2InputTooLongException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and an inner exception.</summary>
    public Gliner2InputTooLongException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
