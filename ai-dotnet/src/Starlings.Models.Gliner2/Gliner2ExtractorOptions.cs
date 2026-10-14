using Starlings.Runtime.Onnx;

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
    /// Texts per ONNX Runtime call. Larger batches run as consecutive padded chunks of this size. Default 8,
    /// GLiNER2's default <c>batch_size</c>.
    /// </summary>
    public int MaxBatchSize { get; init; } = 8;

    /// <summary>
    /// Whether graph calls from concurrent requests run one at a time. Null (default) serializes them on CUDA, where
    /// concurrent runs on one session returned wrong entities (capability <c>concurrent_calls.cuda.unserialized</c>:
    /// unsupported), and lets them run concurrently on CPU (tested equal to sequential runs).
    /// </summary>
    public bool? SerializeNativeCalls { get; init; }

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Session);
        ValidateThreshold(Threshold, nameof(Threshold));
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxSequenceLength, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(MaxSequenceLength, ProfileMaxSequenceLength);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxBatchSize, 1);
    }

    internal static void ValidateThreshold(float threshold, string name)
    {
        if (float.IsNaN(threshold) || threshold < 0f || threshold > 1f)
        {
            throw new ArgumentOutOfRangeException(name, threshold, "The threshold must be in [0, 1].");
        }
    }
}

/// <summary>A text needs more tokens than the profile allows. Nothing is truncated; split the text or shorten it.</summary>
public sealed class Gliner2InputTooLongException : ArgumentException
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

    /// <summary>Index of the offending text in the call (0 for single-text calls).</summary>
    public int InputIndex { get; init; }

    /// <summary>Tokens the text needs, schema prefix included.</summary>
    public int TokenCount { get; init; }

    /// <summary>The limit in effect (<see cref="Gliner2ExtractorOptions.MaxSequenceLength"/>).</summary>
    public int MaxTokens { get; init; }
}
