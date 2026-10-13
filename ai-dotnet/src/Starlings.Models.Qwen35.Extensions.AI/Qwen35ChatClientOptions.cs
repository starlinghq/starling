namespace Starlings.Models.Qwen35.Extensions.AI;

/// <summary>Immutable configuration of a <see cref="Qwen35ChatClient"/>.</summary>
public sealed record Qwen35ChatClientOptions
{
    /// <summary>Default <see cref="StreamingBufferCapacity"/>.</summary>
    public const int DefaultStreamingBufferCapacity = 16;

    /// <summary>
    /// Request settings used when a request's <c>ChatOptions</c> does not set them: <see cref="Qwen35ChatOptions.MaxNewTokens"/>
    /// (default 512), <see cref="Qwen35ChatOptions.EnableThinking"/> (default false), <see cref="Qwen35ChatOptions.StopSequences"/>,
    /// <see cref="Qwen35ChatOptions.AdditionalStopTokenIds"/>, <see cref="Qwen35ChatOptions.Sampling"/> (default greedy) and
    /// <see cref="Qwen35ChatOptions.PrefillChunkTokens"/> (default: 512 on CUDA, one call on CPU). A request value replaces
    /// the default; see <see cref="Qwen35ChatClient"/> for the mapping.
    /// </summary>
    /// <remarks>
    /// On CPU the default one-call prefill cannot be cancelled until it completes (12 s for a 1,666-token prompt on the
    /// M0 workstation). Set <see cref="Qwen35ChatOptions.PrefillChunkTokens"/> to bound that latency; on CPU chunks give
    /// bit-identical logits but a slower prefill (P-016).
    /// </remarks>
    public Qwen35ChatOptions Defaults { get; init; } = new();

    /// <summary>
    /// Streamed updates the worker may produce ahead of the consumer (at least 1). When the buffer is full the worker
    /// waits before its next decode step, so a slow consumer slows generation instead of growing memory. Default 16.
    /// </summary>
    public int StreamingBufferCapacity { get; init; } = DefaultStreamingBufferCapacity;

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Defaults, nameof(Defaults));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(StreamingBufferCapacity, nameof(StreamingBufferCapacity));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Defaults.MaxNewTokens, nameof(Qwen35ChatOptions.MaxNewTokens));
    }
}
