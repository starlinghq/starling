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
    /// the default; see <see cref="Qwen35ChatClient"/> for the mapping. The opt-in settings that <c>ChatOptions</c> cannot
    /// express, <see cref="Qwen35ChatOptions.EscapeSpecialTokens"/> and <see cref="Qwen35ChatOptions.StopAtRoleMarker"/>
    /// (both default false), apply to every request from here.
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

    /// <summary>
    /// Opt-in: when a request has thinking enabled, return the reasoning as <c>TextReasoningContent</c> and the answer
    /// as <c>TextContent</c>, split at the first <c>&lt;/think&gt;</c> (the tag is dropped), in streamed updates too. Default
    /// false: the completion stays one <c>TextContent</c> with the reasoning, <c>&lt;/think&gt;</c> and the answer, as in
    /// earlier versions. Requests without thinking are unaffected. Experimental (capability
    /// <c>extensions_ai.reasoning_split</c>).
    /// </summary>
    public bool SplitReasoning { get; init; }

    /// <summary>
    /// Admission control: the number of requests (<c>GetResponseAsync</c> and <c>GetStreamingResponseAsync</c> together)
    /// that may generate at the same time, at least 1; further requests wait their turn, in order, before any thread or
    /// native generator is created for them, and a request cancelled while waiting throws
    /// <see cref="OperationCanceledException"/> without having started. A streaming request holds its slot from its
    /// first <c>MoveNextAsync</c> until its enumeration completes or is disposed. Null (default) is unbounded, as before:
    /// one dedicated thread and one native generator per request.
    /// </summary>
    public int? MaxConcurrentRequests { get; init; }

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Defaults, nameof(Defaults));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(StreamingBufferCapacity, nameof(StreamingBufferCapacity));
        if (MaxConcurrentRequests is int max)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(max, nameof(MaxConcurrentRequests));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Defaults.MaxNewTokens, nameof(Qwen35ChatOptions.MaxNewTokens));
    }
}
