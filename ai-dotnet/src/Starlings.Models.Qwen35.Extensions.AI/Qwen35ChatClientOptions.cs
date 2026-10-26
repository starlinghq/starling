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
    /// A <see cref="Qwen35ChatOptions.Guidance"/> set here applies to every request whose <c>ChatOptions</c> maps no
    /// constraint of its own (a required tool mode or a JSON response format replaces it), without the request-level
    /// checks those mappings make; leave it null (the default) unless every request should be constrained.
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
    /// Opt-in: build the tool-call grammar of <c>ChatToolMode.RequireAny</c> and <c>RequireSpecific</c> with fast-forward
    /// tokens (<see cref="Qwen35ToolGrammarOptions.FastForward"/>), so its fixed tags are appended without a decode step
    /// each. Default false: one token per step. Applies only where guided decoding is available
    /// (<see cref="Qwen35ModelInfo.GuidanceMissingFiles"/> empty). Experimental.
    /// </summary>
    public bool GuidanceFastForward { get; init; }

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Defaults, nameof(Defaults));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(StreamingBufferCapacity, nameof(StreamingBufferCapacity));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Defaults.MaxNewTokens, nameof(Qwen35ChatOptions.MaxNewTokens));
    }
}
