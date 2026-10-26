using System.Collections.Immutable;
using Starlings.Generation;

namespace Starlings.Models.Qwen35;

/// <summary>Per-request settings for <c>Qwen35ChatModel</c> (Starlings.Models.Qwen35.OnnxGenAI). Immutable; reuse freely.</summary>
public sealed record Qwen35ChatOptions
{
    /// <summary>Maximum completion tokens (at least 1); a stop token never counts. Default 512.</summary>
    public int MaxNewTokens { get; init; } = GenerationOptions.DefaultMaxNewTokens;

    /// <summary>
    /// The template's <c>enable_thinking</c>. Default false (the template default: an empty think block, the model
    /// answers directly). True: the model writes its reasoning, then <c>&lt;/think&gt;</c>, then the answer; the
    /// completion text contains all of it (this package does not split it; the opt-in
    /// <c>Qwen35ChatClientOptions.SplitReasoning</c> of Starlings.Models.Qwen35.Extensions.AI does).
    /// </summary>
    public bool EnableThinking { get; init; }

    /// <summary>
    /// Tool definitions rendered into the chat template's tools block (default none). The completion may then contain
    /// tool calls in the template's format; split them out with <see cref="Qwen35ToolCallParser"/>. Experimental.
    /// </summary>
    public IReadOnlyList<Qwen35Tool> Tools { get; init; } = [];

    /// <summary>Stop strings (see <see cref="GenerationOptions.StopSequences"/>).</summary>
    public IReadOnlyList<string> StopSequences { get; init; } = [];

    /// <summary>Stop token ids in addition to <c>&lt;|im_end|&gt;</c> and <c>&lt;|endoftext|&gt;</c>.</summary>
    public IReadOnlyCollection<int> AdditionalStopTokenIds { get; init; } = [];

    /// <summary>Sampling settings, or null (default) for greedy decoding (the evidence profile).</summary>
    public SamplingOptions? Sampling { get; init; }

    /// <summary>
    /// Prefill chunk size (see <see cref="GenerationOptions.PrefillChunkTokens"/>), or null (default) for the provider
    /// default: <see cref="DefaultCudaPrefillChunkTokens"/> on CUDA, where a one-call prefill allocates about 1 MB of
    /// fp32 logits per prompt token in device memory, and <see cref="DefaultCpuPrefillChunkTokens"/> on CPU, so a prompt
    /// of at most 1,024 tokens is prefilled in one call as before and a longer one in 1,024-token chunks, which bounds
    /// the transient logits buffer to about 1 GB and lets cancellation act between chunks. On CPU, 1,024-token chunks
    /// gave greedy ids and step-0 logits bit-identical to one call; other chunk sizes stayed within the profile's
    /// chunked-prefill tolerance (<c>docs/evidence/cpu-prefill-chunking-2026-10-26/</c>).
    /// <see cref="int.MaxValue"/> forces one call.
    /// </summary>
    public int? PrefillChunkTokens { get; init; }

    /// <summary>Prefill chunk size used on CUDA when <see cref="PrefillChunkTokens"/> is null.</summary>
    public const int DefaultCudaPrefillChunkTokens = 512;

    /// <summary>
    /// Prefill chunk size used on CPU when <see cref="PrefillChunkTokens"/> is null, and so the prompt length above which
    /// the CPU prefill is chunked: prompts up to this many tokens take one call.
    /// </summary>
    public const int DefaultCpuPrefillChunkTokens = 1024;

    /// <summary>
    /// Opt-in: per-token log probabilities in <see cref="GenerationUpdate.Logprobs"/> and
    /// <see cref="GenerationResult.Logprobs"/> (see <see cref="GenerationOptions.ReturnLogprobs"/>). Default false.
    /// Experimental: scripted sequences, and one real-model test on CPU (greedy tokens are each step's most likely).
    /// </summary>
    public bool ReturnLogprobs { get; init; }

    /// <summary>
    /// With <see cref="ReturnLogprobs"/>: the most likely tokens returned per step, 0 (default) to
    /// <see cref="GenerationOptions.MaxTopLogprobs"/> (see <see cref="GenerationOptions.TopLogprobs"/>).
    /// </summary>
    public int TopLogprobs { get; init; }

    /// <summary>
    /// Experimental, opt-in: a constraint on the completion (see <see cref="GenerationOptions.Guidance"/>), for example
    /// <see cref="Qwen35ToolGrammar.Build(IReadOnlyList{Qwen35Tool}, Qwen35ToolGrammarOptions?)"/> for a required tool
    /// call or <see cref="GenerationGuidance.JsonSchema"/> for JSON output; null (default) for unconstrained decoding,
    /// which this option leaves unchanged. Needs <c>tokenizer.json</c> and <c>tokenizer_config.json</c> in the variant's
    /// model directory, which the published bundle does not have (<c>Qwen35ModelInfo.GuidanceMissingFiles</c>); a
    /// guided request on such a directory fails with <see cref="NotSupportedException"/>. With
    /// <see cref="EnableThinking"/>, the constraint applies from the first completion token, which is reasoning text.
    /// </summary>
    public GenerationGuidance? Guidance { get; init; }

    /// <summary>
    /// The decode-loop options of this request; <paramref name="defaultPrefillChunkTokens"/> applies when
    /// <see cref="PrefillChunkTokens"/> is null (the backend picks it per provider).
    /// </summary>
    internal GenerationOptions ToGenerationOptionsCore(int defaultPrefillChunkTokens) => new()
    {
        Guidance = Guidance,
        MaxNewTokens = MaxNewTokens,
        StopSequences = StopSequences,
        AdditionalStopTokenIds = AdditionalStopTokenIds,
        Sampling = Sampling,
        PrefillChunkTokens = PrefillChunkTokens ?? defaultPrefillChunkTokens,
        ReturnLogprobs = ReturnLogprobs,
        TopLogprobs = TopLogprobs,
    };
}

/// <summary>A prompt ready for generation: the rendered text (null when built from ids) and its token ids.</summary>
/// <param name="Text">The rendered chat template, or null for <see cref="FromTokenIds"/>.</param>
/// <param name="TokenIds">Owned-tokenizer ids of <paramref name="Text"/>.</param>
public sealed record Qwen35Prompt(string? Text, ImmutableArray<int> TokenIds)
{
    /// <summary>Advanced: a prompt from token ids rendered and tokenized elsewhere (for example reference fixture ids).</summary>
    public static Qwen35Prompt FromTokenIds(IEnumerable<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        return new Qwen35Prompt(null, [.. tokenIds]);
    }
}

