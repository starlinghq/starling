using System.Collections.Immutable;
using Starlings.Generation.OnnxGenAI;

namespace Starlings.Models.Qwen35;

/// <summary>Immutable load-time settings of a <see cref="Qwen35ChatModel"/>.</summary>
public sealed record Qwen35ModelOptions
{
    /// <summary>
    /// Execution provider. CPU runs <see cref="Qwen35Variant.Fp32x"/>; CUDA runs <see cref="Qwen35Variant.Fp32x"/> (TF32
    /// always off) or <see cref="Qwen35Variant.Fp16x"/>. A provider that cannot be used fails the load; no fallback.
    /// </summary>
    public GenAIExecutionProvider Provider { get; init; } = GenAIExecutionProvider.Cpu;

    /// <summary>CUDA device ordinal among the visible devices. Default 0.</summary>
    public int CudaDeviceId { get; init; }

    /// <summary>Intra-op thread count of the decoder session, or null for the GenAI default.</summary>
    public int? IntraOpThreads { get; init; }

    /// <summary>
    /// Advanced: see <see cref="GenAIModelOptions.SerializeNativeCalls"/>. Null (default) serializes the native work of
    /// concurrent requests on CUDA, where concurrent generators produced wrong tokens, and lets it run concurrently on CPU.
    /// </summary>
    public bool? SerializeNativeCalls { get; init; }
}

/// <summary>Per-request settings for <see cref="Qwen35ChatModel"/>. Immutable; reuse freely.</summary>
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

    /// <summary>
    /// Opt-in: also stop at <c>&lt;|im_start|&gt;</c> (<see cref="Qwen35ChatModel.ImStartTokenId"/>). The decode loop
    /// otherwise stops only at the two end-of-sequence ids, so a model that runs on after its answer into
    /// <c>&lt;|im_start|&gt;user ...</c> streams that role markup as text. Default false (today's behaviour: the token is
    /// decoded as text like any other).
    /// </summary>
    public bool StopAtRoleMarker { get; init; }

    /// <summary>
    /// Opt-in: escape the tokenizer's special-token strings in message content before rendering
    /// (<see cref="Qwen35ChatTemplate.EscapeSpecialTokens"/>): every <c>&lt;|name|&gt;</c> in the content and reasoning
    /// of system, user, assistant and tool messages becomes <c>&lt;¦name¦&gt;</c>, so content containing
    /// <c>&lt;|im_end|&gt;</c> or <c>&lt;|im_start|&gt;</c> cannot rewrite the conversation. Tool definitions and
    /// tool-call arguments are not escaped. Default false, which is faithful to upstream <c>apply_chat_template</c>: the
    /// content is rendered verbatim and tokenized with special tokens enabled.
    /// </summary>
    public bool EscapeSpecialTokens { get; init; }

    /// <summary>Sampling settings, or null (default) for greedy decoding (the evidence profile).</summary>
    public SamplingOptions? Sampling { get; init; }

    /// <summary>
    /// Prefill chunk size (see <see cref="GenerationOptions.PrefillChunkTokens"/>), or null (default) for the provider
    /// default: <see cref="DefaultCudaPrefillChunkTokens"/> on CUDA, where a one-call prefill allocates about 1 MB of
    /// fp32 logits per prompt token in device memory, and one call on CPU, where chunks were slower (the logits are
    /// bit-identical either way on CPU). <see cref="int.MaxValue"/> forces one call.
    /// </summary>
    public int? PrefillChunkTokens { get; init; }

    /// <summary>Prefill chunk size used on CUDA when <see cref="PrefillChunkTokens"/> is null.</summary>
    public const int DefaultCudaPrefillChunkTokens = 512;

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
    /// model directory, which the published bundle does not have (<see cref="Qwen35ModelInfo.GuidanceMissingFiles"/>); a
    /// guided request on such a directory fails with <see cref="NotSupportedException"/>. With
    /// <see cref="EnableThinking"/>, the constraint applies from the first completion token, which is reasoning text.
    /// </summary>
    public GenerationGuidance? Guidance { get; init; }

    internal GenerationOptions ToGenerationOptions(GenAIExecutionProvider provider = GenAIExecutionProvider.Cpu) => new()
    {
        Guidance = Guidance,
        MaxNewTokens = MaxNewTokens,
        StopSequences = StopSequences,
        AdditionalStopTokenIds = StopAtRoleMarker && !AdditionalStopTokenIds.Contains(Qwen35ChatModel.ImStartTokenId)
            ? [.. AdditionalStopTokenIds, Qwen35ChatModel.ImStartTokenId]
            : AdditionalStopTokenIds,
        Sampling = Sampling,
        PrefillChunkTokens = PrefillChunkTokens ?? (provider == GenAIExecutionProvider.Cuda ? DefaultCudaPrefillChunkTokens : null),
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

/// <summary>Identity of a loaded model (plan 5.1: inspect model, tokenizer, backend, provider and artifacts).</summary>
public sealed record Qwen35ModelInfo
{
    /// <summary>Profile id, for example <c>qwen3.5-0.8b/ortgenai-fp32x</c>.</summary>
    public required string Profile { get; init; }

    /// <summary>Hub repository and commit of the checkpoint.</summary>
    public required string Checkpoint { get; init; }

    /// <summary>The built variant.</summary>
    public required Qwen35Variant Variant { get; init; }

    /// <summary>Execution provider.</summary>
    public required GenAIExecutionProvider Provider { get; init; }

    /// <summary>ONNX Runtime GenAI managed binding version (from the dependency manifest).</summary>
    public required string GenAIVersion { get; init; }

    /// <summary>The ONNX Runtime GenAI native package and version in the dependency manifest, or null when unknown.</summary>
    public string? GenAINativePackage { get; init; }

    /// <summary>Owned tokenizer profile and the SHA-256 of its <c>tokenizer.json</c>.</summary>
    public required string Tokenizer { get; init; }

    /// <summary>Chat template renderer and the SHA-256 of the pinned template.</summary>
    public required string ChatTemplate { get; init; }

    /// <summary>Model directory loaded.</summary>
    public required string ModelDirectory { get; init; }

    /// <summary>Configuration overlays applied at load.</summary>
    public required IReadOnlyList<string> ConfigOverlays { get; init; }

    /// <summary>True when concurrent requests' native work runs one call at a time (the CUDA default).</summary>
    public required bool SerializesNativeCalls { get; init; }

    /// <summary>Provider options applied at load.</summary>
    public required IReadOnlyDictionary<string, string> ProviderOptions { get; init; }

    /// <summary>
    /// Longest accepted sequence (prompt plus completion tokens): the smaller of the model's context length and
    /// <see cref="Qwen35ChatModel.MaxContextTokens"/>.
    /// </summary>
    public required int ContextLength { get; init; }

    /// <summary>The model's own context length (<c>genai_config.json</c>, 262,144); longer than the accepted <see cref="ContextLength"/>.</summary>
    public int ModelContextLength { get; init; }

    /// <summary>Stop token ids applied to every request.</summary>
    public required ImmutableArray<int> StopTokenIds { get; init; }

    /// <summary>
    /// Experimental: the files guided decoding (<see cref="Qwen35ChatOptions.Guidance"/>) needs that the model directory
    /// lacks (<see cref="GenAIModel.GuidanceMissingFiles"/>); empty when guided requests can be served. The published
    /// bundle's variant directories have neither <c>tokenizer.json</c> nor <c>tokenizer_config.json</c>. Defaults to both
    /// file names for an instance that <see cref="Qwen35ChatModel.Load"/> did not create.
    /// </summary>
    public IReadOnlyList<string> GuidanceMissingFiles { get; init; } = GenAIModel.GuidanceFiles;
}
