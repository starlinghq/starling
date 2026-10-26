namespace Starlings.Models.Qwen35;

/// <summary>Support status of one capability (plan section 9). A missing entry never means supported.</summary>
public enum Qwen35CapabilityStatus
{
    /// <summary>Qualified for the advertised maturity.</summary>
    Supported,

    /// <summary>Not available; requests for it are rejected or the API does not offer it.</summary>
    Unsupported,

    /// <summary>Implemented and tested against the reference, not yet qualified.</summary>
    Experimental,

    /// <summary>May work, but no sufficient evidence exists.</summary>
    Untested,
}

/// <summary>Capability descriptor for this package version (profiles <c>qwen3.5-0.8b/ortgenai-fp32x</c> and <c>ortgenai-fp16x</c>).</summary>
/// <remarks>
/// <see cref="All"/> is package-wide. Keys without a provider or variant suffix give the status on the Stable CPU profile
/// <see cref="StableProfile"/> (fp32x on Linux x64 CPU); keys with <c>.cuda</c> or <c>fp16x</c> give the Stable CUDA
/// profiles' <see cref="StableCudaProfile"/> and <see cref="StableCudaFp16xProfile"/> (Linux x64, RTX 3090).
/// <see cref="Profiles"/> repeats each Stable profile's statuses keyed exactly as its registry manifest's
/// <c>capabilities</c>, and a test keeps the two equal key for key.
/// </remarks>
public static class Qwen35Capabilities
{
    /// <summary>The profile of this package's Stable contract (the README's "Stable contract" section).</summary>
    public const string StableProfile = "qwen3.5-0.8b/ortgenai-fp32x-cpu";

    /// <summary>The fp32x CUDA profile of this package's Stable contract (the README's "Stable contract: CUDA" section).</summary>
    public const string StableCudaProfile = "qwen3.5-0.8b/ortgenai-fp32x-cuda";

    /// <summary>The fp16x CUDA profile of this package's Stable contract (the README's "Stable contract: CUDA" section).</summary>
    public const string StableCudaFp16xProfile = "qwen3.5-0.8b/ortgenai-fp16x-cuda";

    /// <summary>Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c> and <c>docs/evidence/p016-qwen35/README.md</c>.</summary>
    public static IReadOnlyDictionary<string, Qwen35CapabilityStatus> All { get; } = new Dictionary<string, Qwen35CapabilityStatus>(StringComparer.Ordinal)
    {
        // fp32x on Linux x64 CPU, Stable since 2026-10-16 (the README's Stable contract): G1 and G2 on the P-030 release
        // prompts (32: template and ids exact, greedy ids exact, full-vocabulary step 0 9.3e-4 against 5e-3, cached vs
        // uncached, chunked prefill, concurrency) and tokenizer holdout (232 rows exact), the development fixtures
        // (prompts-v1, up to 1,666 tokens), G3 GSM8K (0.64, budget 0.60), G4 LLM semantics and one-hour soak, G5 (quiet-host
        // rerun), G6 anonymous download and offline run, G7 IChatClient; the weekly qualification reruns G1 to G3.
        // fp32x and fp16x on CUDA (RTX 3090), Stable since 2026-10-16 (the README's "Stable contract: CUDA"): G1 to G3 on the
        // release sets (fp16x under its protected near-tie rule), G4 and G5 (binding round), G6, G7; the per-release GPU
        // regression reruns the CUDA tests, G1 to G3 and a soak smoke test (every check green in run 38857860186).
        ["chat.greedy.fp32x.cpu"] = Qwen35CapabilityStatus.Supported,
        ["chat.greedy.fp32x.cuda"] = Qwen35CapabilityStatus.Supported,
        ["chat.greedy.fp16x.cuda"] = Qwen35CapabilityStatus.Supported,
        ["template.owned_renderer.text"] = Qwen35CapabilityStatus.Supported,
        ["template.enable_thinking"] = Qwen35CapabilityStatus.Experimental,
        ["tokenizer.owned"] = Qwen35CapabilityStatus.Supported,
        ["stop.eos_tokens"] = Qwen35CapabilityStatus.Supported,
        ["stop.sequences"] = Qwen35CapabilityStatus.Supported,
        ["stop.additional_token_ids"] = Qwen35CapabilityStatus.Experimental,
        ["max_new_tokens"] = Qwen35CapabilityStatus.Supported,
        ["streaming"] = Qwen35CapabilityStatus.Supported,
        ["cancellation.between_steps"] = Qwen35CapabilityStatus.Supported,
        ["cancellation.between_prefill_chunks"] = Qwen35CapabilityStatus.Experimental,
        ["prefill.chunked"] = Qwen35CapabilityStatus.Supported,
        ["usage"] = Qwen35CapabilityStatus.Supported,
        ["concurrent_requests.cpu"] = Qwen35CapabilityStatus.Supported,
        ["concurrent_requests.cuda.serialized"] = Qwen35CapabilityStatus.Supported,
        ["sampling.seeded"] = Qwen35CapabilityStatus.Supported,
        ["step0_logits"] = Qwen35CapabilityStatus.Supported,
        ["asset_acquisition.offline"] = Qwen35CapabilityStatus.Supported,
        // Native AOT on Linux x64 with the CPU and CUDA GenAI natives (fp32x, fp16x): no warnings, equal to JIT.
        ["aot"] = Qwen35CapabilityStatus.Supported,
        // With the CUDA natives (tools/aot-check-ai.sh --cuda): one local run, no per-release regression.
        ["aot.cuda"] = Qwen35CapabilityStatus.Experimental,

        // From the hosting repository: fp16x lock downloaded and verified with a read token while it was private (opt-in test,
        // one machine); the repository is public since 2026-10-14.
        ["asset_acquisition.download"] = Qwen35CapabilityStatus.Supported,

        // Microsoft.Extensions.AI IChatClient, in Starlings.Models.Qwen35.Extensions.AI (P-017): tested with
        // scripted sequences in CI and against the reference on the real model (fp32x CPU; fp32x CUDA locally); G7 on
        // fp32x CPU. Tools stay experimental with tool calling.
        ["extensions_ai.chat_client"] = Qwen35CapabilityStatus.Supported,
        ["extensions_ai.streaming"] = Qwen35CapabilityStatus.Supported,
        ["extensions_ai.cancellation"] = Qwen35CapabilityStatus.Supported,
        ["extensions_ai.usage"] = Qwen35CapabilityStatus.Supported,
        ["extensions_ai.sampling.seeded"] = Qwen35CapabilityStatus.Supported,
        ["extensions_ai.concurrent_requests"] = Qwen35CapabilityStatus.Supported,
        ["extensions_ai.middleware"] = Qwen35CapabilityStatus.Supported,
        ["extensions_ai.dependency_injection"] = Qwen35CapabilityStatus.Supported,
        ["extensions_ai.fp16x.cuda"] = Qwen35CapabilityStatus.Supported, // equal to the typed fp16x model on 21 prompts, per release

        // Tool calling: the template's tools block, assistant tool calls and tool messages equal the upstream rendering
        // (transformers 5.18.0) on the proposed P-029 corpus; the output parser and the IChatClient mapping
        // (FunctionCallContent, FunctionResultContent, ToolMode Auto and None, FunctionInvokingChatClient) are tested
        // with scripted sequences. On the real fp32x model on CPU, one get_weather call was parsed and the
        // UseFunctionInvocation loop completed (greedy); a 0.8B model's tool choice is not otherwise qualified.
        ["tool_calling"] = Qwen35CapabilityStatus.Experimental,
        ["template.tools"] = Qwen35CapabilityStatus.Experimental,
        ["template.tool_calls"] = Qwen35CapabilityStatus.Experimental,
        ["template.tool_messages"] = Qwen35CapabilityStatus.Experimental,
        ["tool_call_parsing"] = Qwen35CapabilityStatus.Experimental,
        ["tool_call_parsing.streaming"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.tools"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.function_invocation"] = Qwen35CapabilityStatus.Experimental,

        // Long context: prompts of 4,112 and 8,213 tokens against the reference (proposed fixture) on fp32x CPU and
        // CUDA and fp16x CUDA. Prompt plus new tokens above Qwen35ChatModel.MaxContextTokens (8,192) is rejected with
        // GenerationContextLengthException: at 16,391 tokens cached vs uncached decoding exceeded the protected bound.
        ["long_context.1667_to_8192_tokens"] = Qwen35CapabilityStatus.Experimental,
        ["long_context.over_8192_tokens"] = Qwen35CapabilityStatus.Unsupported,

        // Implemented, without enough evidence.
        ["windows"] = Qwen35CapabilityStatus.Untested,
        ["linux_arm64"] = Qwen35CapabilityStatus.Untested,
        ["macos"] = Qwen35CapabilityStatus.Untested,

        // Not offered.
        ["cancellation.inside_a_step"] = Qwen35CapabilityStatus.Unsupported,
        ["batch"] = Qwen35CapabilityStatus.Unsupported,
        ["concurrent_requests.cuda.unserialized"] = Qwen35CapabilityStatus.Unsupported,
        ["conversation_state_reuse"] = Qwen35CapabilityStatus.Unsupported,
        ["reasoning_split"] = Qwen35CapabilityStatus.Unsupported,
        ["structured_output"] = Qwen35CapabilityStatus.Unsupported,
        ["vision"] = Qwen35CapabilityStatus.Unsupported,
        ["fp16x.cpu"] = Qwen35CapabilityStatus.Unsupported,
        ["tf32"] = Qwen35CapabilityStatus.Unsupported,
        ["quantized"] = Qwen35CapabilityStatus.Unsupported,
        ["lora"] = Qwen35CapabilityStatus.Unsupported,
        ["mtp_speculative_decoding"] = Qwen35CapabilityStatus.Unsupported,
        ["backend_tokenizer"] = Qwen35CapabilityStatus.Unsupported,
        ["extensions_ai.tool_mode.required"] = Qwen35CapabilityStatus.Unsupported,
        ["extensions_ai.structured_output"] = Qwen35CapabilityStatus.Unsupported,
        ["extensions_ai.multimodal"] = Qwen35CapabilityStatus.Unsupported,
        ["extensions_ai.unseeded_sampling"] = Qwen35CapabilityStatus.Unsupported,
        ["extensions_ai.reasoning_options"] = Qwen35CapabilityStatus.Unsupported,
        ["extensions_ai.penalties"] = Qwen35CapabilityStatus.Unsupported,
        ["extensions_ai.conversation_id"] = Qwen35CapabilityStatus.Unsupported,
        ["extensions_ai.background_responses"] = Qwen35CapabilityStatus.Unsupported,
        ["mlnet"] = Qwen35CapabilityStatus.Unsupported,
        ["persistence"] = Qwen35CapabilityStatus.Unsupported,
        ["training"] = Qwen35CapabilityStatus.Unsupported,
    };

    /// <summary>
    /// Status per capability of each profile this package version publishes a contract for, keyed exactly as the
    /// profile's registry manifest (<c>model-pipeline/registry/&lt;profile&gt;/manifest.yaml</c>, <c>capabilities</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, Qwen35CapabilityStatus>> Profiles { get; } =
        new Dictionary<string, IReadOnlyDictionary<string, Qwen35CapabilityStatus>>(StringComparer.Ordinal)
        {
            [StableProfile] = CpuProfile,
            // The CUDA profiles keep the CPU profile's keys and statuses (concurrent requests serialized step by step),
            // except Native AOT with the CUDA natives (one local run).
            [StableCudaProfile] = WithAotExperimental(CpuProfile),
            [StableCudaFp16xProfile] = WithAotExperimental(CpuProfile),
        };

    private static Dictionary<string, Qwen35CapabilityStatus> CpuProfile =>
        new(StringComparer.Ordinal)
        {
            ["batch"] = Qwen35CapabilityStatus.Unsupported,
            ["streaming"] = Qwen35CapabilityStatus.Supported,
            ["cancellation"] = Qwen35CapabilityStatus.Supported,
            ["persistence"] = Qwen35CapabilityStatus.Unsupported,
            ["training"] = Qwen35CapabilityStatus.Unsupported,
            ["aot"] = Qwen35CapabilityStatus.Supported,
            ["extensions_ai"] = Qwen35CapabilityStatus.Supported,
            ["mlnet"] = Qwen35CapabilityStatus.Unsupported,
            ["greedy_generation"] = Qwen35CapabilityStatus.Supported,
            ["sampling_seeded"] = Qwen35CapabilityStatus.Supported,
            ["stop_tokens"] = Qwen35CapabilityStatus.Supported,
            ["stop_sequences"] = Qwen35CapabilityStatus.Supported,
            ["max_new_tokens"] = Qwen35CapabilityStatus.Supported,
            ["usage"] = Qwen35CapabilityStatus.Supported,
            ["concurrent_requests"] = Qwen35CapabilityStatus.Supported,
            ["step0_logits"] = Qwen35CapabilityStatus.Supported,
            ["chat_template_owned_text"] = Qwen35CapabilityStatus.Supported,
            // One accepted development prompt; the P-029 rows are proposed.
            ["enable_thinking"] = Qwen35CapabilityStatus.Experimental,
            ["tokenizer_owned"] = Qwen35CapabilityStatus.Supported,
            ["cancellation_inside_a_step"] = Qwen35CapabilityStatus.Unsupported,
            ["chunked_prefill"] = Qwen35CapabilityStatus.Supported,
            ["conversation_state_reuse"] = Qwen35CapabilityStatus.Unsupported,
            // Proposed P-029 tool cases and one real-model run; tool choice is not qualified.
            ["tool_calling"] = Qwen35CapabilityStatus.Experimental,
            ["reasoning_split"] = Qwen35CapabilityStatus.Unsupported,
            ["structured_output"] = Qwen35CapabilityStatus.Unsupported,
            ["vision"] = Qwen35CapabilityStatus.Unsupported,
            // Proposed long-context fixture, protected bounds.
            ["long_context_1667_to_8192_tokens"] = Qwen35CapabilityStatus.Experimental,
            ["long_context_over_8192_tokens"] = Qwen35CapabilityStatus.Unsupported,
        };

    private static Dictionary<string, Qwen35CapabilityStatus> WithAotExperimental(Dictionary<string, Qwen35CapabilityStatus> table)
    {
        var copy = new Dictionary<string, Qwen35CapabilityStatus>(table, StringComparer.Ordinal)
        {
            ["aot"] = Qwen35CapabilityStatus.Experimental,
        };
        return copy;
    }
}
