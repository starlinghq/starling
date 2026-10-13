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
public static class Qwen35Capabilities
{
    /// <summary>Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c> and <c>docs/evidence/p016-qwen35/README.md</c>.</summary>
    public static IReadOnlyDictionary<string, Qwen35CapabilityStatus> All { get; } = new Dictionary<string, Qwen35CapabilityStatus>(StringComparer.Ordinal)
    {
        // Tested against the transformers fp32 reference on prompts-v1 and by the CI-safe unit tests.
        ["chat.greedy.fp32x.cpu"] = Qwen35CapabilityStatus.Experimental,
        ["chat.greedy.fp32x.cuda"] = Qwen35CapabilityStatus.Experimental,
        ["chat.greedy.fp16x.cuda"] = Qwen35CapabilityStatus.Experimental,
        ["template.owned_renderer.text"] = Qwen35CapabilityStatus.Experimental,
        ["template.enable_thinking"] = Qwen35CapabilityStatus.Experimental,
        ["tokenizer.owned"] = Qwen35CapabilityStatus.Experimental,
        ["stop.eos_tokens"] = Qwen35CapabilityStatus.Experimental,
        ["stop.sequences"] = Qwen35CapabilityStatus.Experimental,
        ["stop.additional_token_ids"] = Qwen35CapabilityStatus.Experimental,
        ["max_new_tokens"] = Qwen35CapabilityStatus.Experimental,
        ["streaming"] = Qwen35CapabilityStatus.Experimental,
        ["cancellation.between_steps"] = Qwen35CapabilityStatus.Experimental,
        ["cancellation.between_prefill_chunks"] = Qwen35CapabilityStatus.Experimental,
        ["prefill.chunked"] = Qwen35CapabilityStatus.Experimental,
        ["usage"] = Qwen35CapabilityStatus.Experimental,
        ["concurrent_requests.cpu"] = Qwen35CapabilityStatus.Experimental,
        ["concurrent_requests.cuda.serialized"] = Qwen35CapabilityStatus.Experimental,
        ["sampling.seeded"] = Qwen35CapabilityStatus.Experimental,
        ["step0_logits"] = Qwen35CapabilityStatus.Experimental,
        ["asset_acquisition.offline"] = Qwen35CapabilityStatus.Experimental,

        // From the private hosting repository with a read token: fp16x lock downloaded and verified (opt-in test, one machine).
        ["asset_acquisition.download"] = Qwen35CapabilityStatus.Experimental,

        // Microsoft.Extensions.AI IChatClient, in Starlings.Models.Qwen35.Extensions.AI (P-017): tested with
        // scripted sequences in CI and against the reference on the real model (fp32x CPU; fp32x CUDA locally).
        ["extensions_ai.chat_client"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.streaming"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.cancellation"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.usage"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.sampling.seeded"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.concurrent_requests"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.middleware"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.dependency_injection"] = Qwen35CapabilityStatus.Experimental,
        ["extensions_ai.fp16x.cuda"] = Qwen35CapabilityStatus.Untested,

        // Implemented, without enough evidence.
        ["long_context.over_1666_tokens"] = Qwen35CapabilityStatus.Untested,
        ["windows"] = Qwen35CapabilityStatus.Untested,
        ["linux_arm64"] = Qwen35CapabilityStatus.Untested,
        ["macos"] = Qwen35CapabilityStatus.Untested,
        ["aot"] = Qwen35CapabilityStatus.Untested,

        // Not offered.
        ["cancellation.inside_a_step"] = Qwen35CapabilityStatus.Unsupported,
        ["batch"] = Qwen35CapabilityStatus.Unsupported,
        ["concurrent_requests.cuda.unserialized"] = Qwen35CapabilityStatus.Unsupported,
        ["conversation_state_reuse"] = Qwen35CapabilityStatus.Unsupported,
        ["tool_calling"] = Qwen35CapabilityStatus.Unsupported,
        ["template.tools"] = Qwen35CapabilityStatus.Unsupported,
        ["template.tool_messages"] = Qwen35CapabilityStatus.Unsupported,
        ["reasoning_split"] = Qwen35CapabilityStatus.Unsupported,
        ["structured_output"] = Qwen35CapabilityStatus.Unsupported,
        ["vision"] = Qwen35CapabilityStatus.Unsupported,
        ["fp16x.cpu"] = Qwen35CapabilityStatus.Unsupported,
        ["tf32"] = Qwen35CapabilityStatus.Unsupported,
        ["quantized"] = Qwen35CapabilityStatus.Unsupported,
        ["mtp_speculative_decoding"] = Qwen35CapabilityStatus.Unsupported,
        ["backend_tokenizer"] = Qwen35CapabilityStatus.Unsupported,
        ["extensions_ai.tools"] = Qwen35CapabilityStatus.Unsupported,
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
}
