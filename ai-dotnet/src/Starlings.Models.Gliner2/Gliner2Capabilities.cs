namespace Starlings.Models.Gliner2;

/// <summary>Support status of one capability (plan section 9). A missing entry never means supported.</summary>
public enum Gliner2CapabilityStatus
{
    /// <summary>Qualified for the advertised maturity.</summary>
    Supported,

    /// <summary>Not available; requests for it are rejected or the API does not offer it.</summary>
    Unsupported,

    /// <summary>Implemented and tested against the reference fixtures, not yet qualified.</summary>
    Experimental,

    /// <summary>May work, but no sufficient evidence exists.</summary>
    Untested,
}

/// <summary>Capability descriptor for this package version (profile <c>gliner2-base-v1/onnx-fp32-cpu</c>).</summary>
public static class Gliner2Capabilities
{
    /// <summary>Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c>.</summary>
    public static IReadOnlyDictionary<string, Gliner2CapabilityStatus> All { get; } = new Dictionary<string, Gliner2CapabilityStatus>(StringComparer.Ordinal)
    {
        // Declared contract (docs/kickoff.md, settled 2026-10-08): English entity extraction with labels and threshold.
        ["entity_extraction.cpu"] = Gliner2CapabilityStatus.Experimental,
        ["entity_extraction.cuda"] = Gliner2CapabilityStatus.Experimental,
        ["entity_extraction.staged"] = Gliner2CapabilityStatus.Experimental,
        ["threshold"] = Gliner2CapabilityStatus.Experimental,
        ["offsets.original_text"] = Gliner2CapabilityStatus.Experimental,
        ["batch"] = Gliner2CapabilityStatus.Experimental,
        ["cancellation"] = Gliner2CapabilityStatus.Experimental,
        ["concurrent_calls"] = Gliner2CapabilityStatus.Experimental,
        ["asset_acquisition.offline"] = Gliner2CapabilityStatus.Experimental,

        // Implemented, without enough evidence: one development case for descriptions; the count gate never closed on
        // the corpus; the hosted ONNX graph is not published yet (placeholder source in Gliner2ModelAssets).
        ["label_descriptions"] = Gliner2CapabilityStatus.Untested,
        ["count_gate.closed"] = Gliner2CapabilityStatus.Untested,
        ["asset_acquisition.download"] = Gliner2CapabilityStatus.Untested,
        ["aot"] = Gliner2CapabilityStatus.Untested,
        ["windows"] = Gliner2CapabilityStatus.Untested,

        // Outside the contract.
        ["input_over_512_tokens"] = Gliner2CapabilityStatus.Unsupported,
        ["truncation"] = Gliner2CapabilityStatus.Unsupported,
        ["max_len_word_cap"] = Gliner2CapabilityStatus.Unsupported,
        ["extract_long"] = Gliner2CapabilityStatus.Unsupported,
        ["plain_deduplicated_output"] = Gliner2CapabilityStatus.Unsupported,
        ["batch.mixed_label_sets"] = Gliner2CapabilityStatus.Unsupported,
        ["classification"] = Gliner2CapabilityStatus.Unsupported,
        ["relation_extraction"] = Gliner2CapabilityStatus.Unsupported,
        ["structured_extraction"] = Gliner2CapabilityStatus.Unsupported,
        ["per_label_thresholds"] = Gliner2CapabilityStatus.Unsupported,
        ["overlap_policies"] = Gliner2CapabilityStatus.Unsupported,
        ["char_splitter"] = Gliner2CapabilityStatus.Unsupported,
        ["non_english"] = Gliner2CapabilityStatus.Unsupported,
        ["tf32"] = Gliner2CapabilityStatus.Unsupported,
        ["fp16"] = Gliner2CapabilityStatus.Unsupported,
        ["quantized"] = Gliner2CapabilityStatus.Unsupported,
        ["streaming"] = Gliner2CapabilityStatus.Unsupported,
        ["persistence"] = Gliner2CapabilityStatus.Unsupported,
        ["training"] = Gliner2CapabilityStatus.Unsupported,
        ["extensions_ai"] = Gliner2CapabilityStatus.Unsupported,
        ["mlnet"] = Gliner2CapabilityStatus.Unsupported,
    };
}
