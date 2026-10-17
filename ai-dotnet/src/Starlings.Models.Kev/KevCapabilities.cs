namespace Starlings.Models.Kev;

/// <summary>Support status of one capability (plan section 9). A missing entry never means supported.</summary>
public enum KevCapabilityStatus
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

/// <summary>
/// Capability descriptor for this package version (profiles <c>kev-0.8b/onnx-fp32-cpu</c> and
/// <c>kev-4b/onnx-fp32-cpu</c>, both Experimental). Rows follow the feature contract of
/// <c>model-pipeline/admissions/kev.md</c>; both profiles have the same statuses.
/// </summary>
public static class KevCapabilities
{
    /// <summary>Status per capability (both profiles, Linux x64 CPU).</summary>
    public static IReadOnlyDictionary<string, KevCapabilityStatus> All { get; } = new Dictionary<string, KevCapabilityStatus>(StringComparer.Ordinal)
    {
        // Implemented and compared with Kev's PyTorch fp32 path (C# against Python ONNX Runtime and the reference on the
        // proposed corpus and fixtures); Experimental until the maintainer adopts references and gates.
        ["decision.noul"] = KevCapabilityStatus.Experimental,
        ["decision.choice"] = KevCapabilityStatus.Experimental,
        ["decision.score"] = KevCapabilityStatus.Experimental,
        ["calibrated_temperature"] = KevCapabilityStatus.Experimental,
        ["raw_probabilities"] = KevCapabilityStatus.Experimental,
        ["delimiter_escaping"] = KevCapabilityStatus.Experimental,
        ["question_isolation"] = KevCapabilityStatus.Experimental,
        ["state_cache_within_request"] = KevCapabilityStatus.Experimental,
        ["state_json_rendering"] = KevCapabilityStatus.Experimental,
        ["cancellation"] = KevCapabilityStatus.Experimental,
        ["concurrent_calls"] = KevCapabilityStatus.Untested,
        ["asset_verification.offline"] = KevCapabilityStatus.Experimental,

        // Refused, never truncated: states over 8,192 tokens and rows over 8,192 tokens (the validated context).
        ["state_over_8192_tokens"] = KevCapabilityStatus.Unsupported,
        ["truncation"] = KevCapabilityStatus.Unsupported,

        // Not offered.
        ["asset_acquisition.download"] = KevCapabilityStatus.Unsupported,
        ["state_prefix_reuse_across_requests"] = KevCapabilityStatus.Unsupported,
        // batched_rows: CPU spike evidence in docs/evidence/kev-batched-rows (internal API only; maintainer decision).
        ["batched_rows"] = KevCapabilityStatus.Unsupported,
        ["probability_rounding"] = KevCapabilityStatus.Unsupported,
        ["usage_output_tokens"] = KevCapabilityStatus.Unsupported,
        ["date_facts"] = KevCapabilityStatus.Unsupported,
        ["option_isolation"] = KevCapabilityStatus.Unsupported,
        ["http_server"] = KevCapabilityStatus.Unsupported,
        ["bf16"] = KevCapabilityStatus.Unsupported,
        ["fp16"] = KevCapabilityStatus.Unsupported,
        ["quantized"] = KevCapabilityStatus.Unsupported,
        ["training"] = KevCapabilityStatus.Unsupported,
        ["calibration_fitting"] = KevCapabilityStatus.Unsupported,
        ["generation"] = KevCapabilityStatus.Unsupported,
        ["non_english"] = KevCapabilityStatus.Unsupported,
        ["extensions_ai"] = KevCapabilityStatus.Unsupported,
        ["mlnet"] = KevCapabilityStatus.Unsupported,

        // Possible in principle, no evidence for this package.
        ["cuda"] = KevCapabilityStatus.Untested,
        ["windows"] = KevCapabilityStatus.Untested,
        ["aot"] = KevCapabilityStatus.Untested,
        ["kev_9b"] = KevCapabilityStatus.Unsupported,
        ["kev_27b"] = KevCapabilityStatus.Unsupported,
    };
}
