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
/// <c>kev-4b/onnx-fp32-cpu</c>, and their CUDA siblings <c>onnx-fp32-cuda</c> on the same bundles; all Experimental).
/// Rows follow the feature contract of <c>model-pipeline/admissions/kev.md</c>; both sizes have the same statuses. Keys
/// without a provider suffix give the CPU status; keys with <c>.cuda</c> the CUDA status.
/// </summary>
/// <remarks>
/// <see cref="All"/> is package-wide. <see cref="Profiles"/> repeats each profile's statuses keyed exactly as its
/// registry manifest's <c>capabilities</c>, and a test keeps the two equal key for key.
/// </remarks>
public static class KevCapabilities
{
    /// <summary>Kev-0.8B on the CPU provider.</summary>
    public const string Kev08BCpuProfile = "kev-0.8b/onnx-fp32-cpu";

    /// <summary>Kev-0.8B on the CUDA provider (the same bundle, <c>use_tf32=0</c>).</summary>
    public const string Kev08BCudaProfile = "kev-0.8b/onnx-fp32-cuda";

    /// <summary>Kev-4B on the CPU provider.</summary>
    public const string Kev4BCpuProfile = "kev-4b/onnx-fp32-cpu";

    /// <summary>Kev-4B on the CUDA provider (the same bundle, <c>use_tf32=0</c>).</summary>
    public const string Kev4BCudaProfile = "kev-4b/onnx-fp32-cuda";

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
        // Four callers on one model, every request golden twice each, bit-identical to sequential calls (ModelParityTests,
        // CPU, not serialized; CudaParityTests, CUDA, serialized by default as KevOptions.SerializeNativeCalls).
        ["concurrent_calls"] = KevCapabilityStatus.Experimental,
        ["concurrent_calls.cuda.serialized"] = KevCapabilityStatus.Experimental,
        // Concurrent runs on one CUDA session gave wrong outputs for GLiNER2 and Qwen3.5 on ONNX Runtime 1.30; not tried for Kev.
        ["concurrent_calls.cuda.unserialized"] = KevCapabilityStatus.Untested,
        ["asset_verification.offline"] = KevCapabilityStatus.Experimental,

        // Refused, never truncated: states over 8,192 tokens and rows over 8,192 tokens (the validated context).
        ["state_over_8192_tokens"] = KevCapabilityStatus.Unsupported,
        ["truncation"] = KevCapabilityStatus.Unsupported,

        // KevModelAssets.DownloadAsync from the public hosting repositories (Starlinghq/kev-0.8b-onnx, kev-4b-onnx at
        // their r1 commits), verified against the lock; anonymous download and offline reuse tested with
        // STARLING_NETWORK_TESTS=1 (HostingDownloadTests). Experimental until the G6 distribution drill.
        ["asset_acquisition.download"] = KevCapabilityStatus.Experimental,

        // KevOptions.MaxBranchTokens and KevOptions.DeclineOverLongQuestions: every over-long question found in one pass,
        // thrown together or declined while the rest are answered with unchanged probabilities (ApiTests, CPU, KevModel.Kev08B).
        ["branch_token_limit"] = KevCapabilityStatus.Experimental,
        ["decline_over_long_questions"] = KevCapabilityStatus.Experimental,
        // KevDecisionModel.Placement (provider, device, profile, serialization, captured node placement) and
        // KevAnswer.IsCalibrated / KevDecisionModel.IsCalibrated (ApiTests, CPU, KevModel.Kev08B).
        ["placement_report"] = KevCapabilityStatus.Experimental,
        ["calibration_flag"] = KevCapabilityStatus.Experimental,
        // KevDecisionModel.CanLoad: options and provider checked without loading the bundle (ApiTests: CPU true, CUDA on the
        // CPU-only native package false with the reason). The CUDA-present path has no GPU evidence yet.
        ["load_check"] = KevCapabilityStatus.Experimental,
        ["load_check.cuda"] = KevCapabilityStatus.Untested,
        // KevState: one state pass reused by later Decide calls, on the bundles as qualified (no new graph). Every request
        // golden bit-identical to Decide, also from four concurrent callers (ApiTests, CPU, KevModel.Kev08B). Not run on CUDA.
        ["state_reuse_across_calls"] = KevCapabilityStatus.Experimental,
        ["state_reuse_across_calls.cuda"] = KevCapabilityStatus.Untested,

        // Not offered: reusing the cache of a common prefix of different states.
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

        // The CPU-built bundle on the CUDA provider with use_tf32=0 (profiles onnx-fp32-cuda; Runtime.Onnx forces TF32 off):
        // the request goldens within the CPU bound, cached equal to uncached, isolated questions and cancellation with no
        // queued device work, both sizes (CudaParityTests, RTX 3090). Experimental until the per-release GPU regression
        // and the maintainer's gates. The builder's own CUDA graph (-e cuda) is not this profile (admissions/kev.md).
        ["cuda"] = KevCapabilityStatus.Experimental,
        ["tf32"] = KevCapabilityStatus.Unsupported,

        // Possible in principle, no evidence for this package.
        ["windows"] = KevCapabilityStatus.Untested,
        ["aot"] = KevCapabilityStatus.Untested,
        ["kev_9b"] = KevCapabilityStatus.Unsupported,
        ["kev_27b"] = KevCapabilityStatus.Unsupported,
    };

    /// <summary>
    /// Status per capability of each profile this package version has a contract for, keyed exactly as the profile's
    /// registry manifest (<c>model-pipeline/registry/&lt;profile&gt;/manifest.yaml</c>, <c>capabilities</c>). Both sizes
    /// have the same statuses on the same provider.
    /// </summary>
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, KevCapabilityStatus>> Profiles { get; } =
        new Dictionary<string, IReadOnlyDictionary<string, KevCapabilityStatus>>(StringComparer.Ordinal)
        {
            [Kev08BCpuProfile] = CpuProfile(),
            [Kev08BCudaProfile] = CudaProfile(),
            [Kev4BCpuProfile] = CpuProfile(),
            [Kev4BCudaProfile] = CudaProfile(),
        };

    private static Dictionary<string, KevCapabilityStatus> CpuProfile() => new(StringComparer.Ordinal)
    {
        ["batch"] = KevCapabilityStatus.Unsupported,
        ["streaming"] = KevCapabilityStatus.Unsupported,
        ["cancellation"] = KevCapabilityStatus.Experimental,
        ["persistence"] = KevCapabilityStatus.Unsupported,
        ["training"] = KevCapabilityStatus.Unsupported,
        ["aot"] = KevCapabilityStatus.Untested,
        ["extensions_ai"] = KevCapabilityStatus.Unsupported,
        ["mlnet"] = KevCapabilityStatus.Unsupported,
        ["decision_noul"] = KevCapabilityStatus.Experimental,
        ["decision_choice"] = KevCapabilityStatus.Experimental,
        ["decision_score"] = KevCapabilityStatus.Experimental,
        ["calibrated_temperature"] = KevCapabilityStatus.Experimental,
        ["raw_probabilities"] = KevCapabilityStatus.Experimental,
        ["delimiter_escaping"] = KevCapabilityStatus.Experimental,
        ["question_isolation"] = KevCapabilityStatus.Experimental,
        ["state_cache_within_request"] = KevCapabilityStatus.Experimental,
        ["state_json_rendering"] = KevCapabilityStatus.Experimental,
        ["concurrent_calls"] = KevCapabilityStatus.Experimental,
        ["asset_verification_offline"] = KevCapabilityStatus.Experimental,
        ["state_over_8192_tokens"] = KevCapabilityStatus.Unsupported,
        ["asset_acquisition_download"] = KevCapabilityStatus.Experimental,
        ["state_prefix_reuse_across_requests"] = KevCapabilityStatus.Unsupported,
        ["probability_rounding"] = KevCapabilityStatus.Unsupported,
        ["usage_output_tokens"] = KevCapabilityStatus.Unsupported,
        ["date_facts"] = KevCapabilityStatus.Unsupported,
        ["option_isolation"] = KevCapabilityStatus.Unsupported,
        ["http_server"] = KevCapabilityStatus.Unsupported,
        ["bf16"] = KevCapabilityStatus.Unsupported,
        ["fp16"] = KevCapabilityStatus.Unsupported,
        ["quantized"] = KevCapabilityStatus.Unsupported,
        ["calibration_fitting"] = KevCapabilityStatus.Unsupported,
        ["generation"] = KevCapabilityStatus.Unsupported,
        ["non_english"] = KevCapabilityStatus.Unsupported,
    };

    private static Dictionary<string, KevCapabilityStatus> CudaProfile()
    {
        var table = CpuProfile();
        // Serialized on CUDA (KevOptions.SerializeNativeCalls, the CUDA default); unserialized runs were not tried.
        table["concurrent_calls"] = KevCapabilityStatus.Experimental;
        table["concurrent_calls_unserialized"] = KevCapabilityStatus.Untested;
        // Runtime.Onnx forces use_tf32=0; TF32 fails every fp32 bound.
        table["tf32"] = KevCapabilityStatus.Unsupported;
        return table;
    }
}
