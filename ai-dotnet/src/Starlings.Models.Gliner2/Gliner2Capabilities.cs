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

/// <summary>Capability descriptor for this package version (profiles <c>gliner2-base-v1/onnx-fp32-cpu</c> and <c>onnx-fp32-cuda</c>).</summary>
/// <remarks>
/// <see cref="All"/> is package-wide. Keys without a provider suffix give the status on the Stable CPU profile
/// <see cref="StableProfile"/> (Linux x64 CPU); keys with <c>.cuda</c> give the status on the Stable CUDA profile
/// <see cref="StableCudaProfile"/> (Linux x64, RTX 3090). <see cref="Profiles"/> repeats each Stable profile's statuses
/// keyed exactly as its registry manifest's <c>capabilities</c>, and a test keeps the two equal key for key.
/// </remarks>
public static class Gliner2Capabilities
{
    /// <summary>The profile of this package's Stable contract (the README's "Stable contract" section).</summary>
    public const string StableProfile = "gliner2-base-v1/onnx-fp32-cpu";

    /// <summary>The CUDA profile of this package's Stable contract (the README's "Stable contract: CUDA" section).</summary>
    public const string StableCudaProfile = "gliner2-base-v1/onnx-fp32-cuda";

    /// <summary>
    /// Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c> and, for
    /// <see cref="Gliner2CapabilityStatus.Supported"/>, in the profile's gate evidence (<c>docs/catalog/README.md</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, Gliner2CapabilityStatus> All { get; } = new Dictionary<string, Gliner2CapabilityStatus>(StringComparer.Ordinal)
    {
        // Declared contract (docs/kickoff.md, settled 2026-10-08): English entity extraction with labels and threshold.
        // Supported on Linux x64 CPU: G1 and G2 on the P-030 release sets (processor outputs and offsets exact, span
        // logits within 1.78e-5 relative against 1e-4, entities 30/30 from inputs and from text, every padded batch equal
        // to single calls), G3 on WNUT-17 (exact F1 0.3626), G4 one-hour soak with MALLOC_MMAP_THRESHOLD_=65536 (the
        // documented Linux setting; concurrent callers, timed-out requests), G6 anonymous download and offline run.
        ["entity_extraction.cpu"] = Gliner2CapabilityStatus.Supported,
        // CUDA (use_tf32 0) on an RTX 3090, Stable since 2026-10-16 (the README's "Stable contract: CUDA"): G1 to G3 on the
        // release sets, G4 one-hour soak and G5 (binding round), G6, G7; the per-release GPU regression reruns the CUDA
        // tests, G1 to G3 and a soak smoke test (every check green in run 38857860186). Other GPUs: untested.
        ["entity_extraction.cuda"] = Gliner2CapabilityStatus.Supported,
        ["entity_extraction.staged"] = Gliner2CapabilityStatus.Supported,
        ["threshold"] = Gliner2CapabilityStatus.Supported,
        ["offsets.original_text"] = Gliner2CapabilityStatus.Supported,
        ["batch"] = Gliner2CapabilityStatus.Supported,
        ["cancellation"] = Gliner2CapabilityStatus.Supported,
        ["concurrent_calls"] = Gliner2CapabilityStatus.Supported,
        // On CUDA, concurrent runs on one session returned wrong entities (6 of 8 test runs); calls are serialized
        // there by default (Gliner2ExtractorOptions.SerializeNativeCalls), and that serialized mode is the CUDA contract's.
        ["concurrent_calls.cuda.serialized"] = Gliner2CapabilityStatus.Supported,
        ["concurrent_calls.cuda.unserialized"] = Gliner2CapabilityStatus.Unsupported,
        ["asset_acquisition.offline"] = Gliner2CapabilityStatus.Supported,

        // Proposed capability cases on public text, run by the pinned GLiNER2: 11 with label descriptions, 12 where the
        // count gate closes; inputs, gate and entities exact on CPU and CUDA (docs/evidence/capability-closure-2026-10-14/).
        // Proposed cases, not accepted fixtures: outside the Stable contract.
        ["label_descriptions"] = Gliner2CapabilityStatus.Experimental,
        ["count_gate.closed"] = Gliner2CapabilityStatus.Experimental,
        // G6: the release drills downloaded the graph from the public hosting repository with no token into an empty
        // cache, verified it and ran the sample offline (docs/evidence/g6-distribution-2026-10-14/).
        ["asset_acquisition.download"] = Gliner2CapabilityStatus.Supported,
        // Native AOT on Linux x64 CPU from the packed package: entity spans exact, confidences equal to JIT (docs/evidence/aot/).
        ["aot"] = Gliner2CapabilityStatus.Supported,
        // Native AOT with the CUDA natives (tools/aot-check-ai.sh --cuda): one local run, no per-release regression.
        ["aot.cuda"] = Gliner2CapabilityStatus.Experimental,
        // Windows x64: build, weight-free tests and the clean-install drill without the model in CI (ci.yml,
        // ai-dotnet-windows); no model run, so outside the Stable contract.
        ["windows"] = Gliner2CapabilityStatus.Experimental,

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

    /// <summary>
    /// Status per capability of each profile this package version publishes a contract for, keyed exactly as the
    /// profile's registry manifest (<c>model-pipeline/registry/&lt;profile&gt;/manifest.yaml</c>, <c>capabilities</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, Gliner2CapabilityStatus>> Profiles { get; } =
        new Dictionary<string, IReadOnlyDictionary<string, Gliner2CapabilityStatus>>(StringComparer.Ordinal)
        {
            [StableProfile] = new Dictionary<string, Gliner2CapabilityStatus>(StringComparer.Ordinal)
            {
                ["batch"] = Gliner2CapabilityStatus.Supported,
                ["streaming"] = Gliner2CapabilityStatus.Unsupported,
                ["cancellation"] = Gliner2CapabilityStatus.Supported,
                ["persistence"] = Gliner2CapabilityStatus.Unsupported,
                ["training"] = Gliner2CapabilityStatus.Unsupported,
                ["aot"] = Gliner2CapabilityStatus.Supported,
                ["extensions_ai"] = Gliner2CapabilityStatus.Unsupported,
                ["mlnet"] = Gliner2CapabilityStatus.Unsupported,
                ["entity_extraction"] = Gliner2CapabilityStatus.Supported,
                ["offsets_original_text"] = Gliner2CapabilityStatus.Supported,
                ["concurrent_calls"] = Gliner2CapabilityStatus.Supported,
                ["label_descriptions"] = Gliner2CapabilityStatus.Experimental,
                ["count_gate_closed"] = Gliner2CapabilityStatus.Experimental,
                ["input_over_512_tokens"] = Gliner2CapabilityStatus.Unsupported,
                ["plain_deduplicated_output"] = Gliner2CapabilityStatus.Unsupported,
                ["mixed_label_sets_in_batch"] = Gliner2CapabilityStatus.Unsupported,
                ["classification"] = Gliner2CapabilityStatus.Unsupported,
                ["relation_extraction"] = Gliner2CapabilityStatus.Unsupported,
                ["structured_extraction"] = Gliner2CapabilityStatus.Unsupported,
            },
            [StableCudaProfile] = new Dictionary<string, Gliner2CapabilityStatus>(StringComparer.Ordinal)
            {
                ["batch"] = Gliner2CapabilityStatus.Supported,
                ["streaming"] = Gliner2CapabilityStatus.Unsupported,
                ["cancellation"] = Gliner2CapabilityStatus.Supported,
                ["persistence"] = Gliner2CapabilityStatus.Unsupported,
                ["training"] = Gliner2CapabilityStatus.Unsupported,
                // One local run with the CUDA natives; not in the per-release GPU regression.
                ["aot"] = Gliner2CapabilityStatus.Experimental,
                ["extensions_ai"] = Gliner2CapabilityStatus.Unsupported,
                ["mlnet"] = Gliner2CapabilityStatus.Unsupported,
                ["entity_extraction"] = Gliner2CapabilityStatus.Supported,
                ["offsets_original_text"] = Gliner2CapabilityStatus.Supported,
                // Serialized (the CUDA default); unserialized runs on one CUDA session are unsupported.
                ["concurrent_calls"] = Gliner2CapabilityStatus.Supported,
                ["concurrent_calls_unserialized"] = Gliner2CapabilityStatus.Unsupported,
                ["label_descriptions"] = Gliner2CapabilityStatus.Experimental,
                ["count_gate_closed"] = Gliner2CapabilityStatus.Experimental,
                ["input_over_512_tokens"] = Gliner2CapabilityStatus.Unsupported,
                ["plain_deduplicated_output"] = Gliner2CapabilityStatus.Unsupported,
                ["mixed_label_sets_in_batch"] = Gliner2CapabilityStatus.Unsupported,
                ["classification"] = Gliner2CapabilityStatus.Unsupported,
                ["relation_extraction"] = Gliner2CapabilityStatus.Unsupported,
                ["structured_extraction"] = Gliner2CapabilityStatus.Unsupported,
            },
        };
}
