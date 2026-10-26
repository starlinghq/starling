using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.Gliner2;

/// <summary>Support status of one capability (plan section 9). A missing entry never means supported.</summary>
[Obsolete("Use Starlings.Models.CapabilityStatus; removed in the next minor version.")]
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
/// <see cref="Statuses"/> is package-wide. Keys without a provider suffix give the status on the Stable CPU profile
/// <see cref="StableProfile"/> (Linux x64 CPU); keys with <c>.cuda</c> give the status on the Stable CUDA profile
/// <see cref="StableCudaProfile"/> (Linux x64, RTX 3090). <see cref="ProfileCapabilities"/> holds the
/// capability record of every profile the package serves, generated from the registry manifests; a test keeps the
/// package-wide table consistent with the Stable profiles' records.
/// </remarks>
public static class Gliner2Capabilities
{
    /// <summary>The profile of this package's Stable contract (the README's "Stable contract" section).</summary>
    public const string StableProfile = "gliner2-base-v1/onnx-fp32-cpu";

    /// <summary>The CUDA profile of this package's Stable contract (the README's "Stable contract: CUDA" section).</summary>
    public const string StableCudaProfile = "gliner2-base-v1/onnx-fp32-cuda";

    /// <summary>
    /// Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c> and, for
    /// <see cref="Status.Supported"/>, in the profile's gate evidence (<c>docs/catalog/README.md</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, Status> Statuses { get; } = new Dictionary<string, Status>(StringComparer.Ordinal)
    {
        // Declared contract (docs/kickoff.md, settled 2026-10-08): English entity extraction with labels and threshold.
        // Supported on Linux x64 CPU: G1 and G2 on the P-030 release sets (processor outputs and offsets exact, span
        // logits within 1.78e-5 relative against 1e-4, entities 30/30 from inputs and from text, every padded batch equal
        // to single calls), G3 on WNUT-17 (exact F1 0.3626), G4 one-hour soak with MALLOC_MMAP_THRESHOLD_=65536 (the
        // documented Linux setting; concurrent callers, timed-out requests), G6 anonymous download and offline run.
        ["entity_extraction.cpu"] = Status.Supported,
        // CUDA (use_tf32 0) on an RTX 3090, Stable since 2026-10-16 (the README's "Stable contract: CUDA"): G1 to G3 on the
        // release sets, G4 one-hour soak and G5 (binding round), G6, G7; the per-release GPU regression reruns the CUDA
        // tests, G1 to G3 and a soak smoke test (every check green in run 38857860186). Other GPUs: untested.
        ["entity_extraction.cuda"] = Status.Supported,
        ["entity_extraction.staged"] = Status.Supported,
        ["threshold"] = Status.Supported,
        ["offsets.original_text"] = Status.Supported,
        ["batch"] = Status.Supported,
        // Opt-in explicit-batch chunking, Gliner2ExtractorOptions.MaxBatchTokens and SortByLength: off by default, and
        // the defaults keep the qualified outputs bit for bit. With sort_by_length, outputs stay within the
        // padded-batch-vs-single bound in the CPU tests; no qualification run yet.
        ["batch.token_budget"] = Status.Experimental,
        ["batch.sort_by_length"] = Status.Experimental,
        ["cancellation"] = Status.Supported,
        ["concurrent_calls"] = Status.Supported,
        // On CUDA, concurrent runs on one session returned wrong entities (6 of 8 test runs); calls are serialized
        // there by default (Gliner2ExtractorOptions.SerializeNativeCalls), and that serialized mode is the CUDA contract's.
        ["concurrent_calls.cuda.serialized"] = Status.Supported,
        ["concurrent_calls.cuda.unserialized"] = Status.Unsupported,
        // Cross-request micro-batching (opt-in Gliner2ExtractorOptions.MicroBatching, ExtractAsync; texts of one schema only): concurrent single requests share padded batches on one
        // dispatcher thread. Weight-free tests (coalescing, ordering, per-caller cancellation, backpressure, grouping,
        // disposal, failure isolation) and one weight-dependent test against direct calls within the padded-batch
        // tolerance; no benchmark adoption decision and no soak yet. Outputs depend on the concurrent traffic (within
        // tolerance, not bit for bit reproducible). Not part of any Stable profile's contract.
        ["micro_batching"] = Status.Experimental,
        // Gliner2ExtractorOptions.ExecutionScheduler: null keeps ExtractAsync inline; a TaskScheduler moves its work onto it.
        // Weight-free tests; not part of any Stable profile's contract.
        ["execution_scheduler"] = Status.Experimental,
        ["asset_acquisition.offline"] = Status.Supported,

        // Proposed capability cases on public text, run by the pinned GLiNER2: 11 with label descriptions, 12 where the
        // count gate closes; inputs, gate and entities exact on CPU and CUDA (docs/evidence/capability-closure-2026-10-14/).
        // Proposed cases, not accepted fixtures: outside the Stable contract.
        ["label_descriptions"] = Status.Experimental,
        ["count_gate.closed"] = Status.Experimental,
        // G6: the release drills downloaded the graph from the public hosting repository with no token into an empty
        // cache, verified it and ran the sample offline (docs/evidence/g6-distribution-2026-10-14/).
        ["asset_acquisition.download"] = Status.Supported,
        // Native AOT on Linux x64 CPU from the packed package: entity spans exact, confidences equal to JIT (docs/evidence/aot/).
        ["aot"] = Status.Supported,
        // Native AOT with the CUDA natives (tools/aot-check-ai.sh --cuda): one local run, no per-release regression.
        ["aot.cuda"] = Status.Experimental,
        // Windows x64: build, weight-free tests and the clean-install drill without the model in CI (ci.yml,
        // ai-dotnet-windows); no model run, so outside the Stable contract.
        ["windows"] = Status.Experimental,

        // Outside the contract.
        ["input_over_512_tokens"] = Status.Unsupported,
        ["truncation"] = Status.Unsupported,
        ["max_len_word_cap"] = Status.Unsupported,
        ["extract_long"] = Status.Unsupported,
        ["plain_deduplicated_output"] = Status.Unsupported,
        ["batch.mixed_label_sets"] = Status.Unsupported,
        ["classification"] = Status.Unsupported,
        ["relation_extraction"] = Status.Unsupported,
        ["structured_extraction"] = Status.Unsupported,
        ["per_label_thresholds"] = Status.Unsupported,
        ["overlap_policies"] = Status.Unsupported,
        ["char_splitter"] = Status.Unsupported,
        ["non_english"] = Status.Unsupported,
        ["tf32"] = Status.Unsupported,
        ["fp16"] = Status.Unsupported,
        ["quantized"] = Status.Unsupported,
        ["streaming"] = Status.Unsupported,
        ["persistence"] = Status.Unsupported,
        ["training"] = Status.Unsupported,
        ["extensions_ai"] = Status.Unsupported,
        ["mlnet"] = Status.Unsupported,
    };

    /// <summary>
    /// <see cref="Statuses"/> in this package's former status enum. Equal key for key; the enum values have the same
    /// names and order.
    /// </summary>
    [Obsolete("Use Gliner2Capabilities.Statuses; removed in the next minor version.")]
    public static IReadOnlyDictionary<string, Gliner2CapabilityStatus> All { get; } = Legacy(Statuses);

    /// <summary>
    /// The capability record of every registry profile this package serves (<see cref="ModelProfileCapabilities"/>),
    /// generated at build time from the profiles' registry manifests by <c>model-pipeline catalog</c>, keyed by profile
    /// id. Each record's <see cref="ModelProfileCapabilities.Capabilities"/> is the manifest's <c>capabilities</c> map
    /// key for key; the package-wide <see cref="Statuses"/> adds finer keys.
    /// </summary>
    public static IReadOnlyDictionary<string, ModelProfileCapabilities> ProfileCapabilities => GeneratedProfileCapabilities.Profiles;

    /// <summary>
    /// Status per capability of each Stable profile, keyed exactly as the profile's registry manifest
    /// (<c>model-pipeline/registry/&lt;profile&gt;/manifest.yaml</c>, <c>capabilities</c>): the Stable entries of
    /// <see cref="ProfileCapabilities"/> in this package's former status enum.
    /// </summary>
    [Obsolete("Use Gliner2Capabilities.ProfileCapabilities[profile].Capabilities; removed in the next minor version.")]
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, Gliner2CapabilityStatus>> Profiles { get; } =
        ProfileCapabilities.Values
            .Where(p => p.Maturity == ProfileMaturity.Stable)
            .ToDictionary(p => p.ProfileId, p => (IReadOnlyDictionary<string, Gliner2CapabilityStatus>)Legacy(p.Capabilities), StringComparer.Ordinal);

    [Obsolete("Converts to the former status enum for the obsolete members only.")]
    private static Dictionary<string, Gliner2CapabilityStatus> Legacy(IReadOnlyDictionary<string, Status> statuses) =>
        statuses.ToDictionary(e => e.Key, e => (Gliner2CapabilityStatus)(int)e.Value, StringComparer.Ordinal);
}
