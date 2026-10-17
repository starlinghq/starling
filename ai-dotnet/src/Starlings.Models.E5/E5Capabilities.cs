namespace Starlings.Models.E5;

/// <summary>Support status of one capability (plan section 9). A missing entry never means supported.</summary>
public enum CapabilityStatus
{
    /// <summary>Qualified for the advertised maturity.</summary>
    Supported,

    /// <summary>Not available; requests for it are rejected.</summary>
    Unsupported,

    /// <summary>Implemented and tested against the reference fixtures, not yet qualified.</summary>
    Experimental,

    /// <summary>May work, but no evidence exists.</summary>
    Untested,
}

/// <summary>Capability descriptor for this package version.</summary>
/// <remarks>
/// <see cref="All"/> is package-wide. Keys without a provider suffix give the status on the Stable CPU profile
/// <see cref="StableProfile"/> (Linux x64 CPU); keys ending in <c>.cuda</c> give the status on the Stable CUDA profile
/// <see cref="StableCudaProfile"/> (Linux x64, RTX 3090). <see cref="Profiles"/> repeats each Stable profile's statuses
/// keyed exactly as its registry manifest's <c>capabilities</c>, and a test keeps the two equal key for key.
/// </remarks>
public static class E5Capabilities
{
    /// <summary>The profile of this package's Stable contract (the README's "Stable contract" section).</summary>
    public const string StableProfile = "multilingual-e5-small/onnx-fp32-cpu";

    /// <summary>The CUDA profile of this package's Stable contract (the README's "Stable contract: CUDA" section).</summary>
    public const string StableCudaProfile = "multilingual-e5-small/onnx-fp32-cuda";

    /// <summary>
    /// Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c> and, for
    /// <see cref="CapabilityStatus.Experimental"/>, in the profile's gate evidence (<c>docs/catalog/README.md</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
        // Linux x64 CPU, Stable since 2026-10-15 (the README's Stable contract): every gate passed, and the weekly
        // qualification reruns G1 to G3 on the release sets and a one-hour soak, green. Evidence: G1 and G2 on the P-030 release set (232 rows: ids exact, embeddings within 1.46e-7
        // of the reference against 1e-5, every padded batch equal to single calls), G3 on MIRACL (nDCG@10 0.7978), G4
        // one-hour soak (concurrent callers and timed-out requests, 1,368 load/unload cycles, plateau), G6 anonymous
        // download into an empty cache and offline run, G7 adapter tests and drill consumers.
        ["embedding.pretokenized.cpu"] = CapabilityStatus.Supported,
        // CUDA (use_tf32 0) on an RTX 3090, Stable since 2026-10-16 (the README's "Stable contract: CUDA"): G1 to G3 on the
        // release sets, G4 one-hour soak and G5 (binding round), G6, G7; the per-release GPU regression reruns the CUDA
        // tests (adapters included), G1 to G3 and a soak smoke test (every check green in run 38857860186).
        ["embedding.pretokenized.cuda"] = CapabilityStatus.Supported,
        ["embedding.text"] = CapabilityStatus.Supported,
        ["batch"] = CapabilityStatus.Supported,
        ["cancellation"] = CapabilityStatus.Supported,
        ["concurrent_calls"] = CapabilityStatus.Supported,
        ["streaming"] = CapabilityStatus.Unsupported,
        ["pair_input"] = CapabilityStatus.Unsupported,
        ["token_type_ids_nonzero"] = CapabilityStatus.Unsupported,
        ["truncation"] = CapabilityStatus.Unsupported,
        ["prefix_insertion"] = CapabilityStatus.Unsupported,
        ["tf32"] = CapabilityStatus.Unsupported,
        ["fp16"] = CapabilityStatus.Unsupported,
        ["quantized"] = CapabilityStatus.Unsupported,
        ["tensorrt"] = CapabilityStatus.Unsupported,
        ["persistence"] = CapabilityStatus.Unsupported,
        ["training"] = CapabilityStatus.Unsupported,
        ["asset_acquisition"] = CapabilityStatus.Supported,

        // Microsoft.Extensions.AI IEmbeddingGenerator, in Starlings.Models.E5.Extensions.AI (P-011). Supported on the
        // CPU profile (G7) and over CUDA (CudaAdapterTests, per release); the explicit input-type prefix option lives there, prefix_insertion stays unsupported here.
        ["extensions_ai.embedding_generator"] = CapabilityStatus.Supported,
        ["extensions_ai.embedding_generator.cuda"] = CapabilityStatus.Supported,
        ["extensions_ai.dimensions_truncation"] = CapabilityStatus.Unsupported,
        // Cross-request micro-batching (opt-in E5EmbeddingGeneratorOptions.MicroBatching): concurrent single requests share padded batches on one
        // dispatcher thread. Weight-free tests (coalescing, ordering, per-caller cancellation, backpressure, grouping,
        // disposal, failure isolation) and one weight-dependent test against direct calls within the padded-batch
        // tolerance; no benchmark adoption decision and no soak yet. Outputs depend on the concurrent traffic (within
        // tolerance, not bit for bit reproducible). Not part of any Stable profile's contract.
        ["extensions_ai.micro_batching"] = CapabilityStatus.Experimental,

        // ML.NET estimator/transformer, in Starlings.Models.E5.MLNet (P-012). Plan 15.2 badges: pipeline
        // composition (embedding_transform, CPU profile), save/load and portable asset packaging (persistence:
        // Save throws; the model is resolved from the asset lock at run time), training (none). Cursors have no
        // cancellation token.
        ["mlnet.embedding_transform"] = CapabilityStatus.Supported,
        ["mlnet.embedding_transform.cuda"] = CapabilityStatus.Supported,
        ["mlnet.batching"] = CapabilityStatus.Supported,
        ["mlnet.persistence"] = CapabilityStatus.Unsupported,
        ["mlnet.cancellation"] = CapabilityStatus.Unsupported,
        ["mlnet.training"] = CapabilityStatus.Unsupported,
        // Native AOT on Linux x64 CPU from the packed packages, results equal to JIT bit for bit (typed API and
        // Extensions.AI generator; docs/evidence/aot/, the release drill's step 12); not the ML.NET adapter.
        ["aot"] = CapabilityStatus.Supported,
        // Native AOT with the CUDA natives (tools/aot-check-ai.sh --cuda): one local run, no per-release regression.
        ["aot.cuda"] = CapabilityStatus.Experimental,
        ["mlnet.aot"] = CapabilityStatus.Unsupported,
        // Windows x64: build, weight-free tests and the clean-install drill without the model in CI; no model run, so
        // outside the Stable contract.
        ["windows"] = CapabilityStatus.Experimental,
    };

    /// <summary>
    /// Status per capability of each profile this package version publishes a contract for, keyed exactly as the
    /// profile's registry manifest (<c>model-pipeline/registry/&lt;profile&gt;/manifest.yaml</c>, <c>capabilities</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, CapabilityStatus>> Profiles { get; } =
        new Dictionary<string, IReadOnlyDictionary<string, CapabilityStatus>>(StringComparer.Ordinal)
        {
            [StableProfile] = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
            {
                ["batch"] = CapabilityStatus.Supported,
                ["streaming"] = CapabilityStatus.Unsupported,
                ["cancellation"] = CapabilityStatus.Supported,
                ["persistence"] = CapabilityStatus.Unsupported,
                ["training"] = CapabilityStatus.Unsupported,
                ["aot"] = CapabilityStatus.Supported,
                ["extensions_ai"] = CapabilityStatus.Supported,
                ["mlnet"] = CapabilityStatus.Supported,
            },
            [StableCudaProfile] = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
            {
                ["batch"] = CapabilityStatus.Supported,
                ["streaming"] = CapabilityStatus.Unsupported,
                ["cancellation"] = CapabilityStatus.Supported,
                ["persistence"] = CapabilityStatus.Unsupported,
                ["training"] = CapabilityStatus.Unsupported,
                // One local run with the CUDA natives; not in the per-release GPU regression.
                ["aot"] = CapabilityStatus.Experimental,
                ["extensions_ai"] = CapabilityStatus.Supported,
                ["mlnet"] = CapabilityStatus.Supported,
            },
        };
}
