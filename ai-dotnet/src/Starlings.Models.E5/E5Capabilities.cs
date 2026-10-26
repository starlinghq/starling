using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.E5;

/// <summary>Support status of one capability (plan section 9). A missing entry never means supported.</summary>
[Obsolete("Use Starlings.Models.CapabilityStatus; removed in the next minor version.")]
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
/// <see cref="Statuses"/> is package-wide. Keys without a provider suffix give the status on the Stable CPU profile
/// <see cref="StableProfile"/> (Linux x64 CPU); keys ending in <c>.cuda</c> give the status on the Stable CUDA profile
/// <see cref="StableCudaProfile"/> (Linux x64, RTX 3090). <see cref="ProfileCapabilities"/> holds the
/// capability record of every profile the package serves, generated from the registry manifests; a test keeps the
/// package-wide table consistent with the Stable profiles' records.
/// </remarks>
public static class E5Capabilities
{
    /// <summary>The profile of this package's Stable contract (the README's "Stable contract" section).</summary>
    public const string StableProfile = "multilingual-e5-small/onnx-fp32-cpu";

    /// <summary>The CUDA profile of this package's Stable contract (the README's "Stable contract: CUDA" section).</summary>
    public const string StableCudaProfile = "multilingual-e5-small/onnx-fp32-cuda";

    /// <summary>
    /// Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c> and, for
    /// <see cref="Status.Experimental"/>, in the profile's gate evidence (<c>docs/catalog/README.md</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, Status> Statuses { get; } = new Dictionary<string, Status>(StringComparer.Ordinal)
    {
        // Linux x64 CPU, Stable since 2026-10-15 (the README's Stable contract): every gate passed, and the weekly
        // qualification reruns G1 to G3 on the release sets and a one-hour soak, green. Evidence: G1 and G2 on the P-030 release set (232 rows: ids exact, embeddings within 1.46e-7
        // of the reference against 1e-5, every padded batch equal to single calls), G3 on MIRACL (nDCG@10 0.7978), G4
        // one-hour soak (concurrent callers and timed-out requests, 1,368 load/unload cycles, plateau), G6 anonymous
        // download into an empty cache and offline run, G7 adapter tests and drill consumers.
        ["embedding.pretokenized.cpu"] = Status.Supported,
        // CUDA (use_tf32 0) on an RTX 3090, Stable since 2026-10-16 (the README's "Stable contract: CUDA"): G1 to G3 on the
        // release sets, G4 one-hour soak and G5 (binding round), G6, G7; the per-release GPU regression reruns the CUDA
        // tests (adapters included), G1 to G3 and a soak smoke test (every check green in run 38857860186).
        ["embedding.pretokenized.cuda"] = Status.Supported,
        ["embedding.text"] = Status.Supported,
        ["batch"] = Status.Supported,
        // Opt-in explicit-batch chunking, E5EncoderOptions.MaxBatchTokens and SortByLength: off by default, and the
        // defaults keep the qualified outputs bit for bit. With sort_by_length, outputs stay within the
        // padded-batch-vs-single bound in the CPU tests; no qualification run yet.
        ["batch.token_budget"] = Status.Experimental,
        ["batch.sort_by_length"] = Status.Experimental,
        ["cancellation"] = Status.Supported,
        ["concurrent_calls"] = Status.Supported,
        ["streaming"] = Status.Unsupported,
        ["pair_input"] = Status.Unsupported,
        ["token_type_ids_nonzero"] = Status.Unsupported,
        // Encoder-side truncation (the README's "truncation in the encoder"): E5Encoder rejects id sequences longer
        // than MaxSequenceLength. Text is truncated on the right to 512 by E5Tokenizer, as the reference and the
        // manifests' behavior.truncation (truncate-right) say; that is part of embedding.text, not of this key.
        ["truncation"] = Status.Unsupported,
        ["prefix_insertion"] = Status.Unsupported,
        ["tf32"] = Status.Unsupported,
        ["fp16"] = Status.Unsupported,
        ["quantized"] = Status.Unsupported,
        ["tensorrt"] = Status.Unsupported,
        ["persistence"] = Status.Unsupported,
        ["training"] = Status.Unsupported,
        ["asset_acquisition"] = Status.Supported,

        // Microsoft.Extensions.AI IEmbeddingGenerator, in Starlings.Models.E5.Extensions.AI (P-011). Supported on the
        // CPU profile (G7) and over CUDA (CudaAdapterTests, per release); the explicit input-type prefix option lives there, prefix_insertion stays unsupported here.
        ["extensions_ai.embedding_generator"] = Status.Supported,
        ["extensions_ai.embedding_generator.cuda"] = Status.Supported,
        ["extensions_ai.dimensions_truncation"] = Status.Unsupported,
        // Cross-request micro-batching (opt-in E5EmbeddingGeneratorOptions.MicroBatching): concurrent single requests share padded batches on one
        // dispatcher thread. Weight-free tests (coalescing, ordering, per-caller cancellation, backpressure, grouping,
        // disposal, failure isolation) and one weight-dependent test against direct calls within the padded-batch
        // tolerance; no benchmark adoption decision and no soak yet. Outputs depend on the concurrent traffic (within
        // tolerance, not bit for bit reproducible). Not part of any Stable profile's contract.
        ["extensions_ai.micro_batching"] = Status.Experimental,

        // ML.NET estimator/transformer, in Starlings.Models.E5.MLNet (P-012). Plan 15.2 badges: pipeline
        // composition (embedding_transform, CPU profile), save/load and portable asset packaging (persistence:
        // Save throws; the model is resolved from the asset lock at run time), training (none). Cursors have no
        // cancellation token.
        ["mlnet.embedding_transform"] = Status.Supported,
        ["mlnet.embedding_transform.cuda"] = Status.Supported,
        ["mlnet.batching"] = Status.Supported,
        ["mlnet.persistence"] = Status.Unsupported,
        ["mlnet.cancellation"] = Status.Unsupported,
        ["mlnet.training"] = Status.Unsupported,
        // Native AOT on Linux x64 CPU from the packed packages, results equal to JIT bit for bit (typed API and
        // Extensions.AI generator; docs/evidence/aot/, the release drill's step 12); not the ML.NET adapter.
        ["aot"] = Status.Supported,
        // Native AOT with the CUDA natives (tools/aot-check-ai.sh --cuda): one local run, no per-release regression.
        ["aot.cuda"] = Status.Experimental,
        ["mlnet.aot"] = Status.Unsupported,
        // Windows x64: build, weight-free tests and the clean-install drill without the model in CI; no model run, so
        // outside the Stable contract.
        ["windows"] = Status.Experimental,
    };

    /// <summary>
    /// <see cref="Statuses"/> in this package's former status enum. Equal key for key; the enum values have the same
    /// names and order.
    /// </summary>
    [Obsolete("Use E5Capabilities.Statuses; removed in the next minor version.")]
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = Legacy(Statuses);

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
    [Obsolete("Use E5Capabilities.ProfileCapabilities[profile].Capabilities; removed in the next minor version.")]
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, CapabilityStatus>> Profiles { get; } =
        ProfileCapabilities.Values
            .Where(p => p.Maturity == ProfileMaturity.Stable)
            .ToDictionary(p => p.ProfileId, p => (IReadOnlyDictionary<string, CapabilityStatus>)Legacy(p.Capabilities), StringComparer.Ordinal);

    [Obsolete("Converts to the former status enum for the obsolete members only.")]
    private static Dictionary<string, CapabilityStatus> Legacy(IReadOnlyDictionary<string, Status> statuses) =>
        statuses.ToDictionary(e => e.Key, e => (CapabilityStatus)(int)e.Value, StringComparer.Ordinal);
}
