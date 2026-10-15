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
/// <see cref="All"/> is package-wide. Keys without a provider suffix give the status on the Stable-contract profile
/// <see cref="StableProfile"/> (Linux x64 CPU); keys ending in <c>.cuda</c> give the status on the experimental CUDA
/// profile <c>multilingual-e5-small/onnx-fp32-cuda</c>. <see cref="Profiles"/> repeats the Stable-contract profile's
/// statuses keyed exactly as its registry manifest's <c>capabilities</c>, and a test keeps the two equal key for key.
/// </remarks>
public static class E5Capabilities
{
    /// <summary>The profile of this package's Stable contract (the README's "Stable contract" section).</summary>
    public const string StableProfile = "multilingual-e5-small/onnx-fp32-cpu";

    /// <summary>
    /// Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c> and, for
    /// <see cref="CapabilityStatus.Experimental"/>, in the profile's gate evidence (<c>docs/catalog/README.md</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
        // Linux x64 CPU passed every gate but stays experimental: the weekly hosted soak fails the growth rule (the
        // README's Stable contract). Evidence: G1 and G2 on the P-030 release set (232 rows: ids exact, embeddings within 1.46e-7
        // of the reference against 1e-5, every padded batch equal to single calls), G3 on MIRACL (nDCG@10 0.7978), G4
        // one-hour soak (concurrent callers and timed-out requests, 1,368 load/unload cycles, plateau), G6 anonymous
        // download into an empty cache and offline run, G7 adapter tests and drill consumers.
        ["embedding.pretokenized.cpu"] = CapabilityStatus.Experimental,
        ["embedding.pretokenized.cuda"] = CapabilityStatus.Experimental,
        ["embedding.text"] = CapabilityStatus.Experimental,
        ["batch"] = CapabilityStatus.Experimental,
        ["cancellation"] = CapabilityStatus.Experimental,
        ["concurrent_calls"] = CapabilityStatus.Experimental,
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
        ["asset_acquisition"] = CapabilityStatus.Experimental,

        // Microsoft.Extensions.AI IEmbeddingGenerator, in Starlings.Models.E5.Extensions.AI (P-011). Supported on the
        // CPU profile (G7), experimental over CUDA; the explicit input-type prefix option lives there, prefix_insertion stays unsupported here.
        ["extensions_ai.embedding_generator"] = CapabilityStatus.Experimental,
        ["extensions_ai.embedding_generator.cuda"] = CapabilityStatus.Experimental,
        ["extensions_ai.dimensions_truncation"] = CapabilityStatus.Unsupported,

        // ML.NET estimator/transformer, in Starlings.Models.E5.MLNet (P-012). Plan 15.2 badges: pipeline
        // composition (embedding_transform, CPU profile), save/load and portable asset packaging (persistence:
        // Save throws; the model is resolved from the asset lock at run time), training (none). Cursors have no
        // cancellation token.
        ["mlnet.embedding_transform"] = CapabilityStatus.Experimental,
        ["mlnet.embedding_transform.cuda"] = CapabilityStatus.Experimental,
        ["mlnet.batching"] = CapabilityStatus.Experimental,
        ["mlnet.persistence"] = CapabilityStatus.Unsupported,
        ["mlnet.cancellation"] = CapabilityStatus.Unsupported,
        ["mlnet.training"] = CapabilityStatus.Unsupported,
        // Native AOT on Linux x64 CPU from the packed packages, results equal to JIT bit for bit (typed API and
        // Extensions.AI generator; docs/evidence/aot/, the release drill's step 12); not the ML.NET adapter.
        ["aot"] = CapabilityStatus.Experimental,
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
                ["batch"] = CapabilityStatus.Experimental,
                ["streaming"] = CapabilityStatus.Unsupported,
                ["cancellation"] = CapabilityStatus.Experimental,
                ["persistence"] = CapabilityStatus.Unsupported,
                ["training"] = CapabilityStatus.Unsupported,
                ["aot"] = CapabilityStatus.Experimental,
                ["extensions_ai"] = CapabilityStatus.Experimental,
                ["mlnet"] = CapabilityStatus.Experimental,
            },
        };
}
