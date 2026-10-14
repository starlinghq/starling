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
public static class E5Capabilities
{
    /// <summary>
    /// Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c>.
    /// </summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
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

        // Microsoft.Extensions.AI IEmbeddingGenerator, in Starlings.Models.E5.Extensions.AI (P-011). Tested
        // over the CPU and CUDA profiles; the explicit input-type prefix option lives there, prefix_insertion stays unsupported here.
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
        // Native AOT on Linux x64, CPU and CUDA natives (docs/evidence/aot/, docs/evidence/capability-closure-2026-10-14/); not the ML.NET adapter.
        ["aot"] = CapabilityStatus.Experimental,
        ["mlnet.aot"] = CapabilityStatus.Unsupported,
        ["windows"] = CapabilityStatus.Untested,
    };
}
