namespace Starlings.Models.Lfm3.Interventions;

/// <summary>Capability descriptor of the LFM3 intervention integration for this version.</summary>
/// <remarks>
/// Everything not listed here is Unsupported. V1 evidence uses synthetic random orthonormal directions on the fp32 graph:
/// no behavioural profile has been evaluated, and nothing is Supported.
/// </remarks>
public static class Lfm3InterventionCapabilities
{
    /// <summary>Status per capability.</summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
        ["graph.fp32"] = CapabilityStatus.Experimental,             // site mapping lfm3-1.7b/fp32/1; synthetic-direction evidence
        ["graph.fp16"] = CapabilityStatus.Unsupported,              // model-pipeline has lfm3-1.7b/fp16/1; not verified or accepted here
        ["selection.per_request"] = CapabilityStatus.Experimental,  // strength fixed per sequence, passed on every call
        ["selection.mid_sequence_change"] = CapabilityStatus.Unsupported, // impossible by construction
        ["profiles.multiple_per_graph"] = CapabilityStatus.Experimental, // strength vector per sequence; synthetic two-profile evidence
        ["operator.directional_addition"] = CapabilityStatus.Experimental,
        ["profiles.hot_swap"] = CapabilityStatus.Unsupported,
        ["profiles.behavioural"] = CapabilityStatus.Unsupported,    // no real behavioural profile ships or is evaluated
        ["cuda"] = CapabilityStatus.Untested,
        ["aot"] = CapabilityStatus.Untested,                        // AOT analyzers on; no AOT publish run
    };
}
