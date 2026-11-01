using Starlings.Models;

namespace Starlings.Interventions;

/// <summary>Capability descriptor of the intervention contract for this version.</summary>
/// <remarks>
/// Everything not listed here is Unsupported. This library holds the contract only; the graph compiler, the derived
/// artifact cache and model integration live elsewhere and carry their own status.
/// V1 evidence uses synthetic directions only: no behavioural profile has been evaluated.
/// </remarks>
public static class InterventionCapabilities
{
    /// <summary>Status per capability.</summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
        ["profile.schema_v1"] = CapabilityStatus.Experimental,           // strict parse, canonical serialization, unit tests
        ["manifest.schema_v1"] = CapabilityStatus.Experimental,
        ["operator.directional_projection"] = CapabilityStatus.Experimental,
        ["target.residual_post"] = CapabilityStatus.Experimental,
        ["directions.float32_raw"] = CapabilityStatus.Experimental,
        ["directions.float16"] = CapabilityStatus.Unsupported,           // rejected: V1 knows only float32
        ["directions.orthonormality_check"] = CapabilityStatus.Unsupported, // diagnostic only; the author's job
        ["operator.other"] = CapabilityStatus.Unsupported,               // unknown operators are rejected
        ["profiles.multiple_per_graph"] = CapabilityStatus.Unsupported,
        ["profiles.hot_swap"] = CapabilityStatus.Unsupported,
        ["profiles.behavioural"] = CapabilityStatus.Unsupported,         // no real behavioural profile ships or is evaluated
        ["access_control"] = CapabilityStatus.Unsupported,               // the host application decides who gets which profile
        ["aot"] = CapabilityStatus.Untested,                             // AOT analyzers on; no AOT publish run
    };
}
