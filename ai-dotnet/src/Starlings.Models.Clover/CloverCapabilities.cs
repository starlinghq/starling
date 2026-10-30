namespace Starlings.Models.Clover;

/// <summary>Support status of one capability. A missing entry never means supported.</summary>
public enum CapabilityStatus
{
    /// <summary>Qualified for the advertised maturity.</summary>
    Supported,

    /// <summary>Not available; requests for it are rejected.</summary>
    Unsupported,

    /// <summary>Implemented and tested against a reference, not yet qualified.</summary>
    Experimental,

    /// <summary>May work, but no evidence exists.</summary>
    Untested,
}

/// <summary>Capability descriptor for this package version.</summary>
/// <remarks>
/// <see cref="All"/> is package-wide. This package is the model definition only: no backend in this repository runs
/// Clover's block-diffusion decoder (<c>model-pipeline/admissions/clover.md</c>, "Execution route"), so generation is
/// Unsupported. Nothing is Supported, and everything not listed here is Unsupported.
/// </remarks>
public static class CloverCapabilities
{
    /// <summary>Status per capability.</summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
        // Template renderings and prompt ids compared with transformers 5.19.0 (development evidence).
        ["chat_template.text"] = CapabilityStatus.Experimental,
        ["chat_template.thinking_switch"] = CapabilityStatus.Experimental,
        ["chat_template.preserve_thinking"] = CapabilityStatus.Experimental,
        ["chat_template.add_non_thinking_prefix"] = CapabilityStatus.Experimental,
        ["checkpoint_lock"] = CapabilityStatus.Experimental,   // digests from the Hub; no runtime here loads the files
        ["generation"] = CapabilityStatus.Unsupported,         // block diffusion: no ONNX export, no ONNX Runtime GenAI model type
        ["generation.self_speculative"] = CapabilityStatus.Unsupported,
        ["sampling"] = CapabilityStatus.Unsupported,
        ["logprobs"] = CapabilityStatus.Unsupported,
        ["guided_decoding"] = CapabilityStatus.Unsupported,
        ["tool_calling"] = CapabilityStatus.Unsupported,       // the template's tool sections are rejected
        ["reasoning_parsing"] = CapabilityStatus.Unsupported,
        ["vision"] = CapabilityStatus.Unsupported,             // the checkpoint is text only
        ["extensions_ai"] = CapabilityStatus.Unsupported,
        ["aot"] = CapabilityStatus.Untested,
    };
}
