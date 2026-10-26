namespace Starlings.Models.Whisper;

/// <summary>Support status of one capability. A missing entry never means supported.</summary>
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
/// <see cref="All"/> is package-wide. Whisper is new and not yet qualified; the transcription path is Experimental and
/// everything not listed here is Unsupported.
/// </remarks>
public static class WhisperCapabilities
{
    /// <summary>The profile of this package's Stable contract (see README).</summary>
    public const string StableProfile = "whisper-large-v3-turbo/onnx-fp16-cpu";

    /// <summary>Status per capability.</summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
        // Log-mel spectrogram, encoder and greedy decoder verified bit-exact against the reference; not yet qualified.
        ["transcription"] = CapabilityStatus.Experimental,
        ["transcription.cuda"] = CapabilityStatus.Untested,
        // Auto-detection prompts with <|startoflm|>; transformers instead picks the most likely language token after
        // <|startoftranscript|>. No reference covers it (the onboarding contract allows explicit languages only).
        ["language_detection"] = CapabilityStatus.Untested,
        ["cancellation"] = CapabilityStatus.Experimental,
        ["concurrent_calls"] = CapabilityStatus.Untested,
        ["asset_acquisition"] = CapabilityStatus.Experimental,
        ["translation"] = CapabilityStatus.Unsupported,
        ["timestamps"] = CapabilityStatus.Unsupported,
        ["chunking"] = CapabilityStatus.Unsupported,
        ["streaming"] = CapabilityStatus.Unsupported,
        ["batch"] = CapabilityStatus.Unsupported,
        ["persistence"] = CapabilityStatus.Unsupported,
        ["training"] = CapabilityStatus.Unsupported,
        ["quantized"] = CapabilityStatus.Unsupported,
        ["aot"] = CapabilityStatus.Untested,
        ["extensions_ai"] = CapabilityStatus.Unsupported,
        ["mlnet"] = CapabilityStatus.Unsupported,
    };
}
