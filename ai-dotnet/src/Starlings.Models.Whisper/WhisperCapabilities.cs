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
    /// <summary>The registered profile of this package: fp16 graphs on the CPU provider. Experimental, not Stable.</summary>
    public const string CpuProfile = "whisper-large-v3-turbo/onnx-fp16-cpu";

    /// <summary>The registered profile, under its old name; the profile is Experimental, not Stable.</summary>
    [Obsolete("Use CpuProfile; removed in the next minor version.")]
    public const string StableProfile = CpuProfile;

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
        // Segment-level timestamps (TranscribeWithTimestamps): window ids, segment boundaries and texts equal the
        // transformers reference and Python ONNX Runtime on the route corpus; not yet qualified.
        ["timestamps"] = CapabilityStatus.Experimental,
        // Word-level timestamps need the alignment heads' cross-attention weights, which the pinned graphs do not export.
        ["word_timestamps"] = CapabilityStatus.Unsupported,
        // Long-form audio over 30 s: transformers' sequential long-form generation (whole-clip spectrogram, 30 s windows,
        // seek to the last closing timestamp), greedy; ids, segment times and texts equal the reference on the route corpus.
        ["chunking"] = CapabilityStatus.Experimental,
        // The KV-cache decoder loop (decoder_with_past, the default with WhisperModelFiles): ids, segments and transcripts
        // equal the full-context loop on the route corpus; logits within the route's kv_logits_max_abs
        // (fp16 summation order differs between the two graphs, so they are not bit-identical).
        ["kv_cache"] = CapabilityStatus.Experimental,
        // transformers' long-form fallbacks and conditioning are not implemented: decoding never re-runs a window at a
        // higher temperature, never skips a window as silence, and never prompts with earlier text.
        ["temperature_fallback"] = CapabilityStatus.Unsupported,
        ["compression_ratio_threshold"] = CapabilityStatus.Unsupported,
        ["logprob_threshold"] = CapabilityStatus.Unsupported,
        ["no_speech_detection"] = CapabilityStatus.Unsupported,
        ["condition_on_previous_text"] = CapabilityStatus.Unsupported,
        ["streaming"] = CapabilityStatus.Unsupported,
        ["batch"] = CapabilityStatus.Unsupported,
        ["persistence"] = CapabilityStatus.Unsupported,
        ["training"] = CapabilityStatus.Unsupported,
        ["quantized"] = CapabilityStatus.Unsupported,
        ["aot"] = CapabilityStatus.Untested,
        ["extensions_ai"] = CapabilityStatus.Unsupported,
        ["mlnet"] = CapabilityStatus.Unsupported,
    };

    /// <summary>
    /// The capability record of every registry profile this package serves (<see cref="ModelProfileCapabilities"/>),
    /// generated from the profiles' registry manifests by <c>model-pipeline catalog</c>, keyed by profile id. Each
    /// record's <see cref="ModelProfileCapabilities.Capabilities"/> is the manifest's <c>capabilities</c> map key for key;
    /// the package-wide <see cref="All"/> adds finer keys.
    /// </summary>
    public static IReadOnlyDictionary<string, ModelProfileCapabilities> ProfileCapabilities => GeneratedProfileCapabilities.Profiles;
}
