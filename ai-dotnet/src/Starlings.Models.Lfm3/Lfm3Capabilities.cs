namespace Starlings.Models.Lfm3;

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
/// <see cref="All"/> is package-wide. LFM3 is new and not admitted to the registry; nothing is Supported, and everything
/// not listed here is Unsupported.
/// </remarks>
public static class Lfm3Capabilities
{
    /// <summary>The candidate profile of this package on CPU: the fp32 graph. Experimental, not registered.</summary>
    public const string CpuProfile = "lfm3-1.7b/onnx-fp32-cpu";

    /// <summary>The candidate profile on CUDA (RTX 3090): the fp16 graph. Untested, not registered.</summary>
    public const string CudaProfile = "lfm3-1.7b/onnx-fp16-cuda";

    /// <summary>Status per capability.</summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
        // Greedy fp32 CPU generation compared with the PyTorch reference and ONNX Runtime in Python (development evidence).
        ["generation.greedy"] = CapabilityStatus.Experimental,
        ["generation.fp16"] = CapabilityStatus.Untested,
        ["generation.cuda"] = CapabilityStatus.Untested,
        ["chat_template.text"] = CapabilityStatus.Experimental,
        ["chat_template.thinking_switch"] = CapabilityStatus.Experimental,
        ["streaming"] = CapabilityStatus.Experimental,
        ["stop_sequences"] = CapabilityStatus.Experimental,
        ["prefill_chunking"] = CapabilityStatus.Experimental,
        ["cancellation"] = CapabilityStatus.Experimental,
        ["concurrent_calls"] = CapabilityStatus.Untested,
        ["asset_acquisition"] = CapabilityStatus.Experimental,
        ["long_context"] = CapabilityStatus.Untested,   // beyond the evidence prompts; the checkpoint has 262,144 positions
        ["sampling"] = CapabilityStatus.Unsupported,
        ["logprobs"] = CapabilityStatus.Unsupported,
        ["guided_decoding"] = CapabilityStatus.Unsupported,
        ["tool_calling"] = CapabilityStatus.Unsupported,
        ["reasoning_parsing"] = CapabilityStatus.Unsupported,
        ["vision"] = CapabilityStatus.Unsupported,
        ["audio"] = CapabilityStatus.Unsupported,
        ["prefix_reuse"] = CapabilityStatus.Unsupported,
        ["batch"] = CapabilityStatus.Unsupported,
        ["quantized"] = CapabilityStatus.Unsupported,
        ["extensions_ai"] = CapabilityStatus.Unsupported,
        ["aot"] = CapabilityStatus.Untested,
    };
}
