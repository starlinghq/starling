using Starlings.Runtime.Onnx;

namespace Starlings.Models.PaddleOcr;

/// <summary>Capability descriptor for this package version.</summary>
/// <remarks>
/// Statuses use the shared <see cref="Starlings.Models.CapabilityStatus"/>. PP-OCRv5 is new and not qualified: the CPU OCR path is Experimental (proposed evidence from the
/// <c>ppocr-onnx-cpu-v1</c> onboarding route) and everything not listed here is Unsupported.
/// </remarks>
public static class PaddleOcrCapabilities
{
    /// <summary>The proposed profile of this package: fp32 graphs on the CPU provider. Experimental.</summary>
    public const string CpuProfile = "ppocrv5-mobile/onnx-fp32-cpu";

    /// <summary>Status per capability.</summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
        // Detector preprocessing and crops equal the Paddle reference byte for byte on the onboarding corpus; graph
        // outputs equal Python ONNX Runtime; DB geometry is an owned port, matched by IoU. Not yet qualified.
        ["text_detection"] = CapabilityStatus.Experimental,
        ["text_recognition"] = CapabilityStatus.Experimental,
        ["ocr"] = CapabilityStatus.Experimental,
        // Simplified Chinese and English are in the evidence; the dictionary also covers Traditional Chinese and
        // Japanese, which no corpus image exercises.
        ["languages.zh_hans"] = CapabilityStatus.Experimental,
        ["languages.en"] = CapabilityStatus.Experimental,
        ["languages.zh_hant"] = CapabilityStatus.Untested,
        ["languages.ja"] = CapabilityStatus.Untested,
        ["rotated_text"] = CapabilityStatus.Experimental,
        ["cancellation"] = CapabilityStatus.Experimental,
        ["asset_acquisition"] = CapabilityStatus.Untested,
        ["cuda"] = CapabilityStatus.Untested,
        ["concurrent_calls"] = CapabilityStatus.Untested,
        ["aot"] = CapabilityStatus.Untested,
        // Separate PP-OCR modules, not implemented.
        ["textline_orientation"] = CapabilityStatus.Unsupported,
        ["document_orientation"] = CapabilityStatus.Unsupported,
        ["document_unwarping"] = CapabilityStatus.Unsupported,
        ["polygon_boxes"] = CapabilityStatus.Unsupported,
        ["word_boxes"] = CapabilityStatus.Unsupported,
        ["batched_recognition"] = CapabilityStatus.Unsupported,
        ["server_models"] = CapabilityStatus.Unsupported,
        ["image_decoding"] = CapabilityStatus.Unsupported,
        ["extensions_ai"] = CapabilityStatus.Unsupported,
        ["training"] = CapabilityStatus.Unsupported,
        ["quantized"] = CapabilityStatus.Unsupported,
    };
}

/// <summary>Options of <see cref="PaddleOcrEngine"/>. The pipeline constants are fixed by the contract.</summary>
public sealed record PaddleOcrOptions
{
    /// <summary>Detector session options.</summary>
    public OnnxSessionOptions DetectorSession { get; init; } = new();

    /// <summary>Recognizer session options.</summary>
    public OnnxSessionOptions RecognizerSession { get; init; } = new();
}
