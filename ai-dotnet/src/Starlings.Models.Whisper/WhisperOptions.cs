using Starlings.Runtime.Onnx;

namespace Starlings.Models.Whisper;

/// <summary>Transcription options for <c>openai/whisper-large-v3-turbo</c>.</summary>
public sealed record WhisperOptions
{
    /// <summary>Language code (for example <c>"en"</c>), or <see langword="null"/> for auto-detection.</summary>
    public string? Language { get; init; }

    /// <summary>Maximum number of generated tokens (excluding the prompt). Default 448.</summary>
    public int MaxLength { get; init; } = 448;

    /// <summary>When <see langword="true"/> (default), generate without timestamp tokens (<c>&lt;|notimestamps|&gt;</c>).</summary>
    public bool NoTimestamps { get; init; } = true;

    /// <summary>Encoder session options.</summary>
    public OnnxSessionOptions EncoderSession { get; init; } = new();

    /// <summary>Decoder session options.</summary>
    public OnnxSessionOptions DecoderSession { get; init; } = new();

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxLength, 1);
    }
}
