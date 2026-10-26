using Starlings.Runtime.Onnx;

namespace Starlings.Models.Whisper;

/// <summary>Transcription options for <c>openai/whisper-large-v3-turbo</c>.</summary>
public sealed record WhisperOptions
{
    /// <summary>The default <see cref="Language"/>: English, the language the onboarding evidence covers.</summary>
    public const string DefaultLanguage = "en";

    /// <summary>
    /// Language code, a key of <see cref="WhisperModel.LanguageToId"/> (case-insensitive). Default
    /// <see cref="DefaultLanguage"/> (<c>"en"</c>). <see langword="null"/> opts in to auto-detection, which prompts with
    /// <c>&lt;|startoflm|&gt;</c> and is untested (<see cref="WhisperCapabilities"/>, <c>language_detection</c>).
    /// </summary>
    public string? Language { get; init; } = DefaultLanguage;

    /// <summary>Maximum number of generated tokens (excluding the prompt). Default 448.</summary>
    public int MaxLength { get; init; } = 448;

    /// <summary>When <see langword="true"/> (default), generate without timestamp tokens (<c>&lt;|notimestamps|&gt;</c>).</summary>
    public bool NoTimestamps { get; init; } = true;

    /// <summary>Encoder session options.</summary>
    public OnnxSessionOptions EncoderSession { get; init; } = new();

    /// <summary>Decoder session options.</summary>
    public OnnxSessionOptions DecoderSession { get; init; } = new();

    /// <exception cref="WhisperUnsupportedLanguageException"><see cref="Language"/> is not a language code of the model.</exception>
    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxLength, 1);
        WhisperModel.LanguageTokenId(Language, nameof(Language));
    }
}
