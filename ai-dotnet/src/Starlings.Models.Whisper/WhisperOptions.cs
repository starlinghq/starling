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

    /// <summary>Full-context decoder session options.</summary>
    public OnnxSessionOptions DecoderSession { get; init; } = new();

    /// <summary>KV-cache decoder (<c>decoder_with_past_model_fp16.onnx</c>) session options.</summary>
    public OnnxSessionOptions DecoderWithPastSession { get; init; } = new();

    /// <summary>
    /// The decoder loop. <see langword="null"/> (default): the KV cache when the <c>decoder_with_past</c> graph is given
    /// (<see cref="WhisperModel.Create(WhisperModelFiles, WhisperOptions?)"/> always gives it), otherwise full context.
    /// <see cref="WhisperDecoderMode.FullContext"/> keeps the reference loop, which re-runs the decoder over every id at
    /// every step and does not load the KV-cache graph.
    /// </summary>
    public WhisperDecoderMode? DecoderMode { get; init; }

    /// <exception cref="WhisperUnsupportedLanguageException"><see cref="Language"/> is not a language code of the model.</exception>
    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxLength, 1);
        WhisperModel.LanguageTokenId(Language, nameof(Language));
    }
}

/// <summary>The decoder loop of <see cref="WhisperModel"/>.</summary>
public enum WhisperDecoderMode
{
    /// <summary>Every step re-runs the full-context decoder over all ids: the reference path.</summary>
    FullContext,

    /// <summary>
    /// One full-context run over the prompt, then one <c>decoder_with_past</c> run per token with cached keys and
    /// values. Logits differ from the full-context loop in the last bits (fp16 summation order); ids are checked equal.
    /// </summary>
    KvCache,
}
