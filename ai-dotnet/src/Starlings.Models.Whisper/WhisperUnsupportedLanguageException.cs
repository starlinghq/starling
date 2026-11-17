namespace Starlings.Models.Whisper;

/// <summary>
/// A language code that is not one of the model's language tokens (<see cref="WhisperModel.LanguageToId"/>), or no
/// language at all: language detection is unsupported. Derives from <see cref="ArgumentException"/>, so callers that
/// catch argument errors still catch it.
/// </summary>
public sealed class WhisperUnsupportedLanguageException : ArgumentException
{
    /// <summary>Creates the exception.</summary>
    public WhisperUnsupportedLanguageException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public WhisperUnsupportedLanguageException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public WhisperUnsupportedLanguageException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Creates the exception with a message and the name of the parameter that held the code.</summary>
    public WhisperUnsupportedLanguageException(string message, string? paramName)
        : base(message, paramName)
    {
    }

    /// <summary>The rejected language code, as given; <see langword="null"/> when no language was given.</summary>
    public string? Language { get; init; }

    internal static WhisperUnsupportedLanguageException For(string language, string? paramName) =>
        new($"'{language}' is not a language code of this model. Use a code of WhisperModel.LanguageToId (for example \"en\").", paramName)
        {
            Language = language,
        };

    internal static WhisperUnsupportedLanguageException ForDetection(string? paramName) =>
        new("A language code is required: language detection is unsupported (starlinghq/starling#332). Use a code of WhisperModel.LanguageToId (for example \"en\").", paramName);
}
