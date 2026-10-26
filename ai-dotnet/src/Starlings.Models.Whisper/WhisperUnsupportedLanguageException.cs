namespace Starlings.Models.Whisper;

/// <summary>
/// A language code that is not one of the model's language tokens (<see cref="WhisperModel.LanguageToId"/>). Derives
/// from <see cref="ArgumentException"/>, so callers that catch argument errors still catch it.
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

    /// <summary>The rejected language code, as given.</summary>
    public string? Language { get; init; }

    internal static WhisperUnsupportedLanguageException For(string language, string? paramName) =>
        new($"'{language}' is not a language code of this model. Use a code of WhisperModel.LanguageToId (for example \"en\"), or null for the untested auto-detect prompt.", paramName)
        {
            Language = language,
        };
}
