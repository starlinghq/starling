namespace Starlings.Models;

/// <summary>
/// An input has more tokens than the model accepts, and the profile rejects it rather than truncating it. Nothing was
/// computed for the call. The base of the packages' own length exceptions (for example
/// <c>EmbeddingGemma2InputTooLongException</c>, <c>Gliner2InputTooLongException</c>), so one <c>catch</c> covers them.
/// </summary>
/// <remarks>
/// It is an <see cref="ArgumentException"/>, as the package exceptions were before, so existing handlers keep
/// working. The limit is <see cref="MaxTokens"/> in every package.
/// </remarks>
public class InputTooLongException : ArgumentException
{
    /// <summary>Creates the exception.</summary>
    public InputTooLongException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    /// <param name="message">The message.</param>
    public InputTooLongException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    /// <param name="message">The message.</param>
    /// <param name="innerException">The underlying failure.</param>
    public InputTooLongException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Creates the exception for input <paramref name="inputIndex"/> with a message and the parameter that carried the input.</summary>
    /// <param name="message">The message.</param>
    /// <param name="paramName">The parameter that carried the input, or null.</param>
    /// <param name="inputIndex">Position of the rejected input in the call.</param>
    /// <param name="tokenCount">Tokens in the rejected input.</param>
    /// <param name="maxTokens">The limit in effect.</param>
    public InputTooLongException(string message, string? paramName, int inputIndex, int tokenCount, int maxTokens)
        : base(message, paramName)
    {
        InputIndex = inputIndex;
        TokenCount = tokenCount;
        MaxTokens = maxTokens;
    }

    /// <summary>Position of the rejected input in the call (0 for single-input calls).</summary>
    public int InputIndex { get; init; }

    /// <summary>Tokens the rejected input has or needs, special tokens and any prompt or prefix included.</summary>
    public int TokenCount { get; init; }

    /// <summary>The limit in effect.</summary>
    public int MaxTokens { get; init; }

    /// <summary>
    /// True when the input was rejected from its length alone, without tokenizing it all; <see cref="TokenCount"/> is
    /// then a lower bound of the tokens it needs. False (default) when <see cref="TokenCount"/> is exact.
    /// </summary>
    public bool TokenCountIsLowerBound { get; init; }
}
