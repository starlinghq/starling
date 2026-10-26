namespace Starlings.Generation;

/// <summary>The prompt plus the requested new tokens exceed the model's context length.</summary>
public sealed class GenerationContextLengthException : Exception
{
    /// <summary>Creates the exception.</summary>
    public GenerationContextLengthException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public GenerationContextLengthException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public GenerationContextLengthException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Prompt tokens.</summary>
    public int PromptTokens { get; init; }

    /// <summary>Requested maximum new tokens.</summary>
    public int MaxNewTokens { get; init; }

    /// <summary>The model's context length.</summary>
    public int ContextLength { get; init; }
}
