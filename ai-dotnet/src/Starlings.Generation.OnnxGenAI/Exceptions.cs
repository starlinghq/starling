namespace Starlings.Generation.OnnxGenAI;

/// <summary>
/// The ONNX Runtime GenAI native library is missing or failed to load. The application references exactly one native
/// package: <c>Microsoft.ML.OnnxRuntimeGenAI</c> (CPU) or <c>Microsoft.ML.OnnxRuntimeGenAI.Cuda</c>.
/// </summary>
public sealed class GenAIRuntimeUnavailableException : Exception
{
    /// <summary>Creates the exception.</summary>
    public GenAIRuntimeUnavailableException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public GenAIRuntimeUnavailableException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public GenAIRuntimeUnavailableException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>
/// The requested execution provider could not be used: the loaded native package does not include it, its native
/// dependencies (CUDA runtime, cuBLAS) are missing, or the device could not be initialized. Never followed by a silent
/// fallback to another provider.
/// </summary>
public sealed class GenAIProviderUnavailableException : Exception
{
    /// <summary>Creates the exception.</summary>
    public GenAIProviderUnavailableException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public GenAIProviderUnavailableException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public GenAIProviderUnavailableException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>ONNX Runtime GenAI rejected the model directory, its configuration or a generation step.</summary>
public sealed class GenAIModelException : Exception
{
    /// <summary>Creates the exception.</summary>
    public GenAIModelException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public GenAIModelException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public GenAIModelException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

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
