namespace Starlings.Runtime.Onnx;

/// <summary>
/// Base type of the exceptions this package throws when ONNX Runtime, an execution provider or a model cannot be
/// used: <see cref="OnnxRuntimeUnavailableException"/>, <see cref="OnnxProviderUnavailableException"/> and
/// <see cref="OnnxModelIntegrityException"/>. Catch it to handle all three in one clause. Argument validation still
/// throws <see cref="ArgumentException"/>, cancellation <see cref="OperationCanceledException"/>, and a failing run
/// ONNX Runtime's own <c>OnnxRuntimeException</c>.
/// </summary>
public class OnnxSessionException : Exception
{
    /// <summary>Creates the exception.</summary>
    public OnnxSessionException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public OnnxSessionException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public OnnxSessionException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>
/// The ONNX Runtime native library is missing, failed to load, or does not match the managed binding version.
/// </summary>
public sealed class OnnxRuntimeUnavailableException : OnnxSessionException
{
    /// <summary>Creates the exception.</summary>
    public OnnxRuntimeUnavailableException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public OnnxRuntimeUnavailableException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public OnnxRuntimeUnavailableException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>
/// The requested execution provider is not available: the loaded native package does not include it, its
/// native dependencies are missing, or the device could not be initialized. Never followed by a silent fallback.
/// </summary>
public sealed class OnnxProviderUnavailableException : OnnxSessionException
{
    /// <summary>Creates the exception.</summary>
    public OnnxProviderUnavailableException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public OnnxProviderUnavailableException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public OnnxProviderUnavailableException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>The model bytes do not match the expected SHA-256.</summary>
public sealed class OnnxModelIntegrityException : OnnxSessionException
{
    /// <summary>Creates the exception.</summary>
    public OnnxModelIntegrityException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public OnnxModelIntegrityException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public OnnxModelIntegrityException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
