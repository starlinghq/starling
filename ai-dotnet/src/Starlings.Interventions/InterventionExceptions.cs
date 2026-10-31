namespace Starlings.Interventions;

/// <summary>
/// An intervention profile, manifest, selection or directions file violates the contract: malformed JSON, an unknown,
/// missing or duplicate property, an unknown schema version, operator, target or dtype, a value out of range, or a
/// directions file of the wrong size or with non-finite values.
/// </summary>
public sealed class InterventionContractException : Exception
{
    /// <summary>Creates the exception.</summary>
    public InterventionContractException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public InterventionContractException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying cause.</summary>
    public InterventionContractException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Creates the exception for the value at a JSON path (for example <c>sites[0].rank</c>).</summary>
    public InterventionContractException(string jsonPath, string message)
        : base(string.IsNullOrEmpty(jsonPath) ? message : $"{jsonPath}: {message}")
    {
        JsonPath = jsonPath;
    }

    /// <summary>The JSON path of the offending value, or <see langword="null"/> when the error is not about one value.</summary>
    public string? JsonPath { get; }
}

/// <summary>
/// Two identities that must agree do not: a SHA-256 of a file differs from the recorded one, a profile is applied to a
/// base graph other than the one it names, or a manifest belongs to another profile.
/// </summary>
public sealed class InterventionMismatchException : Exception
{
    /// <summary>Creates the exception.</summary>
    public InterventionMismatchException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public InterventionMismatchException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying cause.</summary>
    public InterventionMismatchException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
