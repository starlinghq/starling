using System.Net;

namespace Starlings.Acquisition;

/// <summary>Base class of every acquisition failure. Messages never contain the access token.</summary>
public class ModelAcquisitionException : Exception
{
    /// <summary>Creates the exception.</summary>
    public ModelAcquisitionException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public ModelAcquisitionException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public ModelAcquisitionException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Repository of the asset, when known.</summary>
    public string? Repository { get; init; }

    /// <summary>Commit SHA of the asset, when known.</summary>
    public string? Revision { get; init; }

    /// <summary>Path of the asset inside the repository, when the failure concerns one file.</summary>
    public string? AssetPath { get; init; }
}

/// <summary>
/// Offline resolution found no verified copy of one or more files. <see cref="Exception.Message"/>
/// names the missing files, the cache directory searched and how to download them explicitly. Verification completes
/// before this is thrown: when other files are present but fail their size or digest check, the message says so and
/// the first such <see cref="ModelAssetIntegrityException"/> is the <see cref="Exception.InnerException"/>.
/// </summary>
public sealed class ModelAssetMissingException : ModelAcquisitionException
{
    /// <summary>Creates the exception.</summary>
    public ModelAssetMissingException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public ModelAssetMissingException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public ModelAssetMissingException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Paths (inside the repository) that are missing.</summary>
    public IReadOnlyList<string> MissingPaths { get; init; } = [];

    /// <summary>Snapshot directory that was searched.</summary>
    public string? SnapshotDirectory { get; init; }
}

/// <summary>A file's size or SHA-256 differs from the lock: a corrupt cache entry, a truncated download or changed upstream bytes.</summary>
public sealed class ModelAssetIntegrityException : ModelAcquisitionException
{
    /// <summary>Creates the exception.</summary>
    public ModelAssetIntegrityException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public ModelAssetIntegrityException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public ModelAssetIntegrityException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Local file that failed verification (may already be deleted if it was a partial download).</summary>
    public string? FilePath { get; init; }

    /// <summary>Size in the lock.</summary>
    public long ExpectedSize { get; init; }

    /// <summary>Size found.</summary>
    public long ActualSize { get; init; }

    /// <summary>SHA-256 in the lock.</summary>
    public string? ExpectedSha256 { get; init; }

    /// <summary>SHA-256 found, or null when the size already differed and the file was not hashed.</summary>
    public string? ActualSha256 { get; init; }
}

/// <summary>
/// The server refused access (HTTP 401 or 403): the repository is gated or private and needs an access token from an
/// account that accepted its license, or the token is invalid. The Hub also answers 401 for repositories that do not exist.
/// </summary>
public sealed class ModelAssetAccessDeniedException : ModelAcquisitionException
{
    /// <summary>Creates the exception.</summary>
    public ModelAssetAccessDeniedException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public ModelAssetAccessDeniedException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public ModelAssetAccessDeniedException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>HTTP status returned.</summary>
    public HttpStatusCode StatusCode { get; init; }

    /// <summary>Whether a token was sent with the request.</summary>
    public bool TokenProvided { get; init; }
}

/// <summary>The server answered 404: the revision or the file does not exist in the repository.</summary>
public sealed class ModelAssetNotFoundException : ModelAcquisitionException
{
    /// <summary>Creates the exception.</summary>
    public ModelAssetNotFoundException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public ModelAssetNotFoundException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public ModelAssetNotFoundException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>A network or server failure persisted after all retries. A partial file is kept so a later call resumes.</summary>
public sealed class ModelAssetDownloadException : ModelAcquisitionException
{
    /// <summary>Creates the exception.</summary>
    public ModelAssetDownloadException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public ModelAssetDownloadException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public ModelAssetDownloadException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Last HTTP status, when the failure was an HTTP response.</summary>
    public HttpStatusCode? StatusCode { get; init; }

    /// <summary>Number of attempts made.</summary>
    public int Attempts { get; init; }
}
