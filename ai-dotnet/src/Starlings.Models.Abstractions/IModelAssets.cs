using Starlings.Acquisition;

namespace Starlings.Models;

/// <summary>
/// A model package's asset lock and explicit acquisition of its files, as one object: the same members as the
/// package's static <c>XModelAssets</c> class (for example <c>E5ModelAssets</c>), which exposes an instance as
/// <c>XModelAssets.Assets</c>. Lets setup code acquire the files of several models the same way.
/// </summary>
/// <typeparam name="TFiles">The package's verified file record (for example <c>E5ModelFiles</c>).</typeparam>
/// <remarks>
/// Nothing is downloaded unless <see cref="DownloadAsync"/> is called; inference never downloads.
/// </remarks>
public interface IModelAssets<TFiles>
{
    /// <summary>The immutable lock: repository, full commit SHA, and each file's size and SHA-256.</summary>
    ModelAssetLock Lock { get; }

    /// <summary>Explicitly downloads (or verifies already cached) files. See <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/>.</summary>
    /// <param name="options">Cache directory, endpoint and verification; defaults per <see cref="ModelAcquisitionOptions"/>.</param>
    /// <param name="progress">Optional per-file progress.</param>
    /// <param name="cancellationToken">Cancels the download.</param>
    Task<TFiles> DownloadAsync(ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default);

    /// <summary>Offline: the cached files, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    /// <param name="options">Cache directory and verification.</param>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    TFiles OpenLocal(ModelAcquisitionOptions? options = null);

    /// <summary>Verifies an existing snapshot directory (for example a Hub snapshot) against <see cref="Lock"/>. Never downloads.</summary>
    /// <param name="directory">The snapshot directory.</param>
    /// <param name="verification">How cached files are checked; SHA-256 by default.</param>
    TFiles OpenDirectory(string directory, CacheVerification verification = CacheVerification.Sha256);
}
