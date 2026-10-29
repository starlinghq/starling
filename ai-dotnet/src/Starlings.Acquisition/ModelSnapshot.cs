namespace Starlings.Acquisition;

/// <summary>A local directory whose files were verified against a lock. Paths are absolute.</summary>
public sealed class ModelSnapshot
{
    internal ModelSnapshot(ModelAssetLock assetLock, string directory, CacheVerification verification)
    {
        Lock = assetLock;
        Directory = directory;
        Verification = verification;
    }

    /// <summary>The lock the files were verified against.</summary>
    public ModelAssetLock Lock { get; }

    /// <summary>Snapshot directory (absolute).</summary>
    public string Directory { get; }

    /// <summary>How the files were verified when this snapshot was resolved.</summary>
    public CacheVerification Verification { get; }

    /// <summary>Absolute local path of <paramref name="path"/>; throws when the lock does not contain it.</summary>
    public string GetFilePath(string path) => ModelAssetStore.LocalPath(Directory, Lock.GetFile(path).Path);

    /// <summary>
    /// The SHA-256 of <paramref name="path"/> when this snapshot hashed every file against the lock
    /// (<see cref="Verification"/> is <see cref="CacheVerification.Sha256"/>, or the file was downloaded and hashed in
    /// transit); otherwise <see langword="null"/>. Pass it as <c>OnnxSessionOptions.KnownModelSha256</c> so session
    /// creation does not read the file a second time. The value is the lock's digest, which the bytes matched when the
    /// snapshot was resolved; later changes to the file are not detected.
    /// </summary>
    /// <exception cref="KeyNotFoundException">The lock does not contain <paramref name="path"/>.</exception>
    public string? GetVerifiedSha256(string path)
    {
        var file = Lock.GetFile(path);
        return Verification == CacheVerification.Sha256 ? file.Sha256 : null;
    }
}
