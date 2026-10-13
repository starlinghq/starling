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
}
