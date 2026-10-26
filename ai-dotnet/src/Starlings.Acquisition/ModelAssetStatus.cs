namespace Starlings.Acquisition;

/// <summary>Overall state of a lock's files in a directory, from <see cref="ModelAssetStore.Check(ModelAssetLock)"/>.</summary>
public enum ModelAssetState
{
    /// <summary>Every file is present and passed the check.</summary>
    Complete,

    /// <summary>No file of the lock is present.</summary>
    Missing,

    /// <summary>Some files are missing; every present file passed the check.</summary>
    Incomplete,

    /// <summary>At least one present file failed the check (wrong size or digest, or unreadable); other files may also be missing.</summary>
    Invalid,
}

/// <summary>Why a present file failed its check.</summary>
public enum ModelAssetProblemKind
{
    /// <summary>The size differs from the lock (the file was not hashed).</summary>
    SizeMismatch,

    /// <summary>The size matches but the SHA-256 differs.</summary>
    DigestMismatch,

    /// <summary>The file exists but could not be read (permissions, a directory in the way, an I/O error).</summary>
    Unreadable,
}

/// <summary>One present file that failed its check. The same data as a <see cref="ModelAssetIntegrityException"/>, as a value.</summary>
/// <param name="Path">Path inside the repository.</param>
/// <param name="Kind">What is wrong.</param>
/// <param name="FilePath">Absolute local path that was checked.</param>
/// <param name="ExpectedSize">Size in the lock.</param>
/// <param name="ActualSize">Size found; -1 when the file could not be read.</param>
/// <param name="ExpectedSha256">SHA-256 in the lock.</param>
/// <param name="ActualSha256">SHA-256 found, or null when the file was not hashed.</param>
/// <param name="Message">The message <see cref="ModelAssetStore.OpenLocal(ModelAssetLock)"/> would throw for this file.</param>
public sealed record ModelAssetProblem(
    string Path, ModelAssetProblemKind Kind, string FilePath, long ExpectedSize, long ActualSize, string ExpectedSha256, string? ActualSha256, string Message);

/// <summary>
/// Non-throwing result of checking a lock's files in a directory (<see cref="ModelAssetStore.Check(ModelAssetLock)"/>,
/// <see cref="ModelAssetStore.CheckDirectory"/>). Never downloads and changes nothing.
/// </summary>
public sealed class ModelAssetStatus
{
    private readonly ModelAssetIntegrityException? _firstFailure;

    internal ModelAssetStatus(
        ModelAssetLock assetLock, string directory, CacheVerification verification, IReadOnlyList<string> missing,
        IReadOnlyList<ModelAssetProblem> problems, ModelAssetIntegrityException? firstFailure)
    {
        Lock = assetLock;
        SnapshotDirectory = directory;
        Verification = verification;
        MissingPaths = missing;
        Problems = problems;
        _firstFailure = firstFailure;
        State = problems.Count > 0 ? ModelAssetState.Invalid
            : missing.Count == 0 ? ModelAssetState.Complete
            : missing.Count == assetLock.Files.Count ? ModelAssetState.Missing
            : ModelAssetState.Incomplete;
    }

    /// <summary>The lock that was checked.</summary>
    public ModelAssetLock Lock { get; }

    /// <summary>Overall state.</summary>
    public ModelAssetState State { get; }

    /// <summary>True when <see cref="State"/> is <see cref="ModelAssetState.Complete"/>.</summary>
    public bool IsComplete => State == ModelAssetState.Complete;

    /// <summary>Directory that was checked (absolute).</summary>
    public string SnapshotDirectory { get; }

    /// <summary>How present files were checked.</summary>
    public CacheVerification Verification { get; }

    /// <summary>Paths (inside the repository) that are not present, in lock order.</summary>
    public IReadOnlyList<string> MissingPaths { get; }

    /// <summary>Present files that failed their check, in lock order.</summary>
    public IReadOnlyList<ModelAssetProblem> Problems { get; }

    /// <summary>Bytes of the lock still to be acquired: the sizes of the missing and failing files.</summary>
    public long MissingBytes
    {
        get
        {
            var paths = new HashSet<string>(MissingPaths, StringComparer.Ordinal);
            paths.UnionWith(Problems.Select(p => p.Path));
            return Lock.Files.Where(f => paths.Contains(f.Path)).Sum(f => f.Size);
        }
    }

    /// <summary>The verified snapshot when <see cref="IsComplete"/>; otherwise null.</summary>
    public ModelSnapshot? Snapshot => IsComplete ? new ModelSnapshot(Lock, SnapshotDirectory, Verification) : null;

    /// <summary>The integrity failure of the first failing file, for the throwing paths.</summary>
    internal ModelAssetIntegrityException? FirstFailure => _firstFailure;

    /// <inheritdoc />
    public override string ToString() =>
        $"{Lock.Name}: {State} ({MissingPaths.Count} missing, {Problems.Count} failing of {Lock.Files.Count} files in {SnapshotDirectory})";
}

/// <summary>
/// Per-call settings of <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, ModelDownloadOptions, CancellationToken)"/>.
/// Null members use the store's <see cref="ModelAcquisitionOptions"/>, so one store can be kept for the life of the
/// process and each call can still choose its own verification, progress, wait and concurrency.
/// </summary>
public sealed class ModelDownloadOptions
{
    /// <summary>Check applied to files already in the cache; null uses <see cref="ModelAcquisitionOptions.Verification"/>.</summary>
    public CacheVerification? Verification { get; init; }

    /// <summary>Optional progress (per file, with the lock totals).</summary>
    public IProgress<ModelAssetProgress>? Progress { get; init; }

    /// <summary>Longest wait for another downloader's lock; null uses <see cref="ModelAcquisitionOptions.LockWaitTimeout"/>.</summary>
    public TimeSpan? LockWaitTimeout { get; init; }

    /// <summary>Files acquired at the same time; null uses <see cref="ModelAcquisitionOptions.MaxConcurrentFiles"/>.</summary>
    public int? MaxConcurrentFiles { get; init; }
}

/// <summary>What a cache entry listed by <see cref="ModelAssetStore.PlanPrune"/> or <see cref="ModelAssetStore.RemovePartialDownloads"/> is.</summary>
public enum ModelCacheEntryKind
{
    /// <summary>A snapshot directory (<c>&lt;cache&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;commit SHA&gt;</c>) with all its files.</summary>
    Snapshot,

    /// <summary>A partial download under <c>&lt;cache&gt;/.downloads/</c>.</summary>
    PartialDownload,

    /// <summary>A per-file lock file under <c>&lt;cache&gt;/.downloads/</c>. A downloader creates it and never removes it.</summary>
    LockFile,
}

/// <summary>One file or directory of the cache.</summary>
/// <param name="Path">Absolute path.</param>
/// <param name="Kind">What it is.</param>
/// <param name="Bytes">Size in bytes (for a snapshot, the sum of its files).</param>
/// <param name="Repository">Repository of the layout path (<c>owner/name</c>).</param>
/// <param name="Revision">Commit SHA of the layout path.</param>
public sealed record ModelCacheEntry(string Path, ModelCacheEntryKind Kind, long Bytes, string Repository, string Revision);

/// <summary>
/// What <see cref="ModelAssetStore.Prune(ModelCachePrunePlan)"/> would remove, from <see cref="ModelAssetStore.PlanPrune"/>.
/// Listing changes nothing.
/// </summary>
public sealed class ModelCachePrunePlan
{
    internal ModelCachePrunePlan(string cacheDirectory, IReadOnlyList<ModelCacheEntry> entries)
    {
        CacheDirectory = cacheDirectory;
        Entries = entries;
    }

    /// <summary>The cache root the plan was made for.</summary>
    public string CacheDirectory { get; }

    /// <summary>Entries to remove, snapshots first, then the entries under <c>.downloads</c>.</summary>
    public IReadOnlyList<ModelCacheEntry> Entries { get; }

    /// <summary>Sum of <see cref="ModelCacheEntry.Bytes"/>.</summary>
    public long TotalBytes => Entries.Sum(e => e.Bytes);
}

/// <summary>Result of <see cref="ModelAssetStore.Prune(ModelCachePrunePlan)"/> or <see cref="ModelAssetStore.RemovePartialDownloads"/>.</summary>
public sealed class ModelCacheCleanup
{
    internal ModelCacheCleanup(IReadOnlyList<ModelCacheEntry> removed, IReadOnlyList<ModelCacheEntry> skipped)
    {
        Removed = removed;
        Skipped = skipped;
    }

    /// <summary>Entries removed.</summary>
    public IReadOnlyList<ModelCacheEntry> Removed { get; }

    /// <summary>
    /// Entries left in place because a downloader held their lock at that moment, or because removing them failed
    /// (for example a file open on Windows). Calling again later removes them.
    /// </summary>
    public IReadOnlyList<ModelCacheEntry> Skipped { get; }

    /// <summary>Sum of the removed entries' <see cref="ModelCacheEntry.Bytes"/>.</summary>
    public long BytesRemoved => Removed.Sum(e => e.Bytes);
}
