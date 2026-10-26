namespace Starlings.Acquisition;

/// <summary>Cache maintenance: removing partial downloads and stale lock files, and pruning revisions no lock needs.</summary>
public sealed partial class ModelAssetStore
{
    private const string PartialSuffix = ".partial";
    private const string LockSuffix = ".lock";

    /// <summary>
    /// Removes the partial downloads and lock files of <paramref name="assetLock"/>'s files under
    /// <c>&lt;cache&gt;/.downloads/</c>, honouring each file's lock: a file whose lock another downloader holds (in this
    /// process or another) is left in place and listed in <see cref="ModelCacheCleanup.Skipped"/>. Verified files in the
    /// snapshot directory are never touched. A later download of a removed partial file starts from zero. Blocking file I/O.
    /// </summary>
    public ModelCacheCleanup RemovePartialDownloads(ModelAssetLock assetLock)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(assetLock);
        string work = WorkDirectory(GetSnapshotDirectory(assetLock));
        var removed = new List<ModelCacheEntry>();
        var skipped = new List<ModelCacheEntry>();
        foreach (var file in assetLock.Files)
        {
            string stem = LocalPath(work, file.Path);
            RemoveUnderLock(stem, assetLock.Repository, assetLock.Revision, removePartial: true, removed, skipped);
        }

        RemoveEmptyDirectories(work, Path.Combine(CacheDirectory, DownloadsDirectoryName));
        return new ModelCacheCleanup(removed.AsReadOnly(), skipped.AsReadOnly());
    }

    /// <summary>
    /// Lists what <see cref="Prune(ModelCachePrunePlan)"/> would remove, keeping every file of <paramref name="keepLocks"/>.
    /// Changes nothing. The plan holds: every snapshot directory <c>&lt;owner&gt;/&lt;name&gt;/&lt;commit SHA&gt;</c> whose
    /// repository and commit no kept lock names; under <c>.downloads</c>, every partial download that is not of a kept
    /// lock's file, and every lock file (a downloader recreates one when it needs it). Only paths in the cache layout are
    /// listed (a third-level directory named by a 40-character lowercase SHA); anything else under the cache root is left
    /// alone, and a kept snapshot's files are never listed. Pass every lock that any user of this cache root still needs.
    /// </summary>
    public ModelCachePrunePlan PlanPrune(IEnumerable<ModelAssetLock> keepLocks)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(keepLocks);
        var kept = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var keep in keepLocks)
        {
            ArgumentNullException.ThrowIfNull(keep, nameof(keepLocks));
            string key = keep.Repository + "@" + keep.Revision;
            if (!kept.TryGetValue(key, out var paths))
            {
                kept[key] = paths = new HashSet<string>(StringComparer.Ordinal);
            }

            paths.UnionWith(keep.Files.Select(f => f.Path));
        }

        var entries = new List<ModelCacheEntry>();
        foreach (var (directory, repository, revision) in LayoutDirectories(CacheDirectory, skipDownloads: true))
        {
            if (!kept.ContainsKey(repository + "@" + revision))
            {
                entries.Add(new ModelCacheEntry(directory, ModelCacheEntryKind.Snapshot, DirectorySize(directory), repository, revision));
            }
        }

        string downloads = Path.Combine(CacheDirectory, DownloadsDirectoryName);
        foreach (var (directory, repository, revision) in LayoutDirectories(downloads, skipDownloads: false))
        {
            kept.TryGetValue(repository + "@" + revision, out var keptPaths);
            foreach (string path in Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
            {
                string relative = Path.GetRelativePath(directory, path).Replace(Path.DirectorySeparatorChar, '/');
                if (relative.EndsWith(LockSuffix, StringComparison.Ordinal))
                {
                    entries.Add(new ModelCacheEntry(path, ModelCacheEntryKind.LockFile, FileSize(path), repository, revision));
                }
                else if (relative.EndsWith(PartialSuffix, StringComparison.Ordinal)
                    && keptPaths?.Contains(relative[..^PartialSuffix.Length]) != true)
                {
                    entries.Add(new ModelCacheEntry(path, ModelCacheEntryKind.PartialDownload, FileSize(path), repository, revision));
                }
            }
        }

        return new ModelCachePrunePlan(CacheDirectory, entries.AsReadOnly());
    }

    /// <summary>
    /// Removes the entries of <paramref name="plan"/> (from <see cref="PlanPrune"/> on a store with the same cache root).
    /// Each entry is checked again when it is removed: a partial download or lock file whose lock another downloader
    /// holds, and a snapshot whose revision has a lock file held under <c>.downloads</c>, are left in place and listed in
    /// <see cref="ModelCacheCleanup.Skipped"/>. Empty layout directories left behind are removed. Run it when no
    /// download of a revision being removed is starting: a download that has not yet taken its first lock is not seen.
    /// Blocking file I/O.
    /// </summary>
    /// <exception cref="ArgumentException">The plan was made for another cache root.</exception>
    public ModelCacheCleanup Prune(ModelCachePrunePlan plan)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(plan);
        if (!string.Equals(plan.CacheDirectory, CacheDirectory, StringComparison.Ordinal))
        {
            throw new ArgumentException($"The plan was made for the cache {plan.CacheDirectory}, not {CacheDirectory}.", nameof(plan));
        }

        string downloads = Path.Combine(CacheDirectory, DownloadsDirectoryName);
        var removed = new List<ModelCacheEntry>();
        var skipped = new List<ModelCacheEntry>();
        foreach (var entry in plan.Entries)
        {
            switch (entry.Kind)
            {
                case ModelCacheEntryKind.Snapshot:
                    RemoveSnapshot(entry, downloads, removed, skipped);
                    break;
                case ModelCacheEntryKind.PartialDownload:
                case ModelCacheEntryKind.LockFile:
                    string suffix = entry.Kind == ModelCacheEntryKind.PartialDownload ? PartialSuffix : LockSuffix;
                    string stem = entry.Path[..^suffix.Length];
                    RemoveUnderLock(stem, entry.Repository, entry.Revision, removePartial: entry.Kind == ModelCacheEntryKind.PartialDownload, removed, skipped);
                    break;
            }
        }

        foreach (var entry in plan.Entries)
        {
            var parts = entry.Repository.Split('/');
            RemoveEmptyDirectories(Path.Combine(downloads, parts[0], parts[1], entry.Revision), downloads);
            RemoveEmptyDirectories(Path.Combine(CacheDirectory, parts[0], parts[1]), CacheDirectory, descend: false);
        }

        return new ModelCacheCleanup(removed.AsReadOnly(), skipped.AsReadOnly());
    }

    /// <summary><see cref="PlanPrune"/> followed by <see cref="Prune(ModelCachePrunePlan)"/>.</summary>
    public ModelCacheCleanup Prune(IEnumerable<ModelAssetLock> keepLocks) => Prune(PlanPrune(keepLocks));

    /// <summary>
    /// Takes the lock of <c>{stem}.lock</c> without waiting; if it is free, removes <c>{stem}.partial</c> (when
    /// <paramref name="removePartial"/>) and the lock file itself, otherwise records them as skipped.
    /// </summary>
    private static void RemoveUnderLock(
        string stem, string repository, string revision, bool removePartial, List<ModelCacheEntry> removed, List<ModelCacheEntry> skipped)
    {
        string partial = stem + PartialSuffix;
        string lockPath = stem + LockSuffix;
        bool hasPartial = removePartial && File.Exists(partial);
        bool hasLock = File.Exists(lockPath);
        if (!hasPartial && !hasLock)
        {
            return;
        }

        var partialEntry = new ModelCacheEntry(partial, ModelCacheEntryKind.PartialDownload, hasPartial ? FileSize(partial) : 0, repository, revision);
        var lockEntry = new ModelCacheEntry(lockPath, ModelCacheEntryKind.LockFile, FileSize(lockPath), repository, revision);
        FileStream? held = TryLockNow(lockPath);
        if (held is null)
        {
            if (hasPartial)
            {
                skipped.Add(partialEntry);
            }

            skipped.Add(lockEntry);
            return;
        }

        if (hasPartial)
        {
            (TryDeleteFile(partial) ? removed : skipped).Add(partialEntry);
        }

        bool lockRemoved = ReleaseAndDeleteLock(held, lockPath);
        if (hasLock)
        {
            (lockRemoved ? removed : skipped).Add(lockEntry);
        }
    }

    private static void RemoveSnapshot(ModelCacheEntry entry, string downloads, List<ModelCacheEntry> removed, List<ModelCacheEntry> skipped)
    {
        var parts = entry.Repository.Split('/');
        string work = Path.Combine(downloads, parts[0], parts[1], entry.Revision);
        var held = new List<(FileStream Stream, string Path)>();
        try
        {
            if (Directory.Exists(work))
            {
                foreach (string lockPath in Directory.EnumerateFiles(work, "*" + LockSuffix, SearchOption.AllDirectories))
                {
                    var stream = TryLockNow(lockPath);
                    if (stream is null)
                    {
                        skipped.Add(entry);
                        return;
                    }

                    held.Add((stream, lockPath));
                }
            }

            try
            {
                if (Directory.Exists(entry.Path))
                {
                    Directory.Delete(entry.Path, recursive: true);
                }

                removed.Add(entry);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                skipped.Add(entry);
            }
        }
        finally
        {
            foreach (var (stream, _) in held)
            {
                stream.Dispose();
            }
        }
    }

    /// <summary>The lock file opened exclusively, or null when another holder has it (or it cannot be opened).</summary>
    private static FileStream? TryLockNow(string path)
    {
        try
        {
            var stream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1);
            if (File.Exists(path))
            {
                return stream;
            }

            stream.Dispose();
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>
    /// Deletes a lock file this call holds. On Unix the file is unlinked while held, so no other downloader can take it
    /// between release and delete (a downloader re-checks that its lock file still exists after the lock is granted). On
    /// Windows a held file cannot be deleted; it is released first, and the delete fails harmlessly if another
    /// downloader took it in between.
    /// </summary>
    private static bool ReleaseAndDeleteLock(FileStream held, string path)
    {
        if (OperatingSystem.IsWindows())
        {
            held.Dispose();
            return TryDeleteFile(path);
        }

        bool deleted = TryDeleteFile(path);
        held.Dispose();
        return deleted;
    }

    private static bool TryDeleteFile(string path)
    {
        try
        {
            File.Delete(path);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// <c>&lt;root&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;40-hex SHA&gt;</c> directories, with their repository and revision,
    /// in ordinal order. Names that cannot be a repository part (and, at the root of the cache, <c>.downloads</c>) are skipped.
    /// </summary>
    private static IEnumerable<(string Directory, string Repository, string Revision)> LayoutDirectories(string root, bool skipDownloads)
    {
        if (!Directory.Exists(root))
        {
            yield break;
        }

        foreach (string owner in Children(root))
        {
            string ownerName = Path.GetFileName(owner);
            if ((skipDownloads && ownerName == DownloadsDirectoryName) || !IsRepositoryPart(ownerName))
            {
                continue;
            }

            foreach (string name in Children(owner))
            {
                if (!IsRepositoryPart(Path.GetFileName(name)))
                {
                    continue;
                }

                foreach (string revision in Children(name))
                {
                    string sha = Path.GetFileName(revision);
                    if (LockValidation.IsLowerHex(sha, 40))
                    {
                        yield return (revision, ownerName + "/" + Path.GetFileName(name), sha);
                    }
                }
            }
        }

        static IEnumerable<string> Children(string directory) =>
            new DirectoryInfo(directory).EnumerateDirectories()
                .Where(d => (d.Attributes & FileAttributes.ReparsePoint) == 0)
                .Select(d => d.FullName)
                .Order(StringComparer.Ordinal);
    }

    private static bool IsRepositoryPart(string part)
    {
        try
        {
            LockValidation.ValidateRepository(part + "/x");
            return true;
        }
        catch (ArgumentException)
        {
            return false;
        }
    }

    private static long DirectorySize(string directory) =>
        new DirectoryInfo(directory).EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);

    private static long FileSize(string path)
    {
        var info = new FileInfo(path);
        return info.Exists ? info.Length : 0;
    }

    /// <summary>
    /// Removes <paramref name="directory"/> (with <paramref name="descend"/>, first its empty subdirectories) when empty,
    /// and then its empty parents, stopping at (and keeping) <paramref name="stopAt"/>.
    /// </summary>
    private static void RemoveEmptyDirectories(string directory, string stopAt, bool descend = true)
    {
        if (descend && Directory.Exists(directory))
        {
            foreach (string child in Directory.EnumerateDirectories(directory, "*", SearchOption.AllDirectories).OrderByDescending(d => d.Length))
            {
                TryRemoveEmpty(child);
            }
        }

        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(stopAt));
        for (string? current = Path.GetFullPath(directory);
            current is not null && current.Length > root.Length && current.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal);
            current = Path.GetDirectoryName(current))
        {
            if (!TryRemoveEmpty(current))
            {
                break;
            }
        }

        static bool TryRemoveEmpty(string path)
        {
            try
            {
                if (Directory.Exists(path) && !Directory.EnumerateFileSystemEntries(path).Any())
                {
                    Directory.Delete(path);
                }

                return !Directory.Exists(path);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }
    }
}
