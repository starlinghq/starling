using System.Buffers;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;

namespace Starlings.Acquisition;

/// <summary>
/// Local cache of locked model assets. <see cref="OpenLocal(ModelAssetLock)"/> resolves and verifies files without touching the network;
/// <see cref="DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/> is the only member that downloads. Nothing here is called implicitly by model packages.
/// </summary>
/// <remarks>
/// <para>Layout: <c>&lt;cache&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;commit SHA&gt;/&lt;path&gt;</c>, the same layout as
/// <c>tools/fetch-tokenizer-assets.sh</c>. Partial downloads and lock files live under <c>&lt;cache&gt;/.downloads/</c> with
/// the same relative layout, so a snapshot directory only ever contains complete, verified files.</para>
/// <para>A download streams into a partial file while hashing, checks size and SHA-256, flushes, and then renames the
/// partial file into place (atomic on one file system). An interrupted download keeps its partial file; the next attempt
/// re-hashes it and resumes with an HTTP range request. Concurrent acquisitions of the same file, in one process or
/// several, serialize on an exclusive lock file; the second one finds the verified file and does not download it again.</para>
/// <para>A store is thread-safe and holds no per-lock state, and each member that reads or downloads can take its
/// verification for the call (overloads with a verification, or ModelDownloadOptions), so one store can be kept for the life of the process.</para>
/// </remarks>
public sealed partial class ModelAssetStore : IDisposable
{
    private const int BufferSize = 1 << 20;
    private const long ProgressInterval = 8L << 20;
    private const string DownloadsDirectoryName = ".downloads";
    private const int MaxRedirects = 10;
    private static readonly TimeSpan LockPollInterval = TimeSpan.FromMilliseconds(100);
    private static readonly ProductInfoHeaderValue UserAgent = new("Starlings.Acquisition", typeof(ModelAssetStore).Assembly.GetName().Version?.ToString(3) ?? "0");

    private readonly ModelAcquisitionOptions _options;
    private readonly HttpClient? _injectedClient;
    private readonly object _clientGate = new();
    private HttpClient? _ownedClient;
    private volatile bool _disposed;

    /// <summary>Creates a store. No network or file access happens here.</summary>
    /// <param name="options">Cache location, verification and download settings; defaults when null.</param>
    /// <param name="httpClient">
    /// Client used by <see cref="DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/> (proxies, tests). Not disposed by the store. When null, the store creates
    /// its own on first download, following redirects and without a whole-request timeout.
    /// </param>
    public ModelAssetStore(ModelAcquisitionOptions? options = null, HttpClient? httpClient = null)
    {
        _options = options ?? new ModelAcquisitionOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxAttempts, 1, nameof(options));
        ArgumentNullException.ThrowIfNull(_options.Endpoint, nameof(options));
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxConcurrentFiles, 1, nameof(options));
        ValidateLockWaitTimeout(_options.LockWaitTimeout, nameof(options));
        if (_options.FallbackEndpoints is null || _options.FallbackEndpoints.Any(e => e is null || !e.IsAbsoluteUri))
        {
            throw new ArgumentException("FallbackEndpoints must be a list of absolute addresses.", nameof(options));
        }

        if (_options.LocalSources is null || _options.LocalSources.Any(string.IsNullOrEmpty))
        {
            throw new ArgumentException("LocalSources must be a list of directory paths.", nameof(options));
        }

        _injectedClient = httpClient;
        CacheDirectory = _options.ResolveCacheDirectory();
        _localSources = [.. _options.LocalSources.Select(Path.GetFullPath)];
    }

    private readonly string[] _localSources;

    /// <summary>Absolute cache root.</summary>
    public string CacheDirectory { get; }

    /// <summary>Directory that holds (or will hold) the files of <paramref name="assetLock"/>.</summary>
    public string GetSnapshotDirectory(ModelAssetLock assetLock)
    {
        ArgumentNullException.ThrowIfNull(assetLock);
        var parts = assetLock.Repository.Split('/');
        return Path.Combine(CacheDirectory, parts[0], parts[1], assetLock.Revision);
    }

    /// <summary>
    /// Offline resolution: returns the cached files of <paramref name="assetLock"/> after verifying them
    /// (<see cref="ModelAcquisitionOptions.Verification"/>). Never downloads. Blocking file I/O.
    /// </summary>
    /// <exception cref="ModelAssetMissingException">One or more files are not in the cache.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file has the wrong size or digest.</exception>
    public ModelSnapshot OpenLocal(ModelAssetLock assetLock) => OpenLocal(assetLock, _options.Verification);

    /// <summary>
    /// Offline resolution with a verification chosen for this call instead of <see cref="ModelAcquisitionOptions.Verification"/>.
    /// Never downloads. Blocking file I/O.
    /// </summary>
    /// <exception cref="ModelAssetMissingException">One or more files are not in the cache.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file has the wrong size or digest.</exception>
    public ModelSnapshot OpenLocal(ModelAssetLock assetLock, CacheVerification verification)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(assetLock);
        return Verify(assetLock, GetSnapshotDirectory(assetLock), verification, MissingHint(assetLock, CacheDirectory));
    }

    /// <summary>
    /// Offline resolution that does not throw for missing or failing files: true and the verified snapshot when every
    /// file is in the cache and passes <see cref="ModelAcquisitionOptions.Verification"/>; otherwise false and null.
    /// <see cref="Check(ModelAssetLock)"/> says what is wrong. Never downloads and changes nothing. Blocking file I/O.
    /// </summary>
    public bool TryOpenLocal(ModelAssetLock assetLock, [NotNullWhen(true)] out ModelSnapshot? snapshot) =>
        TryOpenLocal(assetLock, _options.Verification, out snapshot);

    /// <summary>As <see cref="TryOpenLocal(ModelAssetLock, out ModelSnapshot?)"/>, with a verification chosen for this call.</summary>
    public bool TryOpenLocal(ModelAssetLock assetLock, CacheVerification verification, [NotNullWhen(true)] out ModelSnapshot? snapshot)
    {
        snapshot = Check(assetLock, verification).Snapshot;
        return snapshot is not null;
    }

    /// <summary>
    /// Checks the cached files of <paramref name="assetLock"/> with <see cref="ModelAcquisitionOptions.Verification"/> and
    /// reports what is present, missing or failing, without throwing for any of those (an unreadable file is a
    /// <see cref="ModelAssetProblemKind.Unreadable"/> problem). Never downloads and changes nothing. Blocking file I/O:
    /// with <see cref="CacheVerification.Sha256"/> every present file is hashed.
    /// </summary>
    public ModelAssetStatus Check(ModelAssetLock assetLock) => Check(assetLock, _options.Verification);

    /// <summary>As <see cref="Check(ModelAssetLock)"/>, with a verification chosen for this call.</summary>
    public ModelAssetStatus Check(ModelAssetLock assetLock, CacheVerification verification)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(assetLock);
        return Inspect(assetLock, GetSnapshotDirectory(assetLock), verification, reportUnreadable: true);
    }

    /// <summary>
    /// <see cref="Check(ModelAssetLock)"/> for an existing directory laid out like the repository (the non-throwing form of
    /// <see cref="VerifyDirectory"/>). Never downloads and changes nothing. Blocking file I/O.
    /// </summary>
    public static ModelAssetStatus CheckDirectory(ModelAssetLock assetLock, string directory, CacheVerification verification = CacheVerification.Sha256)
    {
        ArgumentNullException.ThrowIfNull(assetLock);
        ArgumentException.ThrowIfNullOrEmpty(directory);
        return Inspect(assetLock, Path.GetFullPath(directory), verification, reportUnreadable: true);
    }

    /// <summary>
    /// Verifies an existing directory laid out like the repository (for example a Hub snapshot or a copied bundle) against
    /// <paramref name="assetLock"/>. Never downloads. Blocking file I/O.
    /// </summary>
    /// <exception cref="ModelAssetMissingException">One or more files are missing.</exception>
    /// <exception cref="ModelAssetIntegrityException">A file has the wrong size or digest.</exception>
    public static ModelSnapshot VerifyDirectory(ModelAssetLock assetLock, string directory, CacheVerification verification = CacheVerification.Sha256)
    {
        ArgumentNullException.ThrowIfNull(assetLock);
        ArgumentException.ThrowIfNullOrEmpty(directory);
        return Verify(assetLock, Path.GetFullPath(directory), verification,
            "Copy the files there, or download them explicitly with ModelAssetStore.DownloadAsync and use the cache instead.");
    }

    /// <summary>
    /// Explicit acquisition: ensures every file of <paramref name="assetLock"/> is in the cache and verified, downloading
    /// missing or invalid ones from <c>{Endpoint}/{repository}/resolve/{commit SHA}/{path}</c>, or from the file's explicit
    /// <see cref="ModelAssetFile.Source"/> when it has one. The Hub token goes only over https, to <see cref="ModelAcquisitionOptions.Endpoint"/>
    /// and to sources on <see cref="ModelAcquisitionOptions.TokenHosts"/> (see <see cref="ModelAcquisitionOptions.HuggingFaceToken"/>).
    /// Cached files are checked with
    /// <see cref="ModelAcquisitionOptions.Verification"/>; a cached file that fails the check is downloaded again and replaced.
    /// </summary>
    /// <param name="assetLock">The lock to acquire.</param>
    /// <param name="progress">Optional per-file progress.</param>
    /// <param name="cancellationToken">Cancels waiting, hashing and transfer. A partial file is kept for resumption.</param>
    /// <exception cref="ModelAssetAccessDeniedException">HTTP 401/403: gated, private or nonexistent repository, or bad token.</exception>
    /// <exception cref="ModelAssetNotFoundException">HTTP 404: the revision or file does not exist.</exception>
    /// <exception cref="ModelAssetIntegrityException">Downloaded bytes do not match the lock.</exception>
    /// <exception cref="ModelAssetDownloadException">Network or server failure after all attempts.</exception>
    /// <exception cref="ModelAcquisitionException">Writing to the cache failed (disk full, permissions); not retried, the message names the path.</exception>
    public Task<ModelSnapshot> DownloadAsync(ModelAssetLock assetLock, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default) =>
        DownloadAsync(assetLock, new ModelDownloadOptions { Progress = progress }, cancellationToken);

    /// <summary>
    /// Explicit acquisition with per-call settings (<paramref name="options"/>; null members use the store's
    /// <see cref="ModelAcquisitionOptions"/>). Otherwise the same as
    /// <see cref="DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/>, plus: a missing file
    /// is first copied from <see cref="ModelAcquisitionOptions.LocalSources"/>, then downloaded from
    /// <see cref="ModelAcquisitionOptions.Endpoint"/> and, on failure, from each of
    /// <see cref="ModelAcquisitionOptions.FallbackEndpoints"/>; up to <see cref="ModelDownloadOptions.MaxConcurrentFiles"/>
    /// files are acquired at once; a wait for another downloader's lock is bounded by
    /// <see cref="ModelDownloadOptions.LockWaitTimeout"/>.
    /// </summary>
    /// <exception cref="ModelAssetMissingException"><see cref="ModelAcquisitionOptions.AllowNetwork"/> is false and a file is neither cached nor in a local source.</exception>
    /// <exception cref="ModelAssetLockTimeoutException">Another downloader held a file's lock for longer than the wait timeout.</exception>
    /// <exception cref="ModelAssetAccessDeniedException">HTTP 401/403: gated, private or nonexistent repository, or bad token.</exception>
    /// <exception cref="ModelAssetNotFoundException">HTTP 404: the revision or file does not exist.</exception>
    /// <exception cref="ModelAssetIntegrityException">Downloaded bytes do not match the lock.</exception>
    /// <exception cref="ModelAssetDownloadException">Network or server failure after all attempts.</exception>
    /// <exception cref="ModelAcquisitionException">Writing to the cache failed (disk full, permissions); not retried, the message names the path.</exception>
    public async Task<ModelSnapshot> DownloadAsync(ModelAssetLock assetLock, ModelDownloadOptions options, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(assetLock);
        ArgumentNullException.ThrowIfNull(options);
        int concurrency = options.MaxConcurrentFiles ?? _options.MaxConcurrentFiles;
        ArgumentOutOfRangeException.ThrowIfLessThan(concurrency, 1, nameof(options));
        var call = new CallSettings(
            options.Verification ?? _options.Verification,
            options.LockWaitTimeout ?? _options.LockWaitTimeout);
        ValidateLockWaitTimeout(call.LockWaitTimeout, nameof(options));

        string snapshot = GetSnapshotDirectory(assetLock);
        string work = WorkDirectory(snapshot);
        if (!_options.AllowNetwork)
        {
            ThrowIfNotAvailableOffline(assetLock, snapshot);
        }

        var lockProgress = options.Progress is null ? null : new LockProgress(options.Progress, assetLock);
        int anyCached = 0;
        if (concurrency == 1)
        {
            for (int i = 0; i < assetLock.Files.Count; i++)
            {
                if (await AcquireFileAsync(assetLock, assetLock.Files[i], snapshot, work, call, lockProgress?.ForFile(i), cancellationToken).ConfigureAwait(false))
                {
                    anyCached = 1;
                }
            }
        }
        else
        {
            var parallel = new ParallelOptions { MaxDegreeOfParallelism = concurrency, CancellationToken = cancellationToken };
            await Parallel.ForEachAsync(Enumerable.Range(0, assetLock.Files.Count), parallel, async (i, token) =>
            {
                if (await AcquireFileAsync(assetLock, assetLock.Files[i], snapshot, work, call, lockProgress?.ForFile(i), token).ConfigureAwait(false))
                {
                    Volatile.Write(ref anyCached, 1);
                }
            }).ConfigureAwait(false);
        }

        return new ModelSnapshot(assetLock, snapshot, anyCached != 0 ? call.Verification : CacheVerification.Sha256);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        HttpClient? owned;
        lock (_clientGate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            owned = _ownedClient;
            _ownedClient = null;
        }

        owned?.Dispose();
    }

    internal static string LocalPath(string directory, string relativePath)
    {
        string full = Path.GetFullPath(Path.Combine(directory, relativePath.Replace('/', Path.DirectorySeparatorChar)));
        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory)) + Path.DirectorySeparatorChar;
        if (!full.StartsWith(root, StringComparison.Ordinal))
        {
            throw new ArgumentException($"Path escapes the snapshot directory: {relativePath}.", nameof(relativePath));
        }

        return full;
    }

    internal Uri BuildUri(ModelAssetLock assetLock, ModelAssetFile file) => BuildUri(assetLock, file, _options.Endpoint);

    /// <summary>The file's explicit source, else its Hub-layout address on <paramref name="endpoint"/> in the hosting repository.</summary>
    internal static Uri BuildUri(ModelAssetLock assetLock, ModelAssetFile file, Uri endpoint)
    {
        if (file.Source is not null)
        {
            return file.Source;
        }

        string repo = string.Join('/', assetLock.HostingRepository.Split('/').Select(Uri.EscapeDataString));
        string path = string.Join('/', file.Path.Split('/').Select(Uri.EscapeDataString));
        string baseText = endpoint.ToString();
        if (!baseText.EndsWith('/'))
        {
            baseText += "/";
        }

        return new Uri(new Uri(baseText), $"{repo}/resolve/{assetLock.HostingRevision}/{path}");
    }

    private static string MissingHint(ModelAssetLock assetLock, string cacheDirectory) =>
        "Nothing is downloaded implicitly. Download explicitly with " +
        "`await new ModelAssetStore(options).DownloadAsync(lock)` (from a repository checkout: " +
        $"`dotnet run --project ai-dotnet/tools/Starlings.Acquisition.Tool -- download {assetLock.Name}`), " +
        $"or point ModelAcquisitionOptions.CacheDirectory or ${ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable} at a cache that has them " +
        $"(searched: {cacheDirectory}).";

    private static ModelSnapshot Verify(ModelAssetLock assetLock, string directory, CacheVerification verification, string hint)
    {
        // The whole pass completes before anything is thrown: missing files are reported together, ahead of integrity
        // failures, so a partially populated directory with one corrupt file still says which files are absent.
        var status = Inspect(assetLock, directory, verification, reportUnreadable: false);
        var missing = status.MissingPaths;
        var firstFailure = status.FirstFailure;
        int failures = status.Problems.Count;
        if (missing.Count > 0)
        {
            string alsoFailed = firstFailure is null
                ? string.Empty
                : $" {failures} other file(s) also failed verification, first {firstFailure.AssetPath}: {firstFailure.Message}";
            string message =
                $"{assetLock.Name}: {missing.Count} of {assetLock.Files.Count} files of {assetLock.Repository}@{assetLock.Revision} " +
                $"are not in {directory}: {string.Join(", ", missing)}.{alsoFailed} {hint}";
            throw firstFailure is null
                ? new ModelAssetMissingException(message)
                {
                    Repository = assetLock.Repository,
                    Revision = assetLock.Revision,
                    AssetPath = missing[0],
                    MissingPaths = missing,
                    SnapshotDirectory = directory,
                }
                : new ModelAssetMissingException(message, firstFailure)
                {
                    Repository = assetLock.Repository,
                    Revision = assetLock.Revision,
                    AssetPath = missing[0],
                    MissingPaths = missing,
                    SnapshotDirectory = directory,
                };
        }

        if (firstFailure is not null)
        {
            throw firstFailure;
        }

        return new ModelSnapshot(assetLock, directory, verification);
    }

    /// <summary>
    /// One pass over the lock's files in <paramref name="directory"/>. With <paramref name="reportUnreadable"/> an I/O
    /// failure reading a present file becomes an <see cref="ModelAssetProblemKind.Unreadable"/> problem; without it the
    /// exception propagates, as <see cref="OpenLocal(ModelAssetLock)"/> always did.
    /// </summary>
    private static ModelAssetStatus Inspect(ModelAssetLock assetLock, string directory, CacheVerification verification, bool reportUnreadable)
    {
        var missing = new List<string>();
        var problems = new List<ModelAssetProblem>();
        ModelAssetIntegrityException? firstFailure = null;
        foreach (var file in assetLock.Files)
        {
            string path = LocalPath(directory, file.Path);
            if (!File.Exists(path))
            {
                missing.Add(file.Path);
                continue;
            }

            ModelAssetIntegrityException? failure;
            try
            {
                failure = CheckFile(assetLock, file, path, verification, null, CancellationToken.None);
            }
            catch (Exception ex) when (reportUnreadable && ex is IOException or UnauthorizedAccessException)
            {
                problems.Add(new ModelAssetProblem(
                    file.Path, ModelAssetProblemKind.Unreadable, path, file.Size, -1, file.Sha256, null,
                    $"{assetLock.Name}: {file.Path} at {path} could not be read: {ex.Message}"));
                continue;
            }

            if (failure is not null)
            {
                firstFailure ??= failure;
                problems.Add(new ModelAssetProblem(
                    file.Path,
                    failure.ActualSha256 is null ? ModelAssetProblemKind.SizeMismatch : ModelAssetProblemKind.DigestMismatch,
                    path, file.Size, failure.ActualSize, file.Sha256, failure.ActualSha256, failure.Message));
            }
        }

        return new ModelAssetStatus(assetLock, directory, verification, missing.AsReadOnly(), problems.AsReadOnly(), firstFailure);
    }

    /// <summary>Returns null when the file matches, otherwise the integrity failure (not thrown).</summary>
    private static ModelAssetIntegrityException? CheckFile(
        ModelAssetLock assetLock, ModelAssetFile file, string path, CacheVerification verification, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {
        long size = new FileInfo(path).Length;
        if (size != file.Size)
        {
            return IntegrityFailure(assetLock, file, path, size, null);
        }

        if (verification == CacheVerification.Size)
        {
            return null;
        }

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        HashFile(path, hash, file, progress, cancellationToken);
        string actual = Convert.ToHexStringLower(hash.GetHashAndReset());
        return actual == file.Sha256 ? null : IntegrityFailure(assetLock, file, path, size, actual);
    }

    /// <summary>
    /// Builds the integrity failure. <paramref name="actualSha"/> is null when the size already differed and the bytes were
    /// not hashed; <paramref name="message"/> replaces the default wording.
    /// </summary>
    private static ModelAssetIntegrityException IntegrityFailure(
        ModelAssetLock assetLock, ModelAssetFile file, string path, long actualSize, string? actualSha, string? message = null) =>
        new(message ?? (actualSha is null
            ? $"{assetLock.Name}: {file.Path} at {path} is {actualSize} bytes; the lock for {assetLock.Repository}@{assetLock.Revision} expects {file.Size} bytes (sha256 {file.Sha256})."
            : $"{assetLock.Name}: {file.Path} at {path} has sha256 {actualSha}; the lock for {assetLock.Repository}@{assetLock.Revision} expects {file.Sha256} ({file.Size} bytes)."))
        {
            Repository = assetLock.Repository,
            Revision = assetLock.Revision,
            AssetPath = file.Path,
            FilePath = path,
            ExpectedSize = file.Size,
            ActualSize = actualSize,
            ExpectedSha256 = file.Sha256,
            ActualSha256 = actualSha,
        };

    private static long HashFile(string path, IncrementalHash hash, ModelAssetFile file, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {
        byte[] buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete, 1, FileOptions.SequentialScan);
            long total = 0;
            long nextReport = ProgressInterval;
            int read;
            while ((read = stream.Read(buffer, 0, BufferSize)) > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
                hash.AppendData(buffer, 0, read);
                total += read;
                if (progress is not null && total >= nextReport)
                {
                    progress.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Verifying, total, file.Size));
                    nextReport = total + ProgressInterval;
                }
            }

            return total;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <summary>Returns true when the file was already cached and valid.</summary>
    private async Task<bool> AcquireFileAsync(
        ModelAssetLock assetLock, ModelAssetFile file, string snapshot, string work, CallSettings call, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {
        string target = LocalPath(snapshot, file.Path);
        string partial = LocalPath(work, file.Path) + ".partial";
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        Directory.CreateDirectory(Path.GetDirectoryName(partial)!);

        string lockPath = LocalPath(work, file.Path) + ".lock";
        var waited = Stopwatch.StartNew();
        using var exclusive = await LockFileAsync(
            lockPath,
            call.LockWaitTimeout,
            () => progress?.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Waiting, 0, file.Size)),
            cancellationToken).ConfigureAwait(false)
            ?? throw new ModelAssetLockTimeoutException(
                string.Create(CultureInfo.InvariantCulture,
                    $"{assetLock.Name}: another download of {file.Path} held its lock {lockPath} for {waited.Elapsed.TotalSeconds:0.###} s, ") +
                $"longer than LockWaitTimeout ({call.LockWaitTimeout}). Nothing was changed; call DownloadAsync again to wait again, " +
                "or remove a stale lock with ModelAssetStore.RemovePartialDownloads once no other download runs.")
            {
                Repository = assetLock.Repository,
                Revision = assetLock.Revision,
                AssetPath = file.Path,
                LockFilePath = lockPath,
                Waited = waited.Elapsed,
            };

        if (File.Exists(target) && CheckFile(assetLock, file, target, call.Verification, progress, cancellationToken) is null)
        {
            progress?.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Completed, file.Size, file.Size));
            return true;
        }

        if (!await TryCopyFromLocalSourceAsync(assetLock, file, target, partial, progress, cancellationToken).ConfigureAwait(false))
        {
            if (!_options.AllowNetwork)
            {
                throw OfflineMissing(assetLock, [file.Path], snapshot);
            }

            await DownloadFileAsync(assetLock, file, target, partial, progress, cancellationToken).ConfigureAwait(false);
        }

        progress?.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Completed, file.Size, file.Size));
        return false;
    }

    /// <summary>
    /// Takes the exclusive lock file, polling while another holder has it. Returns null when <paramref name="timeout"/>
    /// passes first. <paramref name="onWaiting"/> runs once, at the first refusal.
    /// </summary>
    private static async Task<FileStream?> LockFileAsync(string path, TimeSpan? timeout, Action? onWaiting, CancellationToken cancellationToken)
    {
        Stopwatch? waiting = null;
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            FileStream stream;
            try
            {
                // FileShare.None is an exclusive lock: a mandatory share mode on Windows, flock(LOCK_EX) on Unix, per open
                // file description, so it also excludes a second acquisition in the same process.
                stream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1);
            }
            catch (DirectoryNotFoundException)
            {
                // Prune or RemovePartialDownloads removed the empty work directory after this call created it.
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                continue;
            }
            catch (IOException) when (File.Exists(path))
            {
                if (waiting is null)
                {
                    waiting = Stopwatch.StartNew();
                    onWaiting?.Invoke();
                }

                var delay = LockPollInterval;
                if (timeout is { } limit)
                {
                    var left = limit - waiting.Elapsed;
                    if (left <= TimeSpan.Zero)
                    {
                        return null;
                    }

                    delay = left < delay ? left : delay;
                }

                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                continue;
            }

            if (!File.Exists(path))
            {
                // RemovePartialDownloads or Prune unlinked the file (Unix) after this open and before the lock was
                // granted: a lock on an unlinked file excludes nobody, so take the lock again by name.
                await stream.DisposeAsync().ConfigureAwait(false);
                continue;
            }

            return stream;
        }
    }

    private static void ValidateLockWaitTimeout(TimeSpan? timeout, string parameterName)
    {
        if (timeout is { } t && t < TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(parameterName, t, "LockWaitTimeout must not be negative.");
        }
    }

    /// <summary>Per-call settings resolved against the store's options.</summary>
    private readonly record struct CallSettings(CacheVerification Verification, TimeSpan? LockWaitTimeout);

    private string WorkDirectory(string snapshot) =>
        Path.Combine(CacheDirectory, DownloadsDirectoryName, Path.GetRelativePath(CacheDirectory, snapshot));

    /// <summary>The file in each local source (cache layout), in order.</summary>
    private IEnumerable<string> LocalSourceCandidates(ModelAssetLock assetLock, ModelAssetFile file)
    {
        var parts = assetLock.Repository.Split('/');
        foreach (string root in _localSources)
        {
            yield return LocalPath(Path.Combine(root, parts[0], parts[1], assetLock.Revision), file.Path);
        }
    }

    /// <summary>
    /// With <see cref="ModelAcquisitionOptions.AllowNetwork"/> false: throws before anything is copied when a file is
    /// neither in the cache nor, with the right size, in a local source.
    /// </summary>
    private void ThrowIfNotAvailableOffline(ModelAssetLock assetLock, string snapshot)
    {
        var missing = assetLock.Files
            .Where(f => !HasSize(LocalPath(snapshot, f.Path), f.Size) && !LocalSourceCandidates(assetLock, f).Any(c => HasSize(c, f.Size)))
            .Select(f => f.Path)
            .ToList();
        if (missing.Count > 0)
        {
            throw OfflineMissing(assetLock, missing, snapshot);
        }

        static bool HasSize(string path, long size)
        {
            var info = new FileInfo(path);
            return info.Exists && info.Length == size;
        }
    }

    private ModelAssetMissingException OfflineMissing(ModelAssetLock assetLock, List<string> missing, string snapshot) =>
        new($"{assetLock.Name}: {missing.Count} of {assetLock.Files.Count} files of {assetLock.Repository}@{assetLock.Revision} are neither in the cache " +
            $"({snapshot}) nor, with a matching size and digest, in a local source ({(_localSources.Length == 0 ? "none configured" : string.Join(", ", _localSources))}): " +
            $"{string.Join(", ", missing)}. ModelAcquisitionOptions.AllowNetwork is false, so nothing was downloaded. Copy the files into a local source " +
            "laid out like the cache, or allow the network.")
        {
            Repository = assetLock.Repository,
            Revision = assetLock.Revision,
            AssetPath = missing[0],
            MissingPaths = missing,
            SnapshotDirectory = snapshot,
        };

    /// <summary>
    /// Copies the file from the first local source whose copy matches the lock, hashing while copying into the partial
    /// file and renaming it into place. Returns false when no source has a matching copy. Never changes a source.
    /// </summary>
    private async Task<bool> TryCopyFromLocalSourceAsync(
        ModelAssetLock assetLock, ModelAssetFile file, string target, string partial, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {
        foreach (string candidate in LocalSourceCandidates(assetLock, file))
        {
            var info = new FileInfo(candidate);
            if (!info.Exists || info.Length != file.Size)
            {
                continue;
            }

            string? sha;
            try
            {
                sha = await CopyHashingAsync(assetLock, candidate, partial, file, progress, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Reading the source failed (writes to the cache raise ModelAcquisitionException): skipped like a mismatching copy.
                TryDelete(partial);
                continue;
            }

            if (sha == file.Sha256)
            {
                File.Move(partial, target, overwrite: true);
                return true;
            }

            TryDelete(partial);
        }

        return false;
    }

    /// <summary>Copies <paramref name="source"/> into <paramref name="partial"/> and returns its SHA-256, or null when it is longer than the lock.</summary>
    private static async Task<string?> CopyHashingAsync(
        ModelAssetLock assetLock, string source, string partial, ModelAssetFile file, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
        try
        {
            await using var input = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.Asynchronous | FileOptions.SequentialScan);
            FileStream output;
            try
            {
                output = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, 1, FileOptions.Asynchronous);
            }
            catch (Exception ex) when (IsLocalWriteFailure(ex))
            {
                throw CacheWriteFailure(assetLock, file, partial, ex);
            }

            await using (output.ConfigureAwait(false))
            {
                long total = 0;
                long nextReport = 0;
                int read;
                while ((read = await input.ReadAsync(buffer.AsMemory(0, BufferSize), cancellationToken).ConfigureAwait(false)) > 0)
                {
                    total += read;
                    if (total > file.Size)
                    {
                        return null;
                    }

                    hash.AppendData(buffer, 0, read);
                    try
                    {
                        await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                    }
                    catch (Exception ex) when (IsLocalWriteFailure(ex))
                    {
                        throw CacheWriteFailure(assetLock, file, partial, ex);
                    }

                    if (progress is not null && total >= nextReport)
                    {
                        progress.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Downloading, total, file.Size));
                        nextReport = total + ProgressInterval;
                    }
                }

                try
                {
                    await output.FlushAsync(cancellationToken).ConfigureAwait(false);
                    output.Flush(flushToDisk: true);
                }
                catch (Exception ex) when (IsLocalWriteFailure(ex))
                {
                    throw CacheWriteFailure(assetLock, file, partial, ex);
                }

                return total == file.Size ? Convert.ToHexStringLower(hash.GetHashAndReset()) : null;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private HttpClient Client
    {
        get
        {
            if (_injectedClient is not null)
            {
                return _injectedClient;
            }

            lock (_clientGate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                // SendFollowingRedirectsAsync follows redirects itself, so the token is attached (or not) per hop.
                return _ownedClient ??= new HttpClient(new SocketsHttpHandler
                {
                    AllowAutoRedirect = false,
                    AutomaticDecompression = DecompressionMethods.None,
                    PooledConnectionLifetime = TimeSpan.FromMinutes(5),
                })
                {
                    Timeout = Timeout.InfiniteTimeSpan,
                };
            }
        }
    }

    /// <summary>
    /// Downloads from the file's explicit source, or from <see cref="ModelAcquisitionOptions.Endpoint"/> and then each
    /// fallback endpoint in order until one delivers the locked bytes.
    /// </summary>
    private async Task DownloadFileAsync(
        ModelAssetLock assetLock, ModelAssetFile file, string target, string partial, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {
        if (file.Source is not null || _options.FallbackEndpoints.Count == 0)
        {
            await DownloadFromAsync(assetLock, file, BuildUri(assetLock, file), target, partial, progress, cancellationToken).ConfigureAwait(false);
            return;
        }

        var failures = new List<ModelAcquisitionException>();
        foreach (var endpoint in (IEnumerable<Uri>)[_options.Endpoint, .. _options.FallbackEndpoints])
        {
            try
            {
                await DownloadFromAsync(assetLock, file, BuildUri(assetLock, file, endpoint), target, partial, progress, cancellationToken).ConfigureAwait(false);
                return;
            }
            catch (ModelAcquisitionException ex) when (ex is ModelAssetDownloadException or ModelAssetNotFoundException or ModelAssetAccessDeniedException or ModelAssetIntegrityException)
            {
                // A local write failure (the base type) is not the endpoint's fault and propagates at once.
                failures.Add(ex);
            }
        }

        var primary = failures[0];
        primary.EndpointFailures = failures.AsReadOnly();
        ExceptionDispatchInfo.Throw(primary);
    }

    private async Task DownloadFromAsync(
        ModelAssetLock assetLock, ModelAssetFile file, Uri uri, string target, string partial, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {

        // Attached per request only where IsTokenHost allows it; never to storage, CDN or other hosts.
        string? token = _options.ResolveToken();
        Exception? lastError = null;
        HttpStatusCode? lastStatus = null;
        bool freshRetryUsed = false;

        for (int attempt = 1; attempt <= _options.MaxAttempts; attempt++)
        {
            if (attempt > 1)
            {
                var delay = _options.RetryDelay * Math.Pow(2, attempt - 2);
                if (delay > TimeSpan.Zero)
                {
                    await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                }
            }

            DownloadOutcome outcome;
            try
            {
                outcome = await DownloadOnceAsync(assetLock, file, uri, token, partial, progress, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (TransientDownloadException ex)
            {
                lastError = ex.InnerException ?? ex;
                lastStatus = ex.StatusCode;
                continue;
            }
            catch (Exception ex) when (ex is HttpRequestException or IOException or OperationCanceledException)
            {
                // Network failure, truncated body, or a response-header timeout. Cache write failures are not IOExceptions
                // here: DownloadOnceAsync raises them as ModelAcquisitionException, which propagates without a retry.
                lastError = ex;
                lastStatus = (ex as HttpRequestException)?.StatusCode;
                continue;
            }

            if (outcome.Sha256 == file.Sha256 && outcome.Size == file.Size)
            {
                File.Move(partial, target, overwrite: true);
                return;
            }

            TryDelete(partial);
            if (outcome.Resumed && !freshRetryUsed)
            {
                // The bytes kept from an earlier attempt may be what is wrong: start over once.
                freshRetryUsed = true;
                attempt--;
                continue;
            }

            throw outcome.Sha256 is null
                ? IntegrityFailure(assetLock, file, uri.ToString(), outcome.Size, null,
                    $"{assetLock.Name}: {file.Path} from {uri} is longer than the lock for {assetLock.Repository}@{assetLock.Revision} expects " +
                    $"({file.Size} bytes; the transfer was stopped after {outcome.Size} bytes and the bytes were not hashed).")
                : IntegrityFailure(assetLock, file, uri.ToString(), outcome.Size, outcome.Sha256);
        }

        throw new ModelAssetDownloadException(
            $"{assetLock.Name}: downloading {file.Path} from {uri} failed after {_options.MaxAttempts} attempts" +
            (lastStatus is { } s ? $" (last status {(int)s} {s})" : string.Empty) +
            $": {lastError?.Message}. The partial file is kept at {partial}; calling DownloadAsync again resumes it.",
            lastError!)
        {
            Repository = assetLock.Repository,
            Revision = assetLock.Revision,
            AssetPath = file.Path,
            StatusCode = lastStatus,
            Attempts = _options.MaxAttempts,
        };
    }

    /// <summary>Result of one attempt. <see cref="Sha256"/> is null when the response was longer than the lock and not hashed.</summary>
    private readonly record struct DownloadOutcome(long Size, string? Sha256, bool Resumed);

    private async Task<DownloadOutcome> DownloadOnceAsync(
        ModelAssetLock assetLock, ModelAssetFile file, Uri uri, string? token, string partial, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        long existing = 0;
        if (File.Exists(partial))
        {
            existing = new FileInfo(partial).Length;
            if (existing > file.Size)
            {
                TryDelete(partial);
                existing = 0;
            }
            else if (existing > 0)
            {
                progress?.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Verifying, 0, file.Size));
                existing = HashFile(partial, hash, file, progress, cancellationToken);
            }
        }

        if (existing == file.Size && existing > 0)
        {
            // Complete from an earlier attempt that was interrupted before the rename.
            return new DownloadOutcome(existing, Convert.ToHexStringLower(hash.GetHashAndReset()), Resumed: true);
        }

        HttpResponseMessage response;
        bool tokenSent;
        using (var headerTimeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
        {
            headerTimeout.CancelAfter(_options.ResponseTimeout);
            (response, tokenSent) = await SendFollowingRedirectsAsync(file, uri, token, existing, headerTimeout.Token).ConfigureAwait(false);
        }

        using (response)
        {
            ThrowForStatus(assetLock, file, uri, response, tokenSent, partial);

            bool resumed = false;
            if (existing > 0 && response.StatusCode == HttpStatusCode.PartialContent)
            {
                if (response.Content.Headers.ContentRange is not { From: { } from } || from != existing)
                {
                    TryDelete(partial);
                    throw new TransientDownloadException("The server answered the range request with an unexpected Content-Range.", null);
                }

                resumed = true;
            }
            else if (existing > 0)
            {
                // 200: the server ignored the range; start from zero.
                hash.GetHashAndReset();
                existing = 0;
            }

            FileStream output;
            try
            {
                output = new FileStream(partial, resumed ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1, FileOptions.Asynchronous);
            }
            catch (Exception ex) when (IsLocalWriteFailure(ex))
            {
                throw CacheWriteFailure(assetLock, file, partial, ex);
            }

            await using (output.ConfigureAwait(false))
            await using (var input = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
            {
                // Each read has its own idle budget (ResponseTimeout): a half-open connection never completes a read, and
                // the owned client has no whole-request timeout. The budget is disarmed while the bytes are written, so a
                // slow disk does not count against it.
                using var idle = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                byte[] buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
                try
                {
                    long total = existing;
                    long nextReport = total;
                    while (true)
                    {
                        int read = await ReadWithIdleTimeoutAsync(input, buffer, idle, total, cancellationToken).ConfigureAwait(false);
                        if (read <= 0)
                        {
                            break;
                        }

                        total += read;
                        if (total > file.Size)
                        {
                            break;
                        }

                        hash.AppendData(buffer, 0, read);
                        try
                        {
                            await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                        }
                        catch (Exception ex) when (IsLocalWriteFailure(ex))
                        {
                            throw CacheWriteFailure(assetLock, file, partial, ex);
                        }

                        if (progress is not null && total >= nextReport)
                        {
                            progress.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Downloading, total, file.Size));
                            nextReport = total + ProgressInterval;
                        }
                    }

                    try
                    {
                        await output.FlushAsync(cancellationToken).ConfigureAwait(false);
                        output.Flush(flushToDisk: true);
                    }
                    catch (Exception ex) when (IsLocalWriteFailure(ex))
                    {
                        throw CacheWriteFailure(assetLock, file, partial, ex);
                    }

                    if (total < file.Size)
                    {
                        throw new TransientDownloadException(
                            string.Create(CultureInfo.InvariantCulture, $"The response ended after {total} of {file.Size} bytes."), null);
                    }

                    progress?.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Downloading, Math.Min(total, file.Size), file.Size));
                    return total > file.Size
                        ? new DownloadOutcome(total, null, resumed)
                        : new DownloadOutcome(total, Convert.ToHexStringLower(hash.GetHashAndReset()), resumed);
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(buffer);
                }
            }
        }
    }

    /// <summary>
    /// One body read bounded by <see cref="ModelAcquisitionOptions.ResponseTimeout"/>. A read that receives nothing within
    /// the budget is a transient failure (the retry loop keeps the partial file and resumes); the caller's own
    /// cancellation still surfaces as <see cref="OperationCanceledException"/>.
    /// </summary>
    private async ValueTask<int> ReadWithIdleTimeoutAsync(Stream input, byte[] buffer, CancellationTokenSource idle, long received, CancellationToken cancellationToken)
    {
        idle.CancelAfter(_options.ResponseTimeout);
        try
        {
            int read = await input.ReadAsync(buffer.AsMemory(0, BufferSize), idle.Token).ConfigureAwait(false);
            idle.CancelAfter(Timeout.InfiniteTimeSpan);
            return read;
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            // Not wrapped: the retry loop reports a transient failure's inner exception when it has one, and the
            // cancellation's own message would read as if the caller had canceled.
            throw new TransientDownloadException(
                string.Create(CultureInfo.InvariantCulture, $"No data was received for {_options.ResponseTimeout.TotalSeconds:0.###} s after {received} bytes."),
                null);
        }
    }

    /// <summary>Failures of the partial file itself (disk full, permissions, a path in the way), as opposed to the transfer.</summary>
    private static bool IsLocalWriteFailure(Exception ex) => ex is IOException or UnauthorizedAccessException;

    private static ModelAcquisitionException CacheWriteFailure(ModelAssetLock assetLock, ModelAssetFile file, string partial, Exception cause) =>
        new($"{assetLock.Name}: writing {file.Path} to the cache at {partial} failed: {cause.Message} " +
            "This is a local failure (disk full, permissions, or a path in the way), not a download failure, and it was not retried. " +
            $"Fix the cache directory, or point ModelAcquisitionOptions.CacheDirectory or ${ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable} elsewhere.",
            cause)
        {
            Repository = assetLock.Repository,
            Revision = assetLock.Revision,
            AssetPath = file.Path,
        };

    /// <summary>
    /// Sends the GET and follows up to <see cref="MaxRedirects"/> redirects itself, with a fresh request per hop that carries
    /// the token only when that hop's address passes <see cref="IsTokenHost"/>. The Hub answers <c>resolve</c> with a
    /// redirect to its storage or CDN host (a signed address that must not receive the token) or, for some files, a relative
    /// redirect on the same host (which needs the token for a private repository). Refuses https-to-http downgrades.
    /// </summary>
    private async Task<(HttpResponseMessage Response, bool TokenSent)> SendFollowingRedirectsAsync(
        ModelAssetFile file, Uri uri, string? token, long rangeFrom, CancellationToken cancellationToken)
    {
        Uri current = uri;
        for (int hop = 0; ; hop++)
        {
            bool sendToken = token is not null && IsTokenHost(file, current);
            using var request = new HttpRequestMessage(HttpMethod.Get, current);
            request.Headers.UserAgent.Add(UserAgent);
            if (sendToken)
            {
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
            }

            if (rangeFrom > 0)
            {
                request.Headers.Range = new RangeHeaderValue(rangeFrom, null);
            }

            var response = await Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
            if (!IsRedirect(response.StatusCode) || response.Headers.Location is not { } location)
            {
                return (response, sendToken);
            }

            response.Dispose();
            if (hop >= MaxRedirects)
            {
                throw new HttpRequestException($"More than {MaxRedirects} redirects for {uri}.");
            }

            var next = location.IsAbsoluteUri ? location : new Uri(current, location);
            if (next.Scheme != Uri.UriSchemeHttps && (next.Scheme != Uri.UriSchemeHttp || current.Scheme == Uri.UriSchemeHttps))
            {
                // Not retried as transient: the same answer would come again.
                throw new ModelAssetDownloadException($"{uri} redirected to a {next.Scheme} address; only https (or http from http) redirects are followed.")
                {
                    AssetPath = file.Path,
                    StatusCode = response.StatusCode,
                    Attempts = 1,
                };
            }

            current = next;
        }
    }

    private static bool IsRedirect(HttpStatusCode status) =>
        status is HttpStatusCode.MovedPermanently or HttpStatusCode.Found or HttpStatusCode.SeeOther
            or HttpStatusCode.TemporaryRedirect or HttpStatusCode.PermanentRedirect;

    /// <summary>
    /// Whether a request to <paramref name="address"/> may carry the token: an https address on the default port whose host
    /// is exactly one of <see cref="ModelAcquisitionOptions.TokenHosts"/>, or, for a file without an explicit source, an
    /// https address with the host and port of <see cref="ModelAcquisitionOptions.Endpoint"/>. Never over http, so an
    /// <c>http://</c> endpoint (a local mirror) is used without the token rather than sending it in cleartext.
    /// </summary>
    internal bool IsTokenHost(ModelAssetFile file, Uri address) =>
        IsAllowedTokenHost(address)
        || (file.Source is null
            && address.Scheme == Uri.UriSchemeHttps
            && Uri.Compare(address, _options.Endpoint, UriComponents.HostAndPort, UriFormat.Unescaped, StringComparison.OrdinalIgnoreCase) == 0);

    private static bool IsAllowedTokenHost(Uri address) =>
        address.Scheme == Uri.UriSchemeHttps && address.IsDefaultPort
        && ModelAcquisitionOptions.TokenHosts.Contains(address.IdnHost, StringComparer.OrdinalIgnoreCase);

    private static void ThrowForStatus(ModelAssetLock assetLock, ModelAssetFile file, Uri uri, HttpResponseMessage response, bool tokenProvided, string partial)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        var status = response.StatusCode;
        string where = $"{assetLock.Name}: {uri} returned {(int)status} {status}";
        if (file.Source is { } hubSource && IsAllowedTokenHost(hubSource) && status is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
        {
            // Explicit source in a Hugging Face repository (for example a private hosting repository of built files).
            throw new ModelAssetAccessDeniedException(
                $"{where}. {file.Path} is hosted in a Hugging Face repository that is private or gated, or does not exist (the Hub answers 401 for " +
                "repositories the caller cannot see). " +
                (tokenProvided
                    ? "A token was sent: check that it is valid, has read access, and that its account has been granted access to that repository. "
                    : $"No token was sent: pass a read token of an account with access to that repository through ModelAcquisitionOptions.HuggingFaceToken or " +
                      $"${ModelAcquisitionOptions.TokenEnvironmentVariable}. ") +
                "Alternatively, place the file in the snapshot directory by other means (the lock verifies size and SHA-256).")
            {
                Repository = assetLock.Repository,
                Revision = assetLock.Revision,
                AssetPath = file.Path,
                StatusCode = status,
                TokenProvided = tokenProvided,
            };
        }

        if (file.Source is not null && status is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden or HttpStatusCode.NotFound)
        {
            // Explicit source (not the Hub): no token is involved, and the asset may not be published yet.
            string explicitMessage = $"{where}. {file.Path} has an explicit source outside the Hub repository; the asset may not be published " +
                "at that address yet, or access to it is restricted. Download it by other means and place it in the snapshot directory " +
                "(the lock verifies size and SHA-256), or open an existing directory with ModelAssetStore.VerifyDirectory.";
            if (status == HttpStatusCode.NotFound)
            {
                throw new ModelAssetNotFoundException(explicitMessage)
                {
                    Repository = assetLock.Repository,
                    Revision = assetLock.Revision,
                    AssetPath = file.Path,
                };
            }

            throw new ModelAssetAccessDeniedException(explicitMessage)
            {
                Repository = assetLock.Repository,
                Revision = assetLock.Revision,
                AssetPath = file.Path,
                StatusCode = status,
                TokenProvided = false,
            };
        }

        switch (status)
        {
            case HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden:
                throw new ModelAssetAccessDeniedException(
                    $"{where}. The repository {assetLock.HostingRepository} is gated or private, or does not exist (the Hub answers 401 for unknown repositories). " +
                    (tokenProvided
                        ? "A token was sent: check that it is valid, has read access, and that its account accepted the repository's license or access conditions on https://huggingface.co/" + assetLock.HostingRepository + "."
                        : $"No token was sent: accept the license or access conditions on https://huggingface.co/{assetLock.HostingRepository} and pass a read token through " +
                          $"ModelAcquisitionOptions.HuggingFaceToken or ${ModelAcquisitionOptions.TokenEnvironmentVariable}."))
                {
                    Repository = assetLock.Repository,
                    Revision = assetLock.Revision,
                    AssetPath = file.Path,
                    StatusCode = status,
                    TokenProvided = tokenProvided,
                };
            case HttpStatusCode.NotFound:
                throw new ModelAssetNotFoundException(
                    $"{where}. Revision {assetLock.HostingRevision} or file {file.Path} does not exist in {assetLock.HostingRepository}; the lock may be wrong or the revision was removed upstream.")
                {
                    Repository = assetLock.Repository,
                    Revision = assetLock.Revision,
                    AssetPath = file.Path,
                };
            case HttpStatusCode.RequestedRangeNotSatisfiable:
                TryDelete(partial);
                throw new TransientDownloadException($"{where} for the resume range; restarting from zero.", status);
            case HttpStatusCode.RequestTimeout or HttpStatusCode.TooManyRequests:
            case >= HttpStatusCode.InternalServerError:
                throw new TransientDownloadException(where, status);
            default:
                throw new ModelAssetDownloadException(where + ".")
                {
                    Repository = assetLock.Repository,
                    Revision = assetLock.Revision,
                    AssetPath = file.Path,
                    StatusCode = status,
                    Attempts = 1,
                };
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private sealed class TransientDownloadException : Exception
    {
        public TransientDownloadException(string message, HttpStatusCode? statusCode)
            : base(message)
        {
            StatusCode = statusCode;
        }

        public HttpStatusCode? StatusCode { get; }
    }

    /// <summary>
    /// Adds the lock totals to each file's progress and reports one value at a time, so a caller's
    /// <see cref="IProgress{T}"/> sees a consistent sum also when files are acquired concurrently.
    /// </summary>
    private sealed class LockProgress(IProgress<ModelAssetProgress> target, ModelAssetLock assetLock)
    {
        private readonly long[] _bytes = new long[assetLock.Files.Count];
        private readonly long _total = assetLock.TotalSize;
        private readonly Lock _gate = new();

        public IProgress<ModelAssetProgress> ForFile(int index) => new FileProgress(this, index);

        private void Report(int index, ModelAssetProgress value)
        {
            lock (_gate)
            {
                if (value.Stage != ModelAssetStage.Waiting)
                {
                    _bytes[index] = value.BytesProcessed;
                }

                long sum = 0;
                foreach (long b in _bytes)
                {
                    sum += b;
                }

                target.Report(value with { FileIndex = index, FileCount = _bytes.Length, LockBytesProcessed = sum, LockTotalBytes = _total });
            }
        }

        private sealed class FileProgress(LockProgress owner, int index) : IProgress<ModelAssetProgress>
        {
            public void Report(ModelAssetProgress value) => owner.Report(index, value);
        }
    }
}
