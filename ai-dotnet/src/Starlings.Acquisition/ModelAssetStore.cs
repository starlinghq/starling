using System.Buffers;
using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace Starlings.Acquisition;

/// <summary>
/// Local cache of locked model assets. <see cref="OpenLocal"/> resolves and verifies files without touching the network;
/// <see cref="DownloadAsync"/> is the only member that downloads. Nothing here is called implicitly by model packages.
/// </summary>
/// <remarks>
/// <para>Layout: <c>&lt;cache&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;commit SHA&gt;/&lt;path&gt;</c>, the same layout as
/// <c>tools/fetch-tokenizer-assets.sh</c>. Partial downloads and lock files live under <c>&lt;cache&gt;/.downloads/</c> with
/// the same relative layout, so a snapshot directory only ever contains complete, verified files.</para>
/// <para>A download streams into a partial file while hashing, checks size and SHA-256, flushes, and then renames the
/// partial file into place (atomic on one file system). An interrupted download keeps its partial file; the next attempt
/// re-hashes it and resumes with an HTTP range request. Concurrent acquisitions of the same file, in one process or
/// several, serialize on an exclusive lock file; the second one finds the verified file and does not download it again.</para>
/// </remarks>
public sealed class ModelAssetStore : IDisposable
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
    private bool _disposed;

    /// <summary>Creates a store. No network or file access happens here.</summary>
    /// <param name="options">Cache location, verification and download settings; defaults when null.</param>
    /// <param name="httpClient">
    /// Client used by <see cref="DownloadAsync"/> (proxies, tests). Not disposed by the store. When null, the store creates
    /// its own on first download, following redirects and without a whole-request timeout.
    /// </param>
    public ModelAssetStore(ModelAcquisitionOptions? options = null, HttpClient? httpClient = null)
    {
        _options = options ?? new ModelAcquisitionOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxAttempts, 1, nameof(options));
        ArgumentNullException.ThrowIfNull(_options.Endpoint, nameof(options));
        _injectedClient = httpClient;
        CacheDirectory = _options.ResolveCacheDirectory();
    }

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
    public ModelSnapshot OpenLocal(ModelAssetLock assetLock)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(assetLock);
        return Verify(assetLock, GetSnapshotDirectory(assetLock), _options.Verification, MissingHint(assetLock, CacheDirectory));
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
    /// <see cref="ModelAssetFile.Source"/> when it has one. The Hub token goes only to <see cref="ModelAcquisitionOptions.Endpoint"/>
    /// and to https sources on <see cref="ModelAcquisitionOptions.TokenHosts"/> (see <see cref="ModelAcquisitionOptions.HuggingFaceToken"/>).
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
    public async Task<ModelSnapshot> DownloadAsync(ModelAssetLock assetLock, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(assetLock);
        string snapshot = GetSnapshotDirectory(assetLock);
        string work = Path.Combine(CacheDirectory, DownloadsDirectoryName, Path.GetRelativePath(CacheDirectory, snapshot));
        bool anyCached = false;
        foreach (var file in assetLock.Files)
        {
            anyCached |= await AcquireFileAsync(assetLock, file, snapshot, work, progress, cancellationToken).ConfigureAwait(false);
        }

        return new ModelSnapshot(assetLock, snapshot, anyCached ? _options.Verification : CacheVerification.Sha256);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _ownedClient?.Dispose();
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

    internal Uri BuildUri(ModelAssetLock assetLock, ModelAssetFile file)
    {
        if (file.Source is not null)
        {
            return file.Source;
        }

        string repo = string.Join('/', assetLock.Repository.Split('/').Select(Uri.EscapeDataString));
        string path = string.Join('/', file.Path.Split('/').Select(Uri.EscapeDataString));
        string baseText = _options.Endpoint.ToString();
        if (!baseText.EndsWith('/'))
        {
            baseText += "/";
        }

        return new Uri(new Uri(baseText), $"{repo}/resolve/{assetLock.Revision}/{path}");
    }

    private static string MissingHint(ModelAssetLock assetLock, string cacheDirectory) =>
        "Nothing is downloaded implicitly. Download explicitly with " +
        "`await new ModelAssetStore(options).DownloadAsync(lock)` (from a repository checkout: " +
        $"`dotnet run --project ai-dotnet/tools/Starlings.Acquisition.Tool -- download {assetLock.Name}`), " +
        $"or point ModelAcquisitionOptions.CacheDirectory or ${ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable} at a cache that has them " +
        $"(searched: {cacheDirectory}).";

    private static ModelSnapshot Verify(ModelAssetLock assetLock, string directory, CacheVerification verification, string hint)
    {
        var missing = new List<string>();
        foreach (var file in assetLock.Files)
        {
            string path = LocalPath(directory, file.Path);
            if (!File.Exists(path))
            {
                missing.Add(file.Path);
                continue;
            }

            var failure = CheckFile(assetLock, file, path, verification, null, CancellationToken.None);
            if (failure is not null)
            {
                throw failure;
            }
        }

        if (missing.Count > 0)
        {
            throw new ModelAssetMissingException(
                $"{assetLock.Name}: {missing.Count} of {assetLock.Files.Count} files of {assetLock.Repository}@{assetLock.Revision} " +
                $"are not in {directory}: {string.Join(", ", missing)}. {hint}")
            {
                Repository = assetLock.Repository,
                Revision = assetLock.Revision,
                AssetPath = missing[0],
                MissingPaths = missing,
                SnapshotDirectory = directory,
            };
        }

        return new ModelSnapshot(assetLock, directory, verification);
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

    private static ModelAssetIntegrityException IntegrityFailure(ModelAssetLock assetLock, ModelAssetFile file, string path, long actualSize, string? actualSha) =>
        new(actualSha is null
            ? $"{assetLock.Name}: {file.Path} at {path} is {actualSize} bytes; the lock for {assetLock.Repository}@{assetLock.Revision} expects {file.Size} bytes (sha256 {file.Sha256})."
            : $"{assetLock.Name}: {file.Path} at {path} has sha256 {actualSha}; the lock for {assetLock.Repository}@{assetLock.Revision} expects {file.Sha256} ({file.Size} bytes).")
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
        ModelAssetLock assetLock, ModelAssetFile file, string snapshot, string work, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {
        string target = LocalPath(snapshot, file.Path);
        string partial = LocalPath(work, file.Path) + ".partial";
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        Directory.CreateDirectory(Path.GetDirectoryName(partial)!);

        using var exclusive = await LockFileAsync(LocalPath(work, file.Path) + ".lock", cancellationToken).ConfigureAwait(false);

        if (File.Exists(target) && CheckFile(assetLock, file, target, _options.Verification, progress, cancellationToken) is null)
        {
            progress?.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Completed, file.Size, file.Size));
            return true;
        }

        await DownloadFileAsync(assetLock, file, target, partial, progress, cancellationToken).ConfigureAwait(false);
        progress?.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Completed, file.Size, file.Size));
        return false;
    }

    private static async Task<FileStream> LockFileAsync(string path, CancellationToken cancellationToken)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                // FileShare.None is an exclusive lock: a mandatory share mode on Windows, flock(LOCK_EX) on Unix, per open
                // file description, so it also excludes a second acquisition in the same process.
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1);
            }
            catch (IOException) when (File.Exists(path))
            {
                await Task.Delay(LockPollInterval, cancellationToken).ConfigureAwait(false);
            }
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

    private async Task DownloadFileAsync(
        ModelAssetLock assetLock, ModelAssetFile file, string target, string partial, IProgress<ModelAssetProgress>? progress, CancellationToken cancellationToken)
    {
        Uri uri = BuildUri(assetLock, file);

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
                // Network failure, truncated body, or a response-header timeout.
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

            throw IntegrityFailure(assetLock, file, uri.ToString(), outcome.Size, outcome.Sha256);
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

    private readonly record struct DownloadOutcome(long Size, string Sha256, bool Resumed);

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

            await using var output = new FileStream(partial, resumed ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1, FileOptions.Asynchronous);
            await using var input = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            byte[] buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
            try
            {
                long total = existing;
                long nextReport = total;
                int read;
                while ((read = await input.ReadAsync(buffer.AsMemory(0, BufferSize), cancellationToken).ConfigureAwait(false)) > 0)
                {
                    total += read;
                    if (total > file.Size)
                    {
                        break;
                    }

                    hash.AppendData(buffer, 0, read);
                    await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                    if (progress is not null && total >= nextReport)
                    {
                        progress.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Downloading, total, file.Size));
                        nextReport = total + ProgressInterval;
                    }
                }

                await output.FlushAsync(cancellationToken).ConfigureAwait(false);
                output.Flush(flushToDisk: true);
                if (total < file.Size)
                {
                    throw new TransientDownloadException(
                        string.Create(CultureInfo.InvariantCulture, $"The response ended after {total} of {file.Size} bytes."), null);
                }

                progress?.Report(new ModelAssetProgress(file.Path, ModelAssetStage.Downloading, Math.Min(total, file.Size), file.Size));
                return total > file.Size
                    ? new DownloadOutcome(total, "(not hashed: longer than the lock)", resumed)
                    : new DownloadOutcome(total, Convert.ToHexStringLower(hash.GetHashAndReset()), resumed);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }
    }

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
    /// address with the scheme, host and port of <see cref="ModelAcquisitionOptions.Endpoint"/>.
    /// </summary>
    internal bool IsTokenHost(ModelAssetFile file, Uri address) =>
        IsAllowedTokenHost(address)
        || (file.Source is null
            && Uri.Compare(address, _options.Endpoint, UriComponents.SchemeAndServer, UriFormat.Unescaped, StringComparison.OrdinalIgnoreCase) == 0);

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
                    $"{where}. The repository {assetLock.Repository} is gated or private, or does not exist (the Hub answers 401 for unknown repositories). " +
                    (tokenProvided
                        ? "A token was sent: check that it is valid, has read access, and that its account accepted the repository's license or access conditions on https://huggingface.co/" + assetLock.Repository + "."
                        : $"No token was sent: accept the license or access conditions on https://huggingface.co/{assetLock.Repository} and pass a read token through " +
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
                    $"{where}. Revision {assetLock.Revision} or file {file.Path} does not exist in {assetLock.Repository}; the lock may be wrong or the revision was removed upstream.")
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
}
