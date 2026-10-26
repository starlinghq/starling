namespace Starlings.Acquisition;

/// <summary>How thoroughly a cached file is checked before it is returned.</summary>
public enum CacheVerification
{
    /// <summary>Size and full SHA-256 of every file (default). Hashing reads every byte (about 1 s per 500 MB on one core).</summary>
    Sha256,

    /// <summary>
    /// Size only. Cheap, but does not detect same-size corruption. Use it when the consumer verifies the digest itself,
    /// for example <c>OnnxSessionOptions.ExpectedModelSha256</c>, or for repeated loads of a cache verified earlier.
    /// </summary>
    Size,
}

/// <summary>Options for <see cref="ModelAssetStore"/>. Not a record, so the token never appears in a generated <c>ToString</c>.</summary>
public sealed class ModelAcquisitionOptions
{
    /// <summary>Environment variable that overrides the default cache directory.</summary>
    public const string CacheDirectoryEnvironmentVariable = "STARLING_MODELS";

    /// <summary>Environment variable read for the Hugging Face access token when <see cref="HuggingFaceToken"/> is null.</summary>
    public const string TokenEnvironmentVariable = "HF_TOKEN";

    /// <summary>Default Hub endpoint.</summary>
    public static readonly Uri DefaultEndpoint = new("https://huggingface.co/");

    /// <summary>
    /// Cache root. Files live at <c>&lt;root&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;commit SHA&gt;/&lt;path&gt;</c>. When null:
    /// <c>$STARLING_MODELS</c> if set, else <see cref="DefaultCacheDirectory"/>.
    /// </summary>
    public string? CacheDirectory { get; init; }

    /// <summary>Verification applied to files already in the cache. Fresh downloads are always verified by size and SHA-256.</summary>
    public CacheVerification Verification { get; init; } = CacheVerification.Sha256;

    /// <summary>
    /// Hosts whose explicit <see cref="ModelAssetFile.Source"/> addresses may receive the token: only <c>huggingface.co</c>,
    /// matched exactly (no subdomains), and only over https on the default port.
    /// </summary>
    public static IReadOnlyList<string> TokenHosts { get; } = ["huggingface.co"];

    /// <summary>
    /// Access token for gated or private repositories. When null, <c>$HF_TOKEN</c> is used if set, unless
    /// <see cref="UseEnvironmentToken"/> is false. Sent only over https, as the <c>Authorization</c> header of requests to
    /// <see cref="Endpoint"/> (same host and port) for files without an explicit source, and of requests to explicit
    /// sources on a host in <see cref="TokenHosts"/>; an <c>http://</c> endpoint is used without the token, never with it
    /// in cleartext. The store follows redirects itself and attaches
    /// the header per hop only when that hop's address qualifies, so it never reaches the Hub's storage or CDN hosts or any
    /// other host. Never logged or put in exception messages. (An injected <see cref="HttpClient"/> whose handler follows
    /// redirects itself bypasses the per-hop check; <see cref="SocketsHttpHandler"/> then drops the header on every redirect.)
    /// </summary>
    public string? HuggingFaceToken { get; init; }

    /// <summary>Whether <c>$HF_TOKEN</c> is read when <see cref="HuggingFaceToken"/> is null. Default true.</summary>
    public bool UseEnvironmentToken { get; init; } = true;

    /// <summary>Hub base address (a mirror or a test server). Default <c>https://huggingface.co/</c>.</summary>
    public Uri Endpoint { get; init; } = DefaultEndpoint;

    /// <summary>Attempts per file for transient failures (network errors, 408, 429, 5xx). Default 4. Resumes from the partial file.</summary>
    public int MaxAttempts { get; init; } = 4;

    /// <summary>Delay before the second attempt; doubled for each further attempt. Default 1 s.</summary>
    public TimeSpan RetryDelay { get; init; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Maximum wait for the response headers of one request, and the idle budget of each body read: a transfer that
    /// delivers nothing for this long counts as a transient failure (one of <see cref="MaxAttempts"/>), keeps its partial
    /// file and resumes. Default 100 s.
    /// </summary>
    public TimeSpan ResponseTimeout { get; init; } = TimeSpan.FromSeconds(100);

    /// <summary>
    /// Longest wait for another downloader (a second store, thread or process) that holds a file's lock, after which
    /// <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, ModelDownloadOptions, CancellationToken)"/> throws
    /// <see cref="ModelAssetLockTimeoutException"/>. Null (default) waits until the lock is released or the call is
    /// canceled, as before. While waiting, progress reports <see cref="ModelAssetStage.Waiting"/> once per file.
    /// </summary>
    public TimeSpan? LockWaitTimeout { get; init; }

    /// <summary>
    /// Files of one lock downloaded or verified at the same time. Default 1: one file after the other, in lock order,
    /// as before. Larger values help for locks with many files; the files still land and verify one by one.
    /// </summary>
    public int MaxConcurrentFiles { get; init; } = 1;

    /// <summary>
    /// Further Hub-layout base addresses (mirrors; http or https, checked when the store is created), tried in order after <see cref="Endpoint"/> for a file without an
    /// explicit <see cref="ModelAssetFile.Source"/> when the previous one fails: a download failure after all
    /// <see cref="MaxAttempts"/>, 401/403, 404 or bytes that do not match the lock. Default empty. A mirror receives the
    /// token only under the same rule as any other address (https, and the host of <see cref="Endpoint"/> or a host in
    /// <see cref="TokenHosts"/>), so a mirror on another host never sees it. When every address fails, the failure of
    /// <see cref="Endpoint"/> is thrown and <see cref="ModelAcquisitionException.EndpointFailures"/> holds all of them.
    /// </summary>
    public IReadOnlyList<Uri> FallbackEndpoints { get; init; } = [];

    /// <summary>
    /// Local directories laid out like the cache (<c>&lt;root&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;commit SHA&gt;/&lt;path&gt;</c>),
    /// for example a cache copied from a connected machine or a read-only share. Before any network request,
    /// <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, ModelDownloadOptions, CancellationToken)"/> copies a
    /// missing file from the first of these that has it, verifying size and SHA-256 while copying; a copy that does not
    /// match is discarded and the next source is tried. Nothing in these directories is changed. Default empty.
    /// </summary>
    public IReadOnlyList<string> LocalSources { get; init; } = [];

    /// <summary>
    /// Whether <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, ModelDownloadOptions, CancellationToken)"/> may use
    /// the network. Default true. False is for air-gapped installs: files come only from the cache and
    /// <see cref="LocalSources"/>, and a file found in neither raises <see cref="ModelAssetMissingException"/> before
    /// anything is copied.
    /// </summary>
    public bool AllowNetwork { get; init; } = true;

    /// <summary>
    /// Default cache root: <c>$XDG_CACHE_HOME/starlings/models</c> or <c>~/.cache/starlings/models</c> on
    /// Linux, <c>~/Library/Caches/starlings/models</c> on macOS, <c>%LOCALAPPDATA%\starlings\models</c> on Windows.
    /// </summary>
    public static string DefaultCacheDirectory
    {
        get
        {
            if (OperatingSystem.IsWindows())
            {
                return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "starlings", "models");
            }

            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (OperatingSystem.IsMacOS())
            {
                return Path.Combine(home, "Library", "Caches", "starlings", "models");
            }

            string? xdg = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
            string cache = !string.IsNullOrEmpty(xdg) && Path.IsPathRooted(xdg) ? xdg : Path.Combine(home, ".cache");
            return Path.Combine(cache, "starlings", "models");
        }
    }

    /// <summary>The cache root these options resolve to.</summary>
    public string ResolveCacheDirectory()
    {
        if (!string.IsNullOrEmpty(CacheDirectory))
        {
            return Path.GetFullPath(CacheDirectory);
        }

        string? fromEnvironment = Environment.GetEnvironmentVariable(CacheDirectoryEnvironmentVariable);
        return Path.GetFullPath(string.IsNullOrEmpty(fromEnvironment) ? DefaultCacheDirectory : fromEnvironment);
    }

    internal string? ResolveToken()
    {
        if (!string.IsNullOrEmpty(HuggingFaceToken))
        {
            return HuggingFaceToken;
        }

        return UseEnvironmentToken && Environment.GetEnvironmentVariable(TokenEnvironmentVariable) is { Length: > 0 } token ? token : null;
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"CacheDirectory={CacheDirectory ?? "(default)"}, Verification={Verification}, Endpoint={Endpoint}, " +
        $"FallbackEndpoints={FallbackEndpoints.Count}, LocalSources={LocalSources.Count}, AllowNetwork={AllowNetwork}, " +
        $"MaxConcurrentFiles={MaxConcurrentFiles}, Token={(HuggingFaceToken is null ? "(none)" : "(set)")}";
}

/// <summary>What <see cref="ModelAssetStore"/> is doing with a file.</summary>
public enum ModelAssetStage
{
    /// <summary>Hashing a file already in the cache, or a partial file before resuming.</summary>
    Verifying,

    /// <summary>Receiving bytes.</summary>
    Downloading,

    /// <summary>The file is in the cache and verified.</summary>
    Completed,

    /// <summary>
    /// Another downloader holds the file's lock; this call waits for it (up to <see cref="ModelAcquisitionOptions.LockWaitTimeout"/>)
    /// and then finds the file verified or continues where the other one stopped. Reported once per wait.
    /// </summary>
    Waiting,
}

/// <summary>
/// Progress of one file, with the totals of its lock. <see cref="ModelAssetStore"/> reports one value at a time, also
/// when several files are acquired at once (<see cref="ModelAcquisitionOptions.MaxConcurrentFiles"/>), so an
/// <see cref="IProgress{T}"/> implementation needs no locking of its own; a slow <c>Report</c> slows the download.
/// </summary>
/// <param name="Path">Path inside the repository.</param>
/// <param name="Stage">Current stage.</param>
/// <param name="BytesProcessed">Bytes downloaded or hashed so far, including bytes resumed from a partial file.</param>
/// <param name="TotalBytes">Expected size from the lock.</param>
public readonly record struct ModelAssetProgress(string Path, ModelAssetStage Stage, long BytesProcessed, long TotalBytes)
{
    /// <summary>Zero-based index of <see cref="Path"/> in <see cref="ModelAssetLock.Files"/>.</summary>
    public int FileIndex { get; init; }

    /// <summary>Number of files in the lock; 0 in a value not reported by <see cref="ModelAssetStore"/>.</summary>
    public int FileCount { get; init; }

    /// <summary>
    /// Bytes processed over the whole lock: the latest <see cref="BytesProcessed"/> of every file reported so far
    /// (the full size of a completed file). Decreases only when a cached or resumed file fails its check and is
    /// downloaded again from zero.
    /// </summary>
    public long LockBytesProcessed { get; init; }

    /// <summary>Size of the whole lock in bytes (<see cref="ModelAssetLock.TotalSize"/>).</summary>
    public long LockTotalBytes { get; init; }
}
