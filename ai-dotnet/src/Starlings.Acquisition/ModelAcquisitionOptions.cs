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
    /// <see cref="UseEnvironmentToken"/> is false. Sent only as the <c>Authorization</c> header of requests to
    /// <see cref="Endpoint"/> (same scheme, host and port) for files without an explicit source, and of requests to explicit
    /// sources whose address is https on a host in <see cref="TokenHosts"/>. The store follows redirects itself and attaches
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

    /// <summary>Maximum wait for response headers of one request. Body transfer is bounded only by cancellation. Default 100 s.</summary>
    public TimeSpan ResponseTimeout { get; init; } = TimeSpan.FromSeconds(100);

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
        $"CacheDirectory={CacheDirectory ?? "(default)"}, Verification={Verification}, Endpoint={Endpoint}, Token={(HuggingFaceToken is null ? "(none)" : "(set)")}";
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
}

/// <summary>Progress of one file.</summary>
/// <param name="Path">Path inside the repository.</param>
/// <param name="Stage">Current stage.</param>
/// <param name="BytesProcessed">Bytes downloaded or hashed so far, including bytes resumed from a partial file.</param>
/// <param name="TotalBytes">Expected size from the lock.</param>
public readonly record struct ModelAssetProgress(string Path, ModelAssetStage Stage, long BytesProcessed, long TotalBytes);
