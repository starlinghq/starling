using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;

namespace Starlings.Runtime.OnnxWeb;

/// <summary>Download progress of one <see cref="OnnxWebAssets.FetchAsync"/> call.</summary>
/// <param name="Url">The requested URL.</param>
/// <param name="BytesReceived">Bytes received so far (decoded).</param>
/// <param name="TotalBytes">Total bytes when the response states them (no content encoding), else <see langword="null"/>.</param>
public readonly record struct OnnxWebDownloadProgress(string Url, long BytesReceived, long? TotalBytes);

/// <summary>Options of <see cref="OnnxWebAssets.FetchAsync"/>.</summary>
public sealed record OnnxWebFetchOptions
{
    /// <summary>
    /// Cache Storage cache that keeps verified copies across page loads, or <see langword="null"/> for none. Only bytes
    /// that matched the expected SHA-256 are stored; unverified fetches are never cached. A cached copy is verified
    /// again when read and fetched anew if it no longer matches. Default <see cref="OnnxWebAssets.DefaultCacheName"/>.
    /// </summary>
    public string? CacheName { get; init; } = OnnxWebAssets.DefaultCacheName;
}

/// <summary>
/// Fetches model files in the browser. Bytes stay in JavaScript: an <see cref="OnnxWebAsset"/> is a handle that an
/// <see cref="OnnxWebSession"/> consumes directly, or that a caller copies into .NET (a tokenizer file, for example).
/// </summary>
public static class OnnxWebAssets
{
    /// <summary>The default Cache Storage cache name.</summary>
    public const string DefaultCacheName = "starlings-models-v1";

    private static int _nextRequestId;

    /// <summary>
    /// Fetches <paramref name="url"/> (relative to the page's base URI, or absolute; cross-origin hosts must send
    /// CORS headers) and verifies its SHA-256 with the browser's SubtleCrypto. Works before
    /// <see cref="OnnxWebRuntime.InitializeAsync"/>.
    /// </summary>
    /// <param name="url">The file URL.</param>
    /// <param name="expectedSha256">Lower- or upper-case hex SHA-256 the bytes must match, or <see langword="null"/> to skip verification (and caching).</param>
    /// <param name="options">Cache settings.</param>
    /// <param name="progress">Receives progress about every 4 MiB and at the end.</param>
    /// <param name="cancellationToken">Aborts the HTTP request.</param>
    /// <exception cref="InvalidDataException">The bytes do not match <paramref name="expectedSha256"/>.</exception>
    /// <exception cref="HttpRequestException">The request failed or returned a non-success status.</exception>
    [SupportedOSPlatform("browser")]
    public static async Task<OnnxWebAsset> FetchAsync(
        string url,
        string? expectedSha256,
        OnnxWebFetchOptions? options = null,
        IProgress<OnnxWebDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(url);
        if (expectedSha256 is not null && !IsSha256Hex(expectedSha256))
        {
            throw new ArgumentException("Expected a SHA-256 as 64 hex characters.", nameof(expectedSha256));
        }

        options ??= new OnnxWebFetchOptions();
        cancellationToken.ThrowIfCancellationRequested();
        await OnnxWebRuntime.EnsureModuleAsync().ConfigureAwait(false);

        int requestId = Interlocked.Increment(ref _nextRequestId);
        Action<double, double> onProgress = progress is null
            ? static (_, _) => { }
            : (received, total) => progress.Report(new OnnxWebDownloadProgress(url, (long)received, total >= 0 ? (long)total : null));
        using var registration = cancellationToken.Register(static state => OnnxWebInterop.AbortFetch((int)state!), requestId);
        JSObject handle;
        try
        {
            handle = await OnnxWebInterop.FetchAsset(url, expectedSha256?.ToLowerInvariant(), options.CacheName, requestId, onProgress).ConfigureAwait(false);
        }
        catch (JSException) when (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException(cancellationToken);
        }
        catch (JSException ex) when (ex.Message.Contains("does not match the expected", StringComparison.Ordinal))
        {
            throw new InvalidDataException(ex.Message, ex);
        }
        catch (JSException ex) when (ex.Message.Contains("SubtleCrypto", StringComparison.Ordinal))
        {
            throw new NotSupportedException(ex.Message, ex);
        }
        catch (JSException ex)
        {
            throw new HttpRequestException($"Fetching {url} failed: {ex.Message}", ex);
        }

        return new OnnxWebAsset(handle);
    }

    internal static bool IsSha256Hex(string value)
    {
        if (value.Length != 64)
        {
            return false;
        }

        foreach (char c in value)
        {
            if (!char.IsAsciiHexDigit(c))
            {
                return false;
            }
        }

        return true;
    }
}

/// <summary>
/// A fetched file whose bytes live in JavaScript. <see cref="OnnxWebSession.CreateAsync"/> consumes them (onnxruntime-web
/// copies or transfers them), after which <see cref="IsAvailable"/> is <see langword="false"/>. Dispose to drop the
/// bytes early.
/// </summary>
[SupportedOSPlatform("browser")]
public sealed class OnnxWebAsset : IDisposable
{
    private readonly JSObject _handle;
    private int _disposed;

    internal OnnxWebAsset(JSObject handle)
    {
        _handle = handle;
        Url = handle.GetPropertyAsString("url") ?? string.Empty;
        Sha256 = handle.GetPropertyAsString("sha256");
        ByteLength = (long)handle.GetPropertyAsDouble("byteLength");
        FromCache = handle.GetPropertyAsBoolean("fromCache");
        Cached = handle.GetPropertyAsBoolean("cached");
        CacheError = handle.GetPropertyAsString("cacheError");
    }

    /// <summary>Absolute URL fetched.</summary>
    public string Url { get; }

    /// <summary>SHA-256 of the bytes (lower-case hex), or <see langword="null"/> when not verified and SubtleCrypto was unavailable.</summary>
    public string? Sha256 { get; }

    /// <summary>Size in bytes.</summary>
    public long ByteLength { get; }

    /// <summary>Whether the bytes came from Cache Storage instead of the network.</summary>
    public bool FromCache { get; }

    /// <summary>Whether a verified copy is now in Cache Storage.</summary>
    public bool Cached { get; }

    /// <summary>Why storing in Cache Storage failed (quota, for example), or <see langword="null"/>.</summary>
    public string? CacheError { get; }

    /// <summary>Whether the bytes are still held (not disposed and not consumed by a session).</summary>
    public bool IsAvailable => Volatile.Read(ref _disposed) == 0 && OnnxWebInterop.AssetLength(_handle) >= 0;

    internal JSObject Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _handle;
        }
    }

    /// <summary>Copies the bytes into a new .NET array.</summary>
    /// <exception cref="InvalidOperationException">The bytes were consumed by a session.</exception>
    public byte[] ToArray()
    {
        long length = (long)OnnxWebInterop.AssetLength(Handle);
        if (length < 0)
        {
            throw new InvalidOperationException("The asset bytes were consumed by a session.");
        }

        var bytes = new byte[checked((int)length)];
        OnnxWebInterop.CopyAsset(_handle, bytes);
        return bytes;
    }

    /// <summary>Releases the bytes and the JavaScript handle. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            OnnxWebInterop.ReleaseAsset(_handle);
            _handle.Dispose();
        }
    }
}
