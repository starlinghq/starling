using System.Collections.Concurrent;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace Starlings.Acquisition.Tests;

/// <summary>
/// In-memory Hub: serves <c>/{repo}/resolve/{sha}/{path}</c> with range support and per-request fault injection.
/// </summary>
internal sealed class FakeHub : HttpMessageHandler
{
    public const string Repo = "acme/tiny-model";
    public const string Sha = "0123456789abcdef0123456789abcdef01234567";

    private readonly ConcurrentDictionary<string, byte[]> _files = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, byte[]> _external = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, byte[]> _hosted = new(StringComparer.Ordinal);
    private int _requestCount;
    private int _inFlight;
    private int _maxInFlight;

    public static Uri Endpoint { get; } = new("https://hub.test/");

    public ConcurrentQueue<HttpRequestMessage> Requests { get; } = new();

    public int RequestCount => Volatile.Read(ref _requestCount);

    /// <summary>Most requests seen at once during <see cref="Latency"/> (to show concurrent file downloads).</summary>
    public int MaxInFlight => Volatile.Read(ref _maxInFlight);

    /// <summary>Further hosts (mirrors) that serve the same files as <see cref="Endpoint"/>.</summary>
    public ConcurrentDictionary<string, bool> MirrorHosts { get; } = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Optional hook per request (1-based request number, request): return a response to short-circuit, or null to serve normally.
    /// </summary>
    public Func<int, HttpRequestMessage, HttpResponseMessage?>? Override { get; set; }

    /// <summary>When set, the body of the given request number throws an IOException after this many bytes.</summary>
    public (int Request, int AfterBytes)? BreakBody { get; set; }

    /// <summary>
    /// When set, the body of the given request number delivers nothing after this many bytes: the read completes only
    /// when its cancellation token fires, like a half-open connection.
    /// </summary>
    public (int Request, int AfterBytes)? StallBody { get; set; }

    /// <summary>When true, range headers are ignored and the full body is returned with 200.</summary>
    public bool IgnoreRange { get; set; }

    /// <summary>Delay before the response headers (to make concurrent callers overlap).</summary>
    public TimeSpan Latency { get; set; }

    /// <summary>Called once the body of the given request number has passed this many bytes (for mid-transfer cancellation).</summary>
    public (int Request, int AfterBytes, Action Action)? OnBodyProgress { get; set; }

    public static byte[] RandomBytes(int length, int seed)
    {
        var bytes = new byte[length];
        new Random(seed).NextBytes(bytes);
        return bytes;
    }

    public static string Sha256(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    public void Add(string path, byte[] content) => _files[path] = content;

    /// <summary>Serves <paramref name="content"/> at an absolute address outside the Hub layout (an explicit lock source).</summary>
    public void AddExternal(Uri address, byte[] content) => _external[address.AbsoluteUri] = content;

    /// <summary>Serves <paramref name="content"/> in the Hub layout of another repository and commit (a hosting repository).</summary>
    public void AddHosted(string repository, string revision, string path, byte[] content) =>
        _hosted[$"/{repository}/resolve/{revision}/{path}"] = content;

    public ModelAssetLock LockFor(params string[] paths) =>
        new("tiny", Repo, Sha, paths.Select(p => new ModelAssetFile(p, _files[p].Length, Sha256(_files[p]))));

    public IEnumerable<HttpRequestMessage> RequestsFor(string path) =>
        Requests.Where(r => r.RequestUri!.AbsolutePath.EndsWith("/" + path, StringComparison.Ordinal));

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        int number = Interlocked.Increment(ref _requestCount);
        Requests.Enqueue(request);
        if (Latency > TimeSpan.Zero)
        {
            int now = Interlocked.Increment(ref _inFlight);
            int seen;
            while (now > (seen = Volatile.Read(ref _maxInFlight)) && Interlocked.CompareExchange(ref _maxInFlight, now, seen) != seen)
            {
            }

            try
            {
                await Task.Delay(Latency, cancellationToken);
            }
            finally
            {
                Interlocked.Decrement(ref _inFlight);
            }
        }

        if (Override?.Invoke(number, request) is { } overridden)
        {
            return overridden;
        }

        string prefix = $"/{Repo}/resolve/{Sha}/";
        string absolute = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
        byte[]? content;
        bool hubHost = request.RequestUri.Host == Endpoint.Host || MirrorHosts.ContainsKey(request.RequestUri.Host);
        if (_external.TryGetValue(request.RequestUri.AbsoluteUri, out var external))
        {
            content = external;
        }
        else if (hubHost && _hosted.TryGetValue(absolute, out var hosted))
        {
            content = hosted;
        }
        else if (request.Method != HttpMethod.Get || !hubHost || !absolute.StartsWith(prefix, StringComparison.Ordinal)
            || !_files.TryGetValue(absolute[prefix.Length..], out content))
        {
            return new HttpResponseMessage(HttpStatusCode.NotFound);
        }

        long from = 0;
        var status = HttpStatusCode.OK;
        if (!IgnoreRange && request.Headers.Range is { } range && range.Ranges.Single().From is { } start)
        {
            if (start >= content.Length)
            {
                return new HttpResponseMessage(HttpStatusCode.RequestedRangeNotSatisfiable);
            }

            from = start;
            status = HttpStatusCode.PartialContent;
        }

        var body = new FaultingStream(
            content.AsMemory((int)from),
            BreakBody is { } b && b.Request == number ? b.AfterBytes : null,
            StallBody is { } st && st.Request == number ? st.AfterBytes : null,
            OnBodyProgress is { } o && o.Request == number ? (o.AfterBytes, o.Action) : null);
        var response = new HttpResponseMessage(status) { Content = new StreamContent(body) };
        response.Content.Headers.ContentLength = content.Length - from;
        if (status == HttpStatusCode.PartialContent)
        {
            response.Content.Headers.ContentRange = new ContentRangeHeaderValue(from, content.Length - 1, content.Length);
        }

        return response;
    }

    /// <summary>Read-only stream that can fail after N bytes, like a dropped connection, or stall there, like a half-open one.</summary>
    private sealed class FaultingStream(ReadOnlyMemory<byte> data, int? breakAfter, int? stallAfter, (int AfterBytes, Action Action)? onProgress) : Stream
    {
        private int _position;
        private (int AfterBytes, Action Action)? _onProgress = onProgress;

        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => data.Length;

        public override long Position { get => _position; set => throw new NotSupportedException(); }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            if (breakAfter is { } limit && _position >= limit)
            {
                throw new IOException("Simulated connection reset.");
            }

            int max = breakAfter is { } l ? Math.Min(buffer.Length, l - _position) : buffer.Length;
            int n = Math.Min(Math.Min(max, data.Length - _position), 4096);
            data.Span.Slice(_position, n).CopyTo(buffer);
            _position += n;
            if (_onProgress is { } hook && _position >= hook.AfterBytes)
            {
                _onProgress = null;
                hook.Action();
            }

            return n;
        }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (stallAfter is { } stall && _position >= stall)
            {
                await Task.Delay(Timeout.Infinite, cancellationToken);
            }

            return Read(buffer.Span);
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

        public override void Flush()
        {
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}

/// <summary>A handler that fails the test if any request is made: proves offline paths never touch the network.</summary>
internal sealed class NoNetwork : HttpMessageHandler
{
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
        throw new InvalidOperationException("Unexpected network request: " + request.RequestUri);
}

/// <summary>Temporary cache directory, deleted on dispose.</summary>
internal sealed class TempCache : IDisposable
{
    public TempCache()
    {
        Directory.CreateDirectory(Root);
    }

    public string Root { get; } = Path.Combine(Path.GetTempPath(), "starlings-acq-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try
        {
            Directory.Delete(Root, recursive: true);
        }
        catch (IOException)
        {
        }
    }
}
