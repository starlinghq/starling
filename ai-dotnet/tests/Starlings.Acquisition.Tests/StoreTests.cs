using System.Collections.Concurrent;
using System.Net;

namespace Starlings.Acquisition.Tests;

public sealed class StoreTests : IDisposable
{
    private const string Model = "onnx/model.onnx";
    private const string Tokenizer = "tokenizer.json";

    private readonly TempCache _cache = new();
    private readonly FakeHub _hub = new();
    private readonly byte[] _model = FakeHub.RandomBytes(300_000, 1);
    private readonly byte[] _tokenizer = FakeHub.RandomBytes(10_000, 2);

    public StoreTests()
    {
        _hub.Add(Model, _model);
        _hub.Add(Tokenizer, _tokenizer);
    }

    private ModelAssetLock Lock => _hub.LockFor(Model, Tokenizer);

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        _hub.Dispose();
        _cache.Dispose();
    }

    [Fact]
    public async Task DownloadsVerifiesAndPlacesFilesInTheCacheLayout()
    {
        var progress = new SyncProgress();
        using var store = Store();
        var snapshot = await store.DownloadAsync(Lock, progress, Ct);

        Assert.Equal(Path.Combine(_cache.Root, "acme", "tiny-model", FakeHub.Sha), snapshot.Directory);
        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        Assert.Equal(_tokenizer, File.ReadAllBytes(snapshot.GetFilePath(Tokenizer)));
        Assert.Equal(CacheVerification.Sha256, snapshot.Verification);
        Assert.Equal(2, _hub.RequestCount);
        Assert.All(_hub.Requests, r => Assert.Contains($"/resolve/{FakeHub.Sha}/", r.RequestUri!.AbsolutePath, StringComparison.Ordinal));
        Assert.All(_hub.Requests, r => Assert.Null(r.Headers.Authorization));
        Assert.Contains(progress.Items, p => p.Path == Model && p.Stage == ModelAssetStage.Completed && p.BytesProcessed == _model.Length);
        Assert.Contains(progress.Items, p => p.Path == Model && p.Stage == ModelAssetStage.Downloading);

        // Only complete, verified files in the snapshot directory; partials and locks are elsewhere.
        var inSnapshot = Directory.GetFiles(snapshot.Directory, "*", SearchOption.AllDirectories)
            .Select(f => Path.GetRelativePath(snapshot.Directory, f).Replace('\\', '/')).Order().ToArray();
        Assert.Equal([Model, Tokenizer], inSnapshot);
        Assert.Empty(Directory.GetFiles(_cache.Root, "*.partial", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task SecondDownloadReusesTheCacheWithoutNetwork()
    {
        using (var first = Store())
        {
            await first.DownloadAsync(Lock, cancellationToken: Ct);
        }

        using var offline = new ModelAssetStore(Options(), new HttpClient(new NoNetwork()));
        var snapshot = await offline.DownloadAsync(Lock, cancellationToken: Ct);
        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
    }

    [Fact]
    public async Task OfflineHitVerifiesAndNeverTouchesTheNetwork()
    {
        using (var first = Store())
        {
            await first.DownloadAsync(Lock, cancellationToken: Ct);
        }

        using var offline = new ModelAssetStore(Options(), new HttpClient(new NoNetwork()));
        var snapshot = offline.OpenLocal(Lock);
        Assert.Equal(_tokenizer, File.ReadAllBytes(snapshot.GetFilePath(Tokenizer)));
        Assert.Throws<KeyNotFoundException>(() => snapshot.GetFilePath("other.bin"));
    }

    [Fact]
    public async Task VerifiedDigestIsCarriedOnlyWhenTheSnapshotHashedTheFiles()
    {
        using (var first = Store())
        {
            var downloaded = await first.DownloadAsync(Lock, cancellationToken: Ct);
            Assert.Equal(FakeHub.Sha256(_model), downloaded.GetVerifiedSha256(Model)); // hashed in transit
        }

        using var full = new ModelAssetStore(Options(), new HttpClient(new NoNetwork()));
        var hashed = full.OpenLocal(Lock);
        Assert.Equal(FakeHub.Sha256(_model), hashed.GetVerifiedSha256(Model));
        Assert.Equal(FakeHub.Sha256(_tokenizer), hashed.GetVerifiedSha256(Tokenizer));
        Assert.Throws<KeyNotFoundException>(() => hashed.GetVerifiedSha256("other.bin"));

        using var sizeOnly = new ModelAssetStore(Options(CacheVerification.Size), new HttpClient(new NoNetwork()));
        Assert.Null(sizeOnly.OpenLocal(Lock).GetVerifiedSha256(Model));
        Assert.Null(ModelAssetStore.VerifyDirectory(Lock, hashed.Directory, CacheVerification.Size).GetVerifiedSha256(Model));
        Assert.Equal(FakeHub.Sha256(_model), ModelAssetStore.VerifyDirectory(Lock, hashed.Directory).GetVerifiedSha256(Model));
    }

    [Fact]
    public void OfflineMissNamesTheFilesAndHowToDownload()
    {
        using var offline = new ModelAssetStore(Options(), new HttpClient(new NoNetwork()));
        var ex = Assert.Throws<ModelAssetMissingException>(() => offline.OpenLocal(Lock));
        Assert.Equal([Model, Tokenizer], ex.MissingPaths);
        Assert.Equal(offline.GetSnapshotDirectory(Lock), ex.SnapshotDirectory);
        Assert.Equal(FakeHub.Repo, ex.Repository);
        Assert.Equal(FakeHub.Sha, ex.Revision);
        Assert.Contains("DownloadAsync", ex.Message, StringComparison.Ordinal);
        Assert.Contains("download tiny", ex.Message, StringComparison.Ordinal);
        Assert.Contains("STARLING_MODELS", ex.Message, StringComparison.Ordinal);
        Assert.Contains(Model, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task OfflinePartialCacheReportsOnlyTheMissingFile()
    {
        using (var first = Store())
        {
            await first.DownloadAsync(_hub.LockFor(Tokenizer), cancellationToken: Ct);
        }

        using var offline = new ModelAssetStore(Options());
        var ex = Assert.Throws<ModelAssetMissingException>(() => offline.OpenLocal(Lock));
        Assert.Equal([Model], ex.MissingPaths);
    }

    [Fact]
    public async Task OfflineCorruptionIsAnIntegrityErrorWithExpectedAndActual()
    {
        string path;
        using (var first = Store())
        {
            path = (await first.DownloadAsync(Lock, cancellationToken: Ct)).GetFilePath(Model);
        }

        // Same size, one byte flipped: caught by the SHA-256 check, not by the size check.
        var bytes = File.ReadAllBytes(path);
        bytes[1234] ^= 0xFF;
        File.WriteAllBytes(path, bytes);

        using var full = new ModelAssetStore(Options());
        var ex = Assert.Throws<ModelAssetIntegrityException>(() => full.OpenLocal(Lock));
        Assert.Equal(Model, ex.AssetPath);
        Assert.Equal(path, ex.FilePath);
        Assert.Equal(FakeHub.Sha256(_model), ex.ExpectedSha256);
        Assert.Equal(FakeHub.Sha256(bytes), ex.ActualSha256);
        Assert.Contains(ex.ExpectedSha256!, ex.Message, StringComparison.Ordinal);
        Assert.Contains(ex.ActualSha256!, ex.Message, StringComparison.Ordinal);

        using var sizeOnly = new ModelAssetStore(Options(CacheVerification.Size));
        Assert.Equal(CacheVerification.Size, sizeOnly.OpenLocal(Lock).Verification);

        // Truncation: caught by the size check in both modes.
        File.WriteAllBytes(path, bytes[..1000]);
        var truncated = Assert.Throws<ModelAssetIntegrityException>(() => sizeOnly.OpenLocal(Lock));
        Assert.Equal(_model.Length, truncated.ExpectedSize);
        Assert.Equal(1000, truncated.ActualSize);
        Assert.Null(truncated.ActualSha256);
    }

    [Fact]
    public async Task DownloadReplacesACorruptCachedFile()
    {
        string path;
        using (var first = Store())
        {
            path = (await first.DownloadAsync(Lock, cancellationToken: Ct)).GetFilePath(Model);
        }

        File.WriteAllBytes(path, new byte[_model.Length]);
        using var store = Store();
        await store.DownloadAsync(Lock, cancellationToken: Ct);
        Assert.Equal(_model, File.ReadAllBytes(path));
    }

    [Fact]
    public async Task WrongBytesFromTheServerAreRejectedAndNothingIsCached()
    {
        var wrong = _hub.LockFor(Model);
        _hub.Add(Model, FakeHub.RandomBytes(_model.Length, 99));
        using var store = Store();
        var ex = await Assert.ThrowsAsync<ModelAssetIntegrityException>(() => store.DownloadAsync(wrong, cancellationToken: Ct));
        Assert.Equal(FakeHub.Sha256(_model), ex.ExpectedSha256);
        Assert.NotEqual(ex.ExpectedSha256, ex.ActualSha256);
        Assert.False(File.Exists(Path.Combine(store.GetSnapshotDirectory(wrong), "onnx", "model.onnx")));
        Assert.Empty(Directory.GetFiles(_cache.Root, "*.partial", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task OversizedResponseIsRejected()
    {
        var expected = _hub.LockFor(Tokenizer);
        _hub.Add(Tokenizer, [.. _tokenizer, 0, 1, 2]);
        using var store = Store();
        var ex = await Assert.ThrowsAsync<ModelAssetIntegrityException>(() => store.DownloadAsync(expected, cancellationToken: Ct));
        Assert.True(ex.ActualSize > ex.ExpectedSize);

        // The bytes are not hashed once they exceed the lock: the digest is null (as documented), the message explains.
        Assert.Null(ex.ActualSha256);
        Assert.Equal(FakeHub.Sha256(_tokenizer), ex.ExpectedSha256);
        Assert.Contains("longer than the lock", ex.Message, StringComparison.Ordinal);
        Assert.Contains(_tokenizer.Length.ToString(System.Globalization.CultureInfo.InvariantCulture), ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("not hashed: longer", ex.ActualSha256 ?? string.Empty, StringComparison.Ordinal);
    }

    [Fact]
    public async Task OfflineMissingAndCorruptFilesReportTheMissingOnesFirst()
    {
        string modelPath;
        using (var first = Store())
        {
            var snapshot = await first.DownloadAsync(Lock, cancellationToken: Ct);
            modelPath = snapshot.GetFilePath(Model);
            File.Delete(snapshot.GetFilePath(Tokenizer));
        }

        // The lock lists the model (corrupt, same size) before the tokenizer (missing): the pass must not stop at the first.
        var bytes = File.ReadAllBytes(modelPath);
        bytes[99] ^= 0xFF;
        File.WriteAllBytes(modelPath, bytes);

        using var offline = new ModelAssetStore(Options(), new HttpClient(new NoNetwork()));
        var ex = Assert.Throws<ModelAssetMissingException>(() => offline.OpenLocal(Lock));
        Assert.Equal([Tokenizer], ex.MissingPaths);
        Assert.Contains(Tokenizer, ex.Message, StringComparison.Ordinal);
        Assert.Contains("also failed verification", ex.Message, StringComparison.Ordinal);
        Assert.Contains(Model, ex.Message, StringComparison.Ordinal);
        var corrupt = Assert.IsType<ModelAssetIntegrityException>(ex.InnerException);
        Assert.Equal(Model, corrupt.AssetPath);
        Assert.Equal(FakeHub.Sha256(bytes), corrupt.ActualSha256);

        // With every file present, the integrity failure is what is thrown.
        File.WriteAllBytes(Path.Combine(offline.GetSnapshotDirectory(Lock), Tokenizer), _tokenizer);
        Assert.Equal(Model, Assert.Throws<ModelAssetIntegrityException>(() => offline.OpenLocal(Lock)).AssetPath);
    }

    [Fact]
    public async Task CacheWriteFailureIsNotRetriedAndNamesThePath()
    {
        // A directory where the partial file goes makes every open of it fail, like a full disk or a read-only cache
        // would (the tests may run as root, so permissions cannot be relied on).
        string partial = Path.Combine(_cache.Root, ".downloads", "acme", "tiny-model", FakeHub.Sha, "onnx", "model.onnx.partial");
        Directory.CreateDirectory(partial);
        using var store = Store();
        var ex = await Assert.ThrowsAsync<ModelAcquisitionException>(() => store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct));

        Assert.Equal(Model, ex.AssetPath);
        Assert.Equal(FakeHub.Repo, ex.Repository);
        Assert.Contains(partial, ex.Message, StringComparison.Ordinal);
        Assert.Contains("not retried", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("calling DownloadAsync again", ex.Message, StringComparison.Ordinal);
        Assert.NotNull(ex.InnerException);
        Assert.Equal(1, _hub.RequestCount);
    }

    [Fact]
    public async Task StalledBodyIsATransientFailureThatResumes()
    {
        // The first body stops delivering after 100 000 bytes without closing; the per-read idle budget ends the
        // attempt, the partial file is kept, and the second request resumes from it.
        _hub.StallBody = (1, 100_000);
        var options = new ModelAcquisitionOptions
        {
            CacheDirectory = _cache.Root,
            Endpoint = FakeHub.Endpoint,
            RetryDelay = TimeSpan.Zero,
            ResponseTimeout = TimeSpan.FromMilliseconds(200),
            UseEnvironmentToken = false,
        };
        using var store = Store(options);
        var snapshot = await store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct);

        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        var requests = _hub.Requests.ToArray();
        Assert.Equal(2, requests.Length);
        Assert.Null(requests[0].Headers.Range);
        long resumedFrom = requests[1].Headers.Range!.Ranges.Single().From!.Value;
        Assert.InRange(resumedFrom, 100_000, _model.Length - 1);
    }

    [Fact]
    public async Task StalledBodyExhaustingAttemptsIsADownloadErrorThatKeepsThePartialFile()
    {
        _hub.StallBody = (1, 100_000);
        _hub.Override = (n, _) => n > 1 ? new HttpResponseMessage(HttpStatusCode.BadGateway) : null;
        var options = new ModelAcquisitionOptions
        {
            CacheDirectory = _cache.Root,
            Endpoint = FakeHub.Endpoint,
            RetryDelay = TimeSpan.Zero,
            MaxAttempts = 2,
            ResponseTimeout = TimeSpan.FromMilliseconds(200),
            UseEnvironmentToken = false,
        };
        using var store = Store(options);
        var ex = await Assert.ThrowsAsync<ModelAssetDownloadException>(() => store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct));
        Assert.Equal(2, ex.Attempts);
        var partial = Assert.Single(Directory.GetFiles(_cache.Root, "*.partial", SearchOption.AllDirectories));
        Assert.InRange(new FileInfo(partial).Length, 100_000, _model.Length - 1);
    }

    [Fact]
    public async Task StalledBodyAsTheLastAttemptReportsTheStallNotACancellation()
    {
        // The stall is the last error: the message must say that nothing was received, not that the operation was canceled.
        _hub.StallBody = (1, 100_000);
        var options = new ModelAcquisitionOptions
        {
            CacheDirectory = _cache.Root,
            Endpoint = FakeHub.Endpoint,
            RetryDelay = TimeSpan.Zero,
            MaxAttempts = 1,
            ResponseTimeout = TimeSpan.FromMilliseconds(200),
            UseEnvironmentToken = false,
        };
        using var store = Store(options);
        var ex = await Assert.ThrowsAsync<ModelAssetDownloadException>(() => store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct));
        Assert.Equal(1, ex.Attempts);
        Assert.Contains("No data was received for 0.2 s after", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("canceled", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task CallerCancellationDuringAStallIsCancellationNotATimeout()
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(Ct);
        _hub.StallBody = (1, 100_000);
        _hub.OnBodyProgress = (1, 100_000, () => cts.CancelAfter(TimeSpan.FromMilliseconds(100)));
        var options = new ModelAcquisitionOptions
        {
            CacheDirectory = _cache.Root,
            Endpoint = FakeHub.Endpoint,
            RetryDelay = TimeSpan.Zero,
            ResponseTimeout = TimeSpan.FromSeconds(30),
            UseEnvironmentToken = false,
        };
        using var store = Store(options);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => store.DownloadAsync(_hub.LockFor(Model), cancellationToken: cts.Token));
        Assert.Equal(1, _hub.RequestCount);
        Assert.Single(Directory.GetFiles(_cache.Root, "*.partial", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task Unauthorized401WithoutTokenExplainsGatingAndToken()
    {
        _hub.Override = (_, _) => new HttpResponseMessage(HttpStatusCode.Unauthorized);
        using var store = Store();
        var ex = await Assert.ThrowsAsync<ModelAssetAccessDeniedException>(() => store.DownloadAsync(Lock, cancellationToken: Ct));
        Assert.Equal(HttpStatusCode.Unauthorized, ex.StatusCode);
        Assert.False(ex.TokenProvided);
        Assert.Contains("gated", ex.Message, StringComparison.Ordinal);
        Assert.Contains("HF_TOKEN", ex.Message, StringComparison.Ordinal);
        Assert.Contains("HuggingFaceToken", ex.Message, StringComparison.Ordinal);
        Assert.Equal(1, _hub.RequestCount);
    }

    [Fact]
    public async Task Forbidden403WithTokenSendsBearerAndNeverEchoesIt()
    {
        const string token = "hf_test_token_do_not_print";
        _hub.Override = (_, _) => new HttpResponseMessage(HttpStatusCode.Forbidden);
        using var store = Store(new ModelAcquisitionOptions { CacheDirectory = _cache.Root, Endpoint = FakeHub.Endpoint, HuggingFaceToken = token, RetryDelay = TimeSpan.Zero });
        var ex = await Assert.ThrowsAsync<ModelAssetAccessDeniedException>(() => store.DownloadAsync(Lock, cancellationToken: Ct));
        Assert.Equal(HttpStatusCode.Forbidden, ex.StatusCode);
        Assert.True(ex.TokenProvided);
        Assert.Contains("license", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain(token, ex.ToString(), StringComparison.Ordinal);
        var auth = _hub.Requests.Single().Headers.Authorization!;
        Assert.Equal("Bearer", auth.Scheme);
        Assert.Equal(token, auth.Parameter);
    }

    [Fact]
    public async Task NotFound404IsTypedAndNotRetried()
    {
        var missing = new ModelAssetLock("tiny", FakeHub.Repo, FakeHub.Sha, [new ModelAssetFile("nope.bin", 1, new string('a', 64))]);
        using var store = Store();
        var ex = await Assert.ThrowsAsync<ModelAssetNotFoundException>(() => store.DownloadAsync(missing, cancellationToken: Ct));
        Assert.Equal("nope.bin", ex.AssetPath);
        Assert.Contains(FakeHub.Sha, ex.Message, StringComparison.Ordinal);
        Assert.Equal(1, _hub.RequestCount);
    }

    [Fact]
    public async Task TransientServerErrorsAreRetried()
    {
        _hub.Override = (n, _) => n <= 2 ? new HttpResponseMessage(n == 1 ? HttpStatusCode.ServiceUnavailable : HttpStatusCode.TooManyRequests) : null;
        using var store = Store();
        var snapshot = await store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct);
        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        Assert.Equal(3, _hub.RequestCount);
    }

    [Fact]
    public async Task PersistentFailureIsADownloadErrorAfterMaxAttempts()
    {
        _hub.Override = (_, _) => new HttpResponseMessage(HttpStatusCode.BadGateway);
        using var store = Store();
        var ex = await Assert.ThrowsAsync<ModelAssetDownloadException>(() => store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct));
        Assert.Equal(HttpStatusCode.BadGateway, ex.StatusCode);
        Assert.Equal(4, ex.Attempts);
        Assert.Equal(4, _hub.RequestCount);
    }

    [Fact]
    public async Task NetworkExceptionIsRetried()
    {
        _hub.Override = (n, _) => n == 1 ? throw new HttpRequestException("Simulated DNS failure.") : null;
        using var store = Store();
        await store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct);
        Assert.Equal(2, _hub.RequestCount);
    }

    [Fact]
    public async Task InterruptedTransferResumesWithARangeRequest()
    {
        _hub.BreakBody = (1, 100_000);
        using var store = Store();
        var snapshot = await store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct);

        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        var requests = _hub.Requests.ToArray();
        Assert.Equal(2, requests.Length);
        Assert.Null(requests[0].Headers.Range);
        Assert.Equal(100_000, requests[1].Headers.Range!.Ranges.Single().From);
    }

    [Fact]
    public async Task ServerIgnoringRangeRestartsFromZero()
    {
        _hub.BreakBody = (1, 100_000);
        _hub.IgnoreRange = true;
        using var store = Store();
        var snapshot = await store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct);
        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
    }

    [Fact]
    public async Task CancellationKeepsThePartialFileAndTheNextCallResumes()
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(Ct);

        // Cancel in the middle of the body: the next read observes the token.
        _hub.OnBodyProgress = (1, 150_000, cts.Cancel);
        var progress = new SyncProgress();
        using (var store = Store())
        {
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => store.DownloadAsync(_hub.LockFor(Model), progress, cts.Token));
        }

        var partial = Assert.Single(Directory.GetFiles(_cache.Root, "*.partial", SearchOption.AllDirectories));
        long kept = new FileInfo(partial).Length;
        Assert.InRange(kept, 1, _model.Length - 1);

        using (var store = Store())
        {
            var snapshot = await store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct);
            Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        }

        Assert.Equal(kept, _hub.Requests.Last().Headers.Range!.Ranges.Single().From);
    }

    [Fact]
    public async Task CorruptPartialFileIsDiscardedAndDownloadedAgain()
    {
        _hub.BreakBody = (1, 100_000);
        _hub.Override = (n, _) => n == 2 ? throw new HttpRequestException("stop after the partial") : null;
        using (var store = Store(maxAttempts: 2))
        {
            await Assert.ThrowsAsync<ModelAssetDownloadException>(() => store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct));
        }

        // Damage the kept prefix: the resumed result fails the digest, so the store retries once from zero.
        var partial = Assert.Single(Directory.GetFiles(_cache.Root, "*.partial", SearchOption.AllDirectories));
        var bytes = File.ReadAllBytes(partial);
        bytes[10] ^= 0xFF;
        File.WriteAllBytes(partial, bytes);

        _hub.Override = null;
        using (var store = Store())
        {
            var snapshot = await store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct);
            Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        }

        var ranges = _hub.Requests.Skip(2).Select(r => r.Headers.Range?.Ranges.Single().From).ToArray();
        Assert.Equal(new long?[] { 100_000, null }, ranges);
    }

    [Fact]
    public async Task ConcurrentAcquisitionsOfTheSameAssetDownloadOnce()
    {
        _hub.Latency = TimeSpan.FromMilliseconds(200);
        var assetLock = _hub.LockFor(Model, Tokenizer);
        var stores = Enumerable.Range(0, 8).Select(_ => Store()).ToArray();
        try
        {
            var snapshots = await Task.WhenAll(stores.Select(s => s.DownloadAsync(assetLock, cancellationToken: Ct)));
            Assert.All(snapshots, s => Assert.Equal(_model, File.ReadAllBytes(s.GetFilePath(Model))));
        }
        finally
        {
            foreach (var s in stores)
            {
                s.Dispose();
            }
        }

        Assert.Single(_hub.RequestsFor(Model));
        Assert.Single(_hub.RequestsFor(Tokenizer));
    }

    [Fact]
    public async Task VerifyDirectoryAcceptsAnExistingSnapshot()
    {
        string dir;
        using (var store = Store())
        {
            dir = (await store.DownloadAsync(Lock, cancellationToken: Ct)).Directory;
        }

        var snapshot = ModelAssetStore.VerifyDirectory(Lock, dir);
        Assert.Equal(dir, snapshot.Directory);
        var ex = Assert.Throws<ModelAssetMissingException>(() => ModelAssetStore.VerifyDirectory(Lock, Path.Combine(dir, "nowhere")));
        Assert.Equal(2, ex.MissingPaths.Count);
    }

    [Fact]
    public async Task DisposedStoreRejectsCalls()
    {
        var store = Store();
        store.Dispose();
        Assert.Throws<ObjectDisposedException>(() => store.OpenLocal(Lock));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => store.DownloadAsync(Lock, cancellationToken: Ct));
    }

    [Fact]
    public async Task ExplicitSourceDownloadsIntoTheSnapshotWithoutTheHubToken()
    {
        const string token = "hf_test_token_hub_only";
        var source = new Uri("https://releases.test/acme/tiny-model/v1/converted.onnx");
        byte[] converted = FakeHub.RandomBytes(50_000, 7);
        _hub.AddExternal(source, converted);
        var mixed = new ModelAssetLock("tiny", FakeHub.Repo, FakeHub.Sha,
        [
            new ModelAssetFile(Tokenizer, _tokenizer.Length, FakeHub.Sha256(_tokenizer)),
            new ModelAssetFile("onnx/converted.onnx", converted.Length, FakeHub.Sha256(converted), source),
        ]);
        using var withToken = new ModelAssetStore(
            new ModelAcquisitionOptions { CacheDirectory = _cache.Root, Endpoint = FakeHub.Endpoint, HuggingFaceToken = token, RetryDelay = TimeSpan.Zero, UseEnvironmentToken = false },
            new HttpClient(_hub, disposeHandler: false));
        var snapshot = await withToken.DownloadAsync(mixed, cancellationToken: Ct);

        Assert.Equal(converted, File.ReadAllBytes(snapshot.GetFilePath("onnx/converted.onnx")));
        Assert.Equal(Path.Combine(withToken.GetSnapshotDirectory(mixed), "onnx", "converted.onnx"), snapshot.GetFilePath("onnx/converted.onnx"));
        var explicitRequest = _hub.Requests.Single(r => r.RequestUri == source);
        Assert.Null(explicitRequest.Headers.Authorization);
        var hubRequest = _hub.RequestsFor(Tokenizer).Single();
        Assert.Equal(token, hubRequest.Headers.Authorization!.Parameter);

        // Offline resolution verifies the explicit-source file like any other.
        using var offline = new ModelAssetStore(Options(), new HttpClient(new NoNetwork()));
        Assert.Equal(snapshot.Directory, offline.OpenLocal(mixed).Directory);
    }

    [Fact]
    public async Task ExplicitSourceNotPublishedIsTypedAndNotRetried()
    {
        var source = new Uri("https://releases.test/acme/tiny-model/unpublished/converted.onnx");
        var missing = new ModelAssetLock("tiny", FakeHub.Repo, FakeHub.Sha, [new ModelAssetFile("onnx/converted.onnx", 10, new string('a', 64), source)]);
        using var store = Store();
        var ex = await Assert.ThrowsAsync<ModelAssetNotFoundException>(() => store.DownloadAsync(missing, cancellationToken: Ct));
        Assert.Equal("onnx/converted.onnx", ex.AssetPath);
        Assert.Contains("explicit source", ex.Message, StringComparison.Ordinal);
        Assert.Contains(source.AbsoluteUri, ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("HF_TOKEN", ex.Message, StringComparison.Ordinal);
        Assert.Equal(1, _hub.RequestCount);
    }

    private const string HubToken = "hf_test_token_never_printed";
    private static readonly Uri HubSource = new("https://huggingface.co/owner/hosting/resolve/fedcba9876543210fedcba9876543210fedcba98/onnx/built.onnx");

    private static ModelAssetLock SourceLock(Uri source, byte[] content) =>
        new("tiny", FakeHub.Repo, FakeHub.Sha, [new ModelAssetFile("onnx/built.onnx", content.Length, FakeHub.Sha256(content), source)]);

    private ModelAssetStore TokenStore(string? token = HubToken) => new(
        new ModelAcquisitionOptions { CacheDirectory = _cache.Root, Endpoint = FakeHub.Endpoint, HuggingFaceToken = token, RetryDelay = TimeSpan.Zero, UseEnvironmentToken = false },
        new HttpClient(_hub, disposeHandler: false));

    [Fact]
    public async Task TokenIsSentToAnHttpsHuggingFaceSource()
    {
        byte[] built = FakeHub.RandomBytes(20_000, 11);
        _hub.AddExternal(HubSource, built);
        using var store = TokenStore();
        var snapshot = await store.DownloadAsync(SourceLock(HubSource, built), cancellationToken: Ct);

        Assert.Equal(built, File.ReadAllBytes(snapshot.GetFilePath("onnx/built.onnx")));
        var auth = _hub.Requests.Single().Headers.Authorization!;
        Assert.Equal("Bearer", auth.Scheme);
        Assert.Equal(HubToken, auth.Parameter);
    }

    [Theory]
    [InlineData("https://releases.test/owner/hosting/built.onnx")]
    [InlineData("https://huggingface.co:8443/owner/hosting/resolve/main/built.onnx")]
    [InlineData("https://cdn.huggingface.co/owner/hosting/built.onnx")]
    [InlineData("https://huggingface.co.example.test/owner/hosting/built.onnx")]
    [InlineData("https://cas-bridge.xethub.hf.co/xet-bridge-us/built.onnx")]
    [InlineData("https://hub.test/acme/tiny-model/resolve/0123456789abcdef0123456789abcdef01234567/onnx/built.onnx")]
    public async Task TokenIsNotSentToAnyOtherSourceHost(string address)
    {
        // The last case is the configured Hub endpoint itself: an explicit source there still gets no token (only
        // huggingface.co is on the allowlist for explicit sources).
        var source = new Uri(address);
        byte[] built = FakeHub.RandomBytes(5_000, 12);
        _hub.AddExternal(source, built);
        using var store = TokenStore();
        await store.DownloadAsync(SourceLock(source, built), cancellationToken: Ct);

        Assert.Null(_hub.Requests.Single().Headers.Authorization);
    }

    [Fact]
    public async Task TokenIsNotSentToAnHttpEndpoint()
    {
        // An http:// endpoint (a local mirror) matches the Hub host rule but would carry the token in cleartext.
        var plainHub = new Uri("http://hub.test/");
        using var store = new ModelAssetStore(
            new ModelAcquisitionOptions { CacheDirectory = _cache.Root, Endpoint = plainHub, HuggingFaceToken = HubToken, RetryDelay = TimeSpan.Zero, UseEnvironmentToken = false },
            new HttpClient(_hub, disposeHandler: false));
        var snapshot = await store.DownloadAsync(_hub.LockFor(Tokenizer), cancellationToken: Ct);

        Assert.Equal(_tokenizer, File.ReadAllBytes(snapshot.GetFilePath(Tokenizer)));
        var request = _hub.Requests.Single();
        Assert.Equal(Uri.UriSchemeHttp, request.RequestUri!.Scheme);
        Assert.Null(request.Headers.Authorization);
        Assert.False(store.IsTokenHost(new ModelAssetFile(Tokenizer, 1, new string('a', 64)), request.RequestUri));
    }

    [Fact]
    public async Task TokenIsNotForwardedOnACrossHostRedirect()
    {
        // The Hub answers resolve with 302 to a signed storage/CDN address; that hop must not carry the token.
        var cdn = new Uri("https://cas-bridge.xethub.hf.co/xet-bridge-us/abc/built.onnx?X-Amz-Signature=sig");
        byte[] built = FakeHub.RandomBytes(40_000, 13);
        _hub.AddExternal(cdn, built);
        _hub.Override = (_, r) => r.RequestUri == HubSource ? Redirect(HttpStatusCode.Found, cdn) : null;
        using var store = TokenStore();
        var snapshot = await store.DownloadAsync(SourceLock(HubSource, built), cancellationToken: Ct);

        Assert.Equal(built, File.ReadAllBytes(snapshot.GetFilePath("onnx/built.onnx")));
        var requests = _hub.Requests.ToArray();
        Assert.Equal(2, requests.Length);
        Assert.Equal(HubSource, requests[0].RequestUri);
        Assert.Equal(HubToken, requests[0].Headers.Authorization!.Parameter);
        Assert.Equal(cdn, requests[1].RequestUri);
        Assert.Null(requests[1].Headers.Authorization);
    }

    [Fact]
    public async Task TokenIsKeptOnARelativeRedirectOnHuggingFaceAndDroppedOnTheNextHost()
    {
        // huggingface.co -> /api/resolve-cache/... (same host, token needed for a private repository) -> CDN (no token).
        var cache = new Uri("https://huggingface.co/api/resolve-cache/models/owner/hosting/fedcba98/onnx/built.onnx");
        var cdn = new Uri("https://cdn-lfs.hf.co/repos/ab/cd/built?Signature=sig");
        byte[] built = FakeHub.RandomBytes(8_000, 14);
        _hub.AddExternal(cdn, built);
        _hub.Override = (_, r) =>
            r.RequestUri == HubSource ? Redirect(HttpStatusCode.TemporaryRedirect, new Uri(cache.PathAndQuery, UriKind.Relative))
            : r.RequestUri == cache ? Redirect(HttpStatusCode.Found, cdn)
            : null;
        using var store = TokenStore();
        await store.DownloadAsync(SourceLock(HubSource, built), cancellationToken: Ct);

        var requests = _hub.Requests.ToArray();
        Assert.Equal([HubSource, cache, cdn], requests.Select(r => r.RequestUri));
        Assert.Equal(HubToken, requests[0].Headers.Authorization!.Parameter);
        Assert.Equal(HubToken, requests[1].Headers.Authorization!.Parameter);
        Assert.Null(requests[2].Headers.Authorization);
    }

    [Fact]
    public async Task HubEndpointRedirectToItsStorageHostDropsTheToken()
    {
        var storage = new Uri("https://storage.test/blobs/model?sig=1");
        _hub.AddExternal(storage, _model);
        _hub.Override = (_, r) => r.RequestUri!.Host == FakeHub.Endpoint.Host ? Redirect(HttpStatusCode.Found, storage) : null;
        using var store = TokenStore();
        await store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct);

        var requests = _hub.Requests.ToArray();
        Assert.Equal(HubToken, requests[0].Headers.Authorization!.Parameter);
        Assert.Equal(storage, requests[1].RequestUri);
        Assert.Null(requests[1].Headers.Authorization);
    }

    [Fact]
    public async Task RedirectFromHttpsToHttpIsRefused()
    {
        var plain = new Uri("http://cdn.test/built.onnx");
        byte[] built = FakeHub.RandomBytes(1_000, 15);
        _hub.AddExternal(plain, built);
        _hub.Override = (_, r) => r.RequestUri == HubSource ? Redirect(HttpStatusCode.Found, plain) : null;
        using var store = TokenStore();
        var ex = await Assert.ThrowsAsync<ModelAssetDownloadException>(() => store.DownloadAsync(SourceLock(HubSource, built), cancellationToken: Ct));
        Assert.Contains("http", ex.Message, StringComparison.Ordinal);
        Assert.Equal(1, _hub.RequestCount);
    }

    [Theory]
    [InlineData(HttpStatusCode.Unauthorized, null)]
    [InlineData(HttpStatusCode.Unauthorized, HubToken)]
    [InlineData(HttpStatusCode.Forbidden, HubToken)]
    public async Task DeniedHuggingFaceSourceExplainsTheTokenAndPrivateRepository(HttpStatusCode status, string? token)
    {
        _hub.Override = (_, _) => new HttpResponseMessage(status);
        using var store = TokenStore(token);
        var ex = await Assert.ThrowsAsync<ModelAssetAccessDeniedException>(() => store.DownloadAsync(SourceLock(HubSource, [1, 2, 3]), cancellationToken: Ct));

        Assert.Equal(status, ex.StatusCode);
        Assert.Equal(token is not null, ex.TokenProvided);
        Assert.Equal("onnx/built.onnx", ex.AssetPath);
        Assert.Contains("private", ex.Message, StringComparison.Ordinal);
        Assert.Contains(HubSource.AbsoluteUri, ex.Message, StringComparison.Ordinal);
        Assert.Contains(token is null ? "No token was sent" : "A token was sent", ex.Message, StringComparison.Ordinal);
        if (token is null)
        {
            Assert.Contains("HF_TOKEN", ex.Message, StringComparison.Ordinal);
        }

        Assert.DoesNotContain(HubToken, ex.ToString(), StringComparison.Ordinal);
        Assert.Equal(1, _hub.RequestCount);
    }

    private static HttpResponseMessage Redirect(HttpStatusCode status, Uri location)
    {
        var response = new HttpResponseMessage(status);
        response.Headers.Location = location;
        return response;
    }

    private ModelAcquisitionOptions Options(CacheVerification verification = CacheVerification.Sha256, int maxAttempts = 4) => new()
    {
        CacheDirectory = _cache.Root,
        Endpoint = FakeHub.Endpoint,
        Verification = verification,
        RetryDelay = TimeSpan.Zero,
        MaxAttempts = maxAttempts,
        UseEnvironmentToken = false,
    };

    private ModelAssetStore Store(ModelAcquisitionOptions? options = null, int maxAttempts = 4) =>
        new(options ?? Options(maxAttempts: maxAttempts), new HttpClient(_hub, disposeHandler: false));

    /// <summary>Synchronous IProgress (Progress&lt;T&gt; posts asynchronously).</summary>
    private sealed class SyncProgress(Action<ModelAssetProgress>? onReport = null) : IProgress<ModelAssetProgress>
    {
        public ConcurrentQueue<ModelAssetProgress> Items { get; } = new();

        public void Report(ModelAssetProgress value)
        {
            Items.Enqueue(value);
            onReport?.Invoke(value);
        }
    }
}
