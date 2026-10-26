using System.Collections.Concurrent;
using System.Net;

namespace Starlings.Acquisition.Tests;

/// <summary>
/// The acquisition surface against the in-process fake Hub: non-throwing status, per-call verification, lock totals in
/// progress, lock waits, concurrent files, fallback endpoints, local sources, hosting repositories, partial-download
/// cleanup and pruning.
/// </summary>
public sealed class SurfaceTests : IDisposable
{
    private const string Model = "onnx/model.onnx";
    private const string Tokenizer = "tokenizer.json";
    private const string OtherSha = "fedcba9876543210fedcba9876543210fedcba98";
    private static readonly Uri Mirror = new("https://mirror.test/");

    private readonly TempCache _cache = new();
    private readonly FakeHub _hub = new();
    private readonly byte[] _model = FakeHub.RandomBytes(300_000, 1);
    private readonly byte[] _tokenizer = FakeHub.RandomBytes(10_000, 2);

    public SurfaceTests()
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

    // Non-throwing status and per-call verification.

    [Fact]
    public async Task CheckReportsMissingIncompleteCompleteAndInvalidWithoutThrowing()
    {
        using var offline = Offline();
        var empty = offline.Check(Lock);
        Assert.Equal(ModelAssetState.Missing, empty.State);
        Assert.Equal([Model, Tokenizer], empty.MissingPaths);
        Assert.Equal(Lock.TotalSize, empty.MissingBytes);
        Assert.Null(empty.Snapshot);
        Assert.False(offline.TryOpenLocal(Lock, out var none));
        Assert.Null(none);

        using (var store = Store())
        {
            await store.DownloadAsync(Lock, cancellationToken: Ct);
        }

        var complete = offline.Check(Lock);
        Assert.Equal(ModelAssetState.Complete, complete.State);
        Assert.True(complete.IsComplete);
        Assert.Empty(complete.MissingPaths);
        Assert.Empty(complete.Problems);
        Assert.Equal(0, complete.MissingBytes);
        Assert.Equal(offline.GetSnapshotDirectory(Lock), complete.SnapshotDirectory);
        Assert.Equal(_model, File.ReadAllBytes(complete.Snapshot!.GetFilePath(Model)));
        Assert.True(offline.TryOpenLocal(Lock, out var opened));
        Assert.Equal(FakeHub.Sha256(_model), opened.GetVerifiedSha256(Model));

        string tokenizerPath = opened.GetFilePath(Tokenizer);
        File.Delete(tokenizerPath);
        var incomplete = offline.Check(Lock);
        Assert.Equal(ModelAssetState.Incomplete, incomplete.State);
        Assert.Equal([Tokenizer], incomplete.MissingPaths);
        Assert.Equal(_tokenizer.Length, incomplete.MissingBytes);

        // Same size, one byte flipped: a digest mismatch with Sha256, invisible with Size.
        string modelPath = opened.GetFilePath(Model);
        var bytes = File.ReadAllBytes(modelPath);
        bytes[7] ^= 0xFF;
        File.WriteAllBytes(modelPath, bytes);
        var invalid = offline.Check(Lock);
        Assert.Equal(ModelAssetState.Invalid, invalid.State);
        Assert.Equal([Tokenizer], invalid.MissingPaths);
        var problem = Assert.Single(invalid.Problems);
        Assert.Equal(Model, problem.Path);
        Assert.Equal(ModelAssetProblemKind.DigestMismatch, problem.Kind);
        Assert.Equal(modelPath, problem.FilePath);
        Assert.Equal(FakeHub.Sha256(bytes), problem.ActualSha256);
        Assert.Equal(FakeHub.Sha256(_model), problem.ExpectedSha256);
        Assert.Contains("has sha256", problem.Message, StringComparison.Ordinal);
        Assert.Equal(Lock.TotalSize, invalid.MissingBytes);
        Assert.Equal(ModelAssetState.Incomplete, offline.Check(Lock, CacheVerification.Size).State);

        File.WriteAllBytes(modelPath, bytes[..1000]);
        var truncated = Assert.Single(offline.Check(Lock, CacheVerification.Size).Problems);
        Assert.Equal(ModelAssetProblemKind.SizeMismatch, truncated.Kind);
        Assert.Equal(1000, truncated.ActualSize);
        Assert.Null(truncated.ActualSha256);
    }

    [Fact]
    public async Task CheckDirectoryIsTheNonThrowingVerifyDirectory()
    {
        string directory;
        using (var store = Store())
        {
            directory = (await store.DownloadAsync(Lock, cancellationToken: Ct)).Directory;
        }

        Assert.True(ModelAssetStore.CheckDirectory(Lock, directory).IsComplete);
        File.Delete(Path.Combine(directory, Tokenizer));
        var status = ModelAssetStore.CheckDirectory(Lock, directory, CacheVerification.Size);
        Assert.Equal(ModelAssetState.Incomplete, status.State);
        Assert.Equal(CacheVerification.Size, status.Verification);
        Assert.Equal(ModelAssetState.Missing, ModelAssetStore.CheckDirectory(Lock, Path.Combine(_cache.Root, "nowhere")).State);
    }

    [Fact]
    public async Task VerificationCanBeChosenPerCallOnOneStore()
    {
        string modelPath;
        using (var first = Store())
        {
            modelPath = (await first.DownloadAsync(Lock, cancellationToken: Ct)).GetFilePath(Model);
        }

        var bytes = File.ReadAllBytes(modelPath);
        bytes[1] ^= 0xFF;
        File.WriteAllBytes(modelPath, bytes);
        int before = _hub.RequestCount;

        using var store = Store();
        Assert.Throws<ModelAssetIntegrityException>(() => store.OpenLocal(Lock));
        var sizeOnly = store.OpenLocal(Lock, CacheVerification.Size);
        Assert.Equal(CacheVerification.Size, sizeOnly.Verification);
        Assert.Null(sizeOnly.GetVerifiedSha256(Model));
        Assert.True(store.TryOpenLocal(Lock, CacheVerification.Size, out _));
        Assert.False(store.TryOpenLocal(Lock, out _));

        // Size verification for this call keeps the same-size cached file; the store default (Sha256) replaces it.
        var kept = await store.DownloadAsync(Lock, new ModelDownloadOptions { Verification = CacheVerification.Size }, Ct);
        Assert.Equal(CacheVerification.Size, kept.Verification);
        Assert.Equal(before, _hub.RequestCount);
        Assert.Equal(bytes, File.ReadAllBytes(modelPath));

        await store.DownloadAsync(Lock, new ModelDownloadOptions(), Ct);
        Assert.Equal(_model, File.ReadAllBytes(modelPath));
        Assert.Equal(before + 1, _hub.RequestCount);
    }

    // Progress with lock totals.

    [Fact]
    public async Task ProgressCarriesLockTotalsFileIndexAndCount()
    {
        var progress = new SyncProgress();
        using var store = Store();
        await store.DownloadAsync(Lock, progress, Ct);

        var items = progress.Items.ToArray();
        Assert.All(items, p => Assert.Equal(2, p.FileCount));
        Assert.All(items, p => Assert.Equal(Lock.TotalSize, p.LockTotalBytes));
        Assert.All(items, p => Assert.Equal(p.Path == Model ? 0 : 1, p.FileIndex));
        Assert.Equal(Lock.TotalSize, items[^1].LockBytesProcessed);
        Assert.Equal(ModelAssetStage.Completed, items[^1].Stage);
        Assert.Contains(items, p => p.Path == Model && p.Stage == ModelAssetStage.Completed && p.LockBytesProcessed == _model.Length);

        // A fresh sequential download only moves forward.
        for (int i = 1; i < items.Length; i++)
        {
            Assert.True(items[i].LockBytesProcessed >= items[i - 1].LockBytesProcessed);
        }
    }

    // Lock waits.

    [Fact]
    public async Task LockWaitTimeoutStopsTheWaitAndReportsWaiting()
    {
        using var held = HoldLock(Model);
        var progress = new SyncProgress();
        using var store = Store(Options(lockWait: TimeSpan.FromMilliseconds(300)));
        var ex = await Assert.ThrowsAsync<ModelAssetLockTimeoutException>(() => store.DownloadAsync(_hub.LockFor(Model), progress, Ct));

        Assert.Equal(Model, ex.AssetPath);
        Assert.Equal(FakeHub.Repo, ex.Repository);
        Assert.EndsWith("model.onnx.lock", ex.LockFilePath, StringComparison.Ordinal);
        Assert.True(ex.Waited >= TimeSpan.FromMilliseconds(250), ex.Waited.ToString());
        Assert.Contains("LockWaitTimeout", ex.Message, StringComparison.Ordinal);
        Assert.Equal(ModelAssetStage.Waiting, Assert.Single(progress.Items).Stage);
        Assert.Equal(0, _hub.RequestCount);

        // A per-call zero timeout gives up at the first refusal.
        await Assert.ThrowsAsync<ModelAssetLockTimeoutException>(() => store.DownloadAsync(
            _hub.LockFor(Model), new ModelDownloadOptions { LockWaitTimeout = TimeSpan.Zero }, Ct));
    }

    [Fact]
    public async Task WaitingIsReportedAndTheDownloadContinuesWhenTheLockIsReleased()
    {
        var held = HoldLock(Model);
        var waiting = new TaskCompletionSource();
        var progress = new SyncProgress(p =>
        {
            if (p.Stage == ModelAssetStage.Waiting)
            {
                waiting.TrySetResult();
            }
        });
        using var store = Store();
        var download = store.DownloadAsync(Lock, progress, Ct);
        await waiting.Task.WaitAsync(TimeSpan.FromSeconds(30), Ct);
        Assert.False(download.IsCompleted);
        held.Dispose();

        var snapshot = await download;
        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        Assert.Equal(1, progress.Items.Count(p => p.Stage == ModelAssetStage.Waiting));
    }

    [Fact]
    public void NegativeLockWaitTimeoutIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModelAssetStore(Options(lockWait: TimeSpan.FromSeconds(-1))));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModelAssetStore(Options(concurrent: 0)));
    }

    // Concurrent files.

    [Fact]
    public async Task FilesDownloadConcurrentlyUpToTheLimit()
    {
        var paths = Enumerable.Range(0, 6).Select(i => $"shard-{i}.bin").ToArray();
        for (int i = 0; i < paths.Length; i++)
        {
            _hub.Add(paths[i], FakeHub.RandomBytes(50_000 + i, 10 + i));
        }

        var assetLock = _hub.LockFor(paths);
        _hub.Latency = TimeSpan.FromMilliseconds(150);
        var progress = new SyncProgress();
        using var store = Store();
        var snapshot = await store.DownloadAsync(assetLock, new ModelDownloadOptions { MaxConcurrentFiles = 3, Progress = progress }, Ct);

        Assert.Equal(3, _hub.MaxInFlight);
        Assert.Equal(paths.Length, _hub.RequestCount);
        Assert.Equal(CacheVerification.Sha256, snapshot.Verification);
        foreach (var file in assetLock.Files)
        {
            Assert.Equal(file.Sha256, FakeHub.Sha256(File.ReadAllBytes(snapshot.GetFilePath(file.Path))));
        }

        Assert.Equal(paths.Length, progress.Items.Count(p => p.Stage == ModelAssetStage.Completed));
        Assert.Equal(assetLock.TotalSize, progress.Items.Max(p => p.LockBytesProcessed));
        Assert.True(store.Check(assetLock).IsComplete);
    }

    [Fact]
    public async Task DefaultIsOneFileAtATime()
    {
        _hub.Latency = TimeSpan.FromMilliseconds(50);
        using var store = Store();
        await store.DownloadAsync(Lock, cancellationToken: Ct);
        Assert.Equal(1, _hub.MaxInFlight);
    }

    [Fact]
    public async Task AConcurrentFailureSurfacesItsOwnException()
    {
        var missing = new ModelAssetLock("tiny", FakeHub.Repo, FakeHub.Sha,
            [.. Lock.Files, new ModelAssetFile("absent.bin", 10, new string('a', 64))]);
        using var store = Store();
        var ex = await Assert.ThrowsAsync<ModelAssetNotFoundException>(() =>
            store.DownloadAsync(missing, new ModelDownloadOptions { MaxConcurrentFiles = 3 }, Ct));
        Assert.Equal("absent.bin", ex.AssetPath);
    }

    // Fallback endpoints.

    [Fact]
    public async Task AFailingEndpointFallsBackToTheMirrorWithoutTheToken()
    {
        _hub.MirrorHosts["mirror.test"] = true;
        _hub.Override = (_, request) => request.RequestUri!.Host == FakeHub.Endpoint.Host ? new HttpResponseMessage(HttpStatusCode.ServiceUnavailable) : null;
        using var store = Store(Options(maxAttempts: 2, fallbacks: [Mirror], token: "hf_mirror_test_token"));
        var snapshot = await store.DownloadAsync(Lock, cancellationToken: Ct);

        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        var mirrored = _hub.Requests.Where(r => r.RequestUri!.Host == "mirror.test").ToArray();
        Assert.Equal(2, mirrored.Length);
        Assert.All(mirrored, r => Assert.Null(r.Headers.Authorization));
        Assert.All(mirrored, r => Assert.Contains($"/{FakeHub.Repo}/resolve/{FakeHub.Sha}/", r.RequestUri!.AbsolutePath, StringComparison.Ordinal));
        Assert.Equal(4, _hub.Requests.Count(r => r.RequestUri!.Host == FakeHub.Endpoint.Host));
    }

    [Fact]
    public async Task NotFoundAndWrongBytesAlsoFallBack()
    {
        _hub.MirrorHosts["mirror.test"] = true;
        _hub.Override = (_, request) => request.RequestUri!.Host == FakeHub.Endpoint.Host && request.RequestUri.AbsolutePath.EndsWith(Model, StringComparison.Ordinal)
            ? new HttpResponseMessage(HttpStatusCode.NotFound)
            : request.RequestUri!.Host == FakeHub.Endpoint.Host
                ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(new byte[_tokenizer.Length]) }
                : null;
        using var store = Store(Options(fallbacks: [Mirror]));
        var snapshot = await store.DownloadAsync(Lock, cancellationToken: Ct);
        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        Assert.Equal(_tokenizer, File.ReadAllBytes(snapshot.GetFilePath(Tokenizer)));
    }

    [Fact]
    public async Task WhenEveryEndpointFailsThePrimaryFailureIsThrownWithAllOfThem()
    {
        var secondMirror = new Uri("https://mirror2.test/");
        // Every address answers 404: the primary through the override, the mirrors because the fake Hub does not serve them.
        _hub.Override = (_, request) => request.RequestUri!.Host == FakeHub.Endpoint.Host ? new HttpResponseMessage(HttpStatusCode.NotFound) : null;
        using var store = Store(Options(fallbacks: [Mirror, secondMirror]));
        var ex = await Assert.ThrowsAsync<ModelAssetNotFoundException>(() => store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct));

        Assert.Contains(FakeHub.Endpoint.Host, ex.Message, StringComparison.Ordinal);
        Assert.Equal(3, ex.EndpointFailures.Count);
        Assert.Same(ex, ex.EndpointFailures[0]);
        Assert.Contains("mirror.test", ex.EndpointFailures[1].Message, StringComparison.Ordinal);
        Assert.Contains("mirror2.test", ex.EndpointFailures[2].Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ExplicitSourcesDoNotUseTheMirrors()
    {
        var source = new Uri("https://downloads.test/built.onnx");
        using var store = Store(Options(fallbacks: [Mirror]));
        var sourced = new ModelAssetLock("tiny", FakeHub.Repo, FakeHub.Sha, [new ModelAssetFile("built.onnx", 5, new string('b', 64), source)]);
        var ex = await Assert.ThrowsAsync<ModelAssetNotFoundException>(() => store.DownloadAsync(sourced, cancellationToken: Ct));
        Assert.Empty(ex.EndpointFailures);
        Assert.DoesNotContain(_hub.Requests, r => r.RequestUri!.Host == "mirror.test");
    }

    // Local sources and offline installs.

    [Fact]
    public async Task LocalSourceIsCopiedAndVerifiedWithoutTheNetwork()
    {
        string source = await SeedLocalSource();
        var before = Directory.GetFiles(source, "*", SearchOption.AllDirectories).Select(File.ReadAllBytes).ToArray();
        using var cache = new TempCache();
        using var store = new ModelAssetStore(
            new ModelAcquisitionOptions { CacheDirectory = cache.Root, LocalSources = [source], AllowNetwork = false, UseEnvironmentToken = false },
            new HttpClient(new NoNetwork()));
        var progress = new SyncProgress();
        var snapshot = await store.DownloadAsync(Lock, progress, Ct);

        Assert.StartsWith(cache.Root, snapshot.Directory, StringComparison.Ordinal);
        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        Assert.Equal(FakeHub.Sha256(_model), snapshot.GetVerifiedSha256(Model));
        Assert.Equal(Lock.TotalSize, progress.Items.Last().LockBytesProcessed);
        Assert.Equal(before, Directory.GetFiles(source, "*", SearchOption.AllDirectories).Select(File.ReadAllBytes).ToArray());
    }

    [Fact]
    public async Task ACorruptLocalCopyIsSkippedForTheNextSourceOrTheNetwork()
    {
        string source = await SeedLocalSource();
        string copy = Path.Combine(source, "acme", "tiny-model", FakeHub.Sha, "onnx", "model.onnx");
        var bytes = File.ReadAllBytes(copy);
        bytes[3] ^= 0xFF;
        File.WriteAllBytes(copy, bytes);

        using var cache = new TempCache();
        using var store = new ModelAssetStore(
            new ModelAcquisitionOptions { CacheDirectory = cache.Root, Endpoint = FakeHub.Endpoint, LocalSources = [source], RetryDelay = TimeSpan.Zero, UseEnvironmentToken = false },
            new HttpClient(_hub, disposeHandler: false));
        int before = _hub.RequestCount;
        var snapshot = await store.DownloadAsync(Lock, cancellationToken: Ct);
        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        Assert.Equal(before + 1, _hub.RequestCount);
        Assert.Equal(bytes, File.ReadAllBytes(copy));
    }

    [Fact]
    public async Task WithoutTheNetworkAMissingFileFailsBeforeAnythingIsCopied()
    {
        string source = await SeedLocalSource();
        File.Delete(Path.Combine(source, "acme", "tiny-model", FakeHub.Sha, Tokenizer));
        using var cache = new TempCache();
        using var store = new ModelAssetStore(
            new ModelAcquisitionOptions { CacheDirectory = cache.Root, LocalSources = [source], AllowNetwork = false, UseEnvironmentToken = false },
            new HttpClient(new NoNetwork()));
        var ex = await Assert.ThrowsAsync<ModelAssetMissingException>(() => store.DownloadAsync(Lock, cancellationToken: Ct));

        Assert.Equal([Tokenizer], ex.MissingPaths);
        Assert.Contains("AllowNetwork is false", ex.Message, StringComparison.Ordinal);
        Assert.Contains(source, ex.Message, StringComparison.Ordinal);
        Assert.False(Directory.Exists(store.GetSnapshotDirectory(Lock)));
    }

    // Hosting repository (lock format version 3).

    [Fact]
    public async Task HostedFilesComeFromTheHostingRepositoryAndLandUnderTheLockIdentity()
    {
        const string hostingRepo = "hosts/tiny-model-built";
        _hub.AddHosted(hostingRepo, OtherSha, Model, _model);
        var hosted = new ModelAssetLock("tiny", FakeHub.Repo, FakeHub.Sha, _hub.LockFor(Model).Files)
        {
            Hosting = new ModelAssetHosting(hostingRepo, OtherSha),
            DisplayName = "Tiny Model (built)",
        };
        using var store = Store();
        var snapshot = await store.DownloadAsync(hosted, cancellationToken: Ct);

        Assert.Equal(Path.Combine(_cache.Root, "acme", "tiny-model", FakeHub.Sha), snapshot.Directory);
        Assert.Equal(_model, File.ReadAllBytes(snapshot.GetFilePath(Model)));
        var request = Assert.Single(_hub.Requests);
        Assert.Equal($"/{hostingRepo}/resolve/{OtherSha}/{Model}", request.RequestUri!.AbsolutePath);

        var missing = new ModelAssetLock("tiny", FakeHub.Repo, FakeHub.Sha, _hub.LockFor(Tokenizer).Files) { Hosting = new ModelAssetHosting(hostingRepo, OtherSha) };
        var ex = await Assert.ThrowsAsync<ModelAssetNotFoundException>(() => store.DownloadAsync(missing, cancellationToken: Ct));
        Assert.Contains(hostingRepo, ex.Message, StringComparison.Ordinal);
        Assert.Equal(FakeHub.Repo, ex.Repository);
    }

    // Partial-download cleanup.

    [Fact]
    public async Task RemovePartialDownloadsDeletesPartialAndLockFilesButNotVerifiedFiles()
    {
        _hub.BreakBody = (2, 5_000);
        using var store = Store(maxAttempts: 1);
        await Assert.ThrowsAsync<ModelAssetDownloadException>(() => store.DownloadAsync(Lock, cancellationToken: Ct));

        string work = Path.Combine(_cache.Root, ".downloads", "acme", "tiny-model", FakeHub.Sha);
        string partial = Path.Combine(work, "tokenizer.json.partial");
        _hub.BreakBody = null;

        // The model downloaded first and is verified; the tokenizer transfer broke after 5,000 bytes.
        Assert.True(File.Exists(Path.Combine(store.GetSnapshotDirectory(Lock), "onnx", "model.onnx")));
        Assert.Equal(5_000, new FileInfo(partial).Length);

        var result = store.RemovePartialDownloads(Lock);
        Assert.Empty(result.Skipped);
        Assert.Contains(result.Removed, e => e.Kind == ModelCacheEntryKind.LockFile && e.Path.EndsWith("model.onnx.lock", StringComparison.Ordinal));
        Assert.Contains(result.Removed, e => e.Kind == ModelCacheEntryKind.LockFile && e.Path.EndsWith("tokenizer.json.lock", StringComparison.Ordinal));
        Assert.Contains(result.Removed, e => e.Kind == ModelCacheEntryKind.PartialDownload && e.Path == partial && e.Bytes == 5_000);
        Assert.All(result.Removed, e => Assert.Equal(FakeHub.Repo, e.Repository));
        Assert.Empty(Directory.GetFiles(_cache.Root, "*.partial", SearchOption.AllDirectories));
        Assert.Empty(Directory.GetFiles(_cache.Root, "*.lock", SearchOption.AllDirectories));
        Assert.False(Directory.Exists(work));
        Assert.Equal(ModelAssetState.Incomplete, store.Check(Lock).State);

        // Nothing left: a second call removes nothing, and the next download works from zero.
        Assert.Empty(store.RemovePartialDownloads(Lock).Removed);
        await store.DownloadAsync(Lock, cancellationToken: Ct);
        Assert.True(store.Check(Lock).IsComplete);
    }

    [Fact]
    public async Task RemovePartialDownloadsReportsThePartialBytes()
    {
        _hub.BreakBody = (1, 100_000);
        using var store = Store(maxAttempts: 1);
        await Assert.ThrowsAsync<ModelAssetDownloadException>(() => store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct));
        var result = store.RemovePartialDownloads(Lock);
        var partial = Assert.Single(result.Removed, e => e.Kind == ModelCacheEntryKind.PartialDownload);
        Assert.Equal(100_000, partial.Bytes);
        Assert.Equal(100_000, result.BytesRemoved);
    }

    [Fact]
    public async Task RemovePartialDownloadsSkipsAFileWhoseLockIsHeld()
    {
        _hub.BreakBody = (1, 100_000);
        using var store = Store(maxAttempts: 1);
        await Assert.ThrowsAsync<ModelAssetDownloadException>(() => store.DownloadAsync(_hub.LockFor(Model), cancellationToken: Ct));

        using (HoldLock(Model))
        {
            var result = store.RemovePartialDownloads(Lock);
            Assert.Contains(result.Skipped, e => e.Kind == ModelCacheEntryKind.PartialDownload);
            Assert.Contains(result.Skipped, e => e.Kind == ModelCacheEntryKind.LockFile);
            Assert.Single(Directory.GetFiles(_cache.Root, "*.partial", SearchOption.AllDirectories));
        }

        Assert.Single(store.RemovePartialDownloads(Lock).Removed, e => e.Kind == ModelCacheEntryKind.PartialDownload);
    }

    // Pruning.

    [Fact]
    public async Task PruneListsThenRemovesWhatNoKeptLockNeeds()
    {
        using var store = Store();
        await store.DownloadAsync(Lock, cancellationToken: Ct);
        string root = _cache.Root;
        string oldSnapshot = Write(Path.Combine(root, "acme", "old-model", OtherSha, "weights.bin"), 1234);
        string oldPartial = Write(Path.Combine(root, ".downloads", "acme", "old-model", OtherSha, "weights.bin.partial"), 100);
        string orphan = Write(Path.Combine(root, ".downloads", "acme", "tiny-model", FakeHub.Sha, "dropped.bin.partial"), 50);
        string keptPartial = Write(Path.Combine(root, ".downloads", "acme", "tiny-model", FakeHub.Sha, "tokenizer.json.partial"), 10);
        string notLayout = Write(Path.Combine(root, "notes", "readme.txt"), 5);
        string branchDir = Write(Path.Combine(root, "acme", "tiny-model", "main", "x.bin"), 5);

        var plan = store.PlanPrune([Lock]);
        Assert.Equal(root, plan.CacheDirectory);
        var snapshotEntry = Assert.Single(plan.Entries, e => e.Kind == ModelCacheEntryKind.Snapshot);
        Assert.Equal(Path.GetDirectoryName(oldSnapshot), snapshotEntry.Path);
        Assert.Equal(1234, snapshotEntry.Bytes);
        Assert.Equal("acme/old-model", snapshotEntry.Repository);
        Assert.Equal(OtherSha, snapshotEntry.Revision);
        var partials = plan.Entries.Where(e => e.Kind == ModelCacheEntryKind.PartialDownload).Select(e => e.Path).Order().ToArray();
        Assert.Equal(new[] { oldPartial, orphan }.Order().ToArray(), partials);
        Assert.Equal(2, plan.Entries.Count(e => e.Kind == ModelCacheEntryKind.LockFile)); // the two files of the kept lock
        Assert.Equal(1234 + 100 + 50, plan.TotalBytes);

        // Listing changed nothing.
        Assert.True(File.Exists(oldSnapshot) && File.Exists(oldPartial) && File.Exists(orphan));

        var result = store.Prune(plan);
        Assert.Empty(result.Skipped);
        Assert.Equal(plan.Entries.Count, result.Removed.Count);
        Assert.False(Directory.Exists(Path.Combine(root, "acme", "old-model")));
        Assert.False(Directory.Exists(Path.Combine(root, ".downloads", "acme", "old-model")));
        Assert.False(File.Exists(orphan));
        Assert.True(File.Exists(keptPartial));
        Assert.True(File.Exists(notLayout));
        Assert.True(File.Exists(branchDir));
        Assert.True(store.Check(Lock).IsComplete);
        Assert.DoesNotContain(store.PlanPrune([Lock]).Entries, e => e.Kind != ModelCacheEntryKind.LockFile);
    }

    [Fact]
    public async Task PruneSkipsASnapshotWhoseDownloadHoldsALock()
    {
        using var store = Store();
        await store.DownloadAsync(Lock, cancellationToken: Ct);
        var plan = store.PlanPrune([]);
        Assert.Equal(_model.Length + _tokenizer.Length, Assert.Single(plan.Entries, e => e.Kind == ModelCacheEntryKind.Snapshot).Bytes);

        using (HoldLock(Model))
        {
            var result = store.Prune(plan);
            Assert.Contains(result.Skipped, e => e.Kind == ModelCacheEntryKind.Snapshot);
            Assert.True(store.Check(Lock).IsComplete);
        }

        var later = store.Prune([]);
        Assert.Contains(later.Removed, e => e.Kind == ModelCacheEntryKind.Snapshot);
        Assert.Equal(ModelAssetState.Missing, store.Check(Lock).State);
        Assert.Empty(Directory.GetFiles(_cache.Root, "*", SearchOption.AllDirectories));
    }

    [Fact]
    public void APlanForAnotherCacheIsRefused()
    {
        using var other = new TempCache();
        using var first = new ModelAssetStore(new ModelAcquisitionOptions { CacheDirectory = other.Root });
        using var store = Store();
        var plan = first.PlanPrune([]);
        Assert.Throws<ArgumentException>(() => store.Prune(plan));
    }

    [Fact]
    public async Task ADownloadAfterPruneRecreatesItsWorkDirectories()
    {
        using var store = Store();
        await store.DownloadAsync(Lock, cancellationToken: Ct);
        store.Prune([]);
        await store.DownloadAsync(Lock, cancellationToken: Ct);
        Assert.True(store.Check(Lock).IsComplete);
    }

    private static string Write(string path, int length)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, new byte[length]);
        return path;
    }

    /// <summary>Holds a file's download lock as another downloader would.</summary>
    private FileStream HoldLock(string path)
    {
        string lockPath = Path.Combine(_cache.Root, ".downloads", "acme", "tiny-model", FakeHub.Sha, path.Replace('/', Path.DirectorySeparatorChar) + ".lock");
        Directory.CreateDirectory(Path.GetDirectoryName(lockPath)!);
        return new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1);
    }

    /// <summary>A directory laid out like a cache, holding the lock's files (a copy from a connected machine).</summary>
    private async Task<string> SeedLocalSource()
    {
        string source = Path.Combine(_cache.Root, "usb");
        using var seeding = new ModelAssetStore(Options(cacheDirectory: source), new HttpClient(_hub, disposeHandler: false));
        await seeding.DownloadAsync(Lock, cancellationToken: Ct);
        seeding.RemovePartialDownloads(Lock);
        return source;
    }

    private ModelAssetStore Offline() => new(Options(), new HttpClient(new NoNetwork()));

    private ModelAcquisitionOptions Options(
        int maxAttempts = 4, TimeSpan? lockWait = null, int concurrent = 1, IReadOnlyList<Uri>? fallbacks = null, string? token = null, string? cacheDirectory = null) => new()
    {
        CacheDirectory = cacheDirectory ?? _cache.Root,
        Endpoint = FakeHub.Endpoint,
        RetryDelay = TimeSpan.Zero,
        MaxAttempts = maxAttempts,
        UseEnvironmentToken = false,
        LockWaitTimeout = lockWait,
        MaxConcurrentFiles = concurrent,
        FallbackEndpoints = fallbacks ?? [],
        HuggingFaceToken = token,
    };

    private ModelAssetStore Store(ModelAcquisitionOptions? options = null, int maxAttempts = 4) =>
        new(options ?? Options(maxAttempts), new HttpClient(_hub, disposeHandler: false));

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
