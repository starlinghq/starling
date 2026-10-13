namespace Starlings.Acquisition.Tests;

/// <summary>
/// Real downloads from huggingface.co. Skipped unless <c>STARLING_NETWORK_TESTS=1</c>, so CI and offline runs
/// never touch the network.
/// </summary>
public sealed class HubIntegrationTests
{
    public const string EnableVariable = "STARLING_NETWORK_TESTS";

    // tokenizer.json of intfloat/multilingual-e5-small at the pinned commit (registry manifest origin.files).
    private static readonly ModelAssetLock E5Tokenizer = new(
        "multilingual-e5-small-tokenizer",
        "intfloat/multilingual-e5-small",
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        [new ModelAssetFile("tokenizer.json", 17082730, "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39")]);

    private static void RequireNetwork() =>
        Assert.SkipUnless(Environment.GetEnvironmentVariable(EnableVariable) == "1", $"Set {EnableVariable}=1 to run real Hub downloads.");

    [Fact]
    public async Task RealDownloadThenOfflineReuse()
    {
        RequireNetwork();
        using var cache = new TempCache();
        var options = new ModelAcquisitionOptions { CacheDirectory = cache.Root, UseEnvironmentToken = false };
        var started = DateTime.UtcNow;
        using (var store = new ModelAssetStore(options))
        {
            var snapshot = await store.DownloadAsync(E5Tokenizer, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(17082730, new FileInfo(snapshot.GetFilePath("tokenizer.json")).Length);
        }

        TestContext.Current.SendDiagnosticMessage($"tokenizer.json downloaded and verified in {(DateTime.UtcNow - started).TotalSeconds:F1} s");

        using var offline = new ModelAssetStore(options, new HttpClient(new NoNetwork()));
        var local = offline.OpenLocal(E5Tokenizer);
        Assert.True(File.Exists(local.GetFilePath("tokenizer.json")));
    }

    [Fact]
    public async Task RealMissingRevisionIsNotFoundOrDenied()
    {
        RequireNetwork();
        using var cache = new TempCache();
        var bogus = new ModelAssetLock("bogus", "intfloat/multilingual-e5-small", new string('0', 40),
            [new ModelAssetFile("tokenizer.json", 1, new string('a', 64))]);
        using var store = new ModelAssetStore(new ModelAcquisitionOptions { CacheDirectory = cache.Root, UseEnvironmentToken = false });
        var ex = await Assert.ThrowsAnyAsync<ModelAcquisitionException>(() => store.DownloadAsync(bogus, cancellationToken: TestContext.Current.CancellationToken));
        Assert.True(ex is ModelAssetNotFoundException or ModelAssetAccessDeniedException, ex.ToString());
        TestContext.Current.SendDiagnosticMessage($"{ex.GetType().Name}: {ex.Message}");
    }

    [Fact]
    public async Task RealGatedRepositoryWithoutTokenIsAccessDenied()
    {
        RequireNetwork();
        using var cache = new TempCache();

        // meta-llama/Llama-3.2-1B is gated; without a token the Hub answers 401. The revision does not need to be real:
        // the gate is checked before the revision.
        var gated = new ModelAssetLock("gated", "meta-llama/Llama-3.2-1B", new string('0', 40),
            [new ModelAssetFile("config.json", 1, new string('a', 64))]);
        using var store = new ModelAssetStore(new ModelAcquisitionOptions { CacheDirectory = cache.Root, UseEnvironmentToken = false });
        var ex = await Assert.ThrowsAsync<ModelAssetAccessDeniedException>(() => store.DownloadAsync(gated, cancellationToken: TestContext.Current.CancellationToken));
        Assert.False(ex.TokenProvided);
        TestContext.Current.SendDiagnosticMessage($"{(int)ex.StatusCode}: {ex.Message}");
    }
}
