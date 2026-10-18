using Starlings.Acquisition;
using Starlings.Models.Kev;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// The hosting contract of the locks (every file has a source at the pinned commit of its size's repository) and, with
/// <c>STARLING_NETWORK_TESTS=1</c>, a real anonymous download of the Kev-0.8B bundle (3.1 GB) and its offline reuse.
/// </summary>
public sealed class HostingDownloadTests
{
    private const string EnableVariable = "STARLING_NETWORK_TESTS";

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Theory]
    [InlineData(KevModel.Kev08B, "Starlinghq/kev-0.8b-onnx")]
    [InlineData(KevModel.Kev4B, "Starlinghq/kev-4b-onnx")]
    public void EveryLockedFileIsHostedAtThePinnedCommit(KevModel model, string repository)
    {
        Assert.Equal(repository, KevModelAssets.HostingRepository(model));
        Assert.Matches("^[0-9a-f]{40}$", KevModelAssets.HostingRevision(model));
        Assert.False(KevModelAssets.HostingIsPrivate);
        string baseUrl = $"https://huggingface.co/{repository}/resolve/{KevModelAssets.HostingRevision(model)}/";
        Assert.Equal(baseUrl, KevModelAssets.HostingBaseUrl(model));
        var files = KevModelAssets.Lock(model).Files;
        Assert.Equal(5, files.Count);
        Assert.All(files, f => Assert.Equal(new Uri(baseUrl + f.Path), f.Source));
    }

    [Fact]
    public async Task Kev08BDownloadsFromTheHostingRepositoryAndVerifies()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable(EnableVariable) == "1", $"Set {EnableVariable}=1 to run real downloads.");
        string cache = Directory.CreateTempSubdirectory("starling-kev-hosting-").FullName;
        try
        {
            // Anonymous: the repository is public, so no token is sent even if $HF_TOKEN is set.
            var options = new ModelAcquisitionOptions { CacheDirectory = cache, UseEnvironmentToken = false };
            var started = DateTime.UtcNow;
            var files = await KevModelAssets.DownloadAsync(KevModel.Kev08B, options, cancellationToken: Ct);
            TestContext.Current.SendDiagnosticMessage(
                $"kev-0.8b lock ({KevModelAssets.Kev08BLock.Files.Sum(f => f.Size)} bytes) downloaded and verified in {(DateTime.UtcNow - started).TotalSeconds:F0} s");

            // Offline: the cache verifies by size and SHA-256 without the network or a token.
            var local = KevModelAssets.OpenLocal(KevModel.Kev08B, new ModelAcquisitionOptions { CacheDirectory = cache, UseEnvironmentToken = false });
            Assert.Equal(files.ModelPath, local.ModelPath);
            Assert.Equal(KevModelProfile.Kev08B, local.Profile);
        }
        finally
        {
            Directory.Delete(cache, recursive: true);
        }
    }
}
