using Starlings.Acquisition;
using Starlings.Models.QwenChat;

namespace Starlings.Models.QwenChat.Tests;

/// <summary>
/// The hosting contract of the locks (every pinned bundle file has a source at the pinned commit of its profile's
/// repository) and, with <c>STARLING_NETWORK_TESTS=1</c>, a real anonymous download of the Qwen2.5 bundle (2.0 GB) and its
/// offline reuse.
/// </summary>
public sealed class HostingDownloadTests
{
    private const string EnableVariable = "STARLING_NETWORK_TESTS";

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Theory]
    [InlineData("qwen3-0.6b/ortgenai-fp32-cpu", "Starlinghq/qwen3-0.6b-ortgenai")]
    [InlineData("qwen2.5-0.5b-instruct/ortgenai-fp32-cpu", "Starlinghq/qwen2.5-0.5b-instruct-ortgenai")]
    public void EveryPinnedFileIsHostedAtThePinnedCommit(string id, string repository)
    {
        QwenChatProfile profile = QwenChatProfile.FromId(id);
        Assert.Equal(repository, QwenChatModelAssets.HostingRepository(profile));
        Assert.Matches("^[0-9a-f]{40}$", QwenChatModelAssets.HostingRevision(profile));
        Assert.False(QwenChatModelAssets.HostingIsPrivate);
        string baseUrl = $"https://huggingface.co/{repository}/resolve/{QwenChatModelAssets.HostingRevision(profile)}/";
        Assert.Equal(baseUrl, QwenChatModelAssets.HostingBaseUrl(profile));
        ModelAssetLock assetLock = QwenChatModelAssets.Lock(profile);
        Assert.Equal(profile.Repository, assetLock.Repository);
        Assert.Equal(profile.Revision, assetLock.Revision);
        Assert.Equal(profile.BundleFiles.Count, assetLock.Files.Count);
        foreach (QwenChatBundleFile pinned in profile.BundleFiles)
        {
            ModelAssetFile file = assetLock.GetFile(pinned.Path);
            Assert.Equal((pinned.Size, pinned.Sha256), (file.Size, file.Sha256));
            Assert.Equal(new Uri(baseUrl + pinned.Path), file.Source);
        }
    }

    [Fact]
    public async Task Qwen25DownloadsFromTheHostingRepositoryAndVerifies()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable(EnableVariable) == "1", $"Set {EnableVariable}=1 to run real downloads.");
        string cache = Directory.CreateTempSubdirectory("starling-qwenchat-hosting-").FullName;
        try
        {
            // Anonymous: the repository is public, so no token is sent even if $HF_TOKEN is set.
            var options = new ModelAcquisitionOptions { CacheDirectory = cache, UseEnvironmentToken = false };
            QwenChatProfile profile = QwenChatProfile.Qwen25;
            string directory = await QwenChatModelAssets.DownloadAsync(profile, options, cancellationToken: Ct);
            Assert.Equal(directory, QwenChatModelAssets.OpenLocal(profile, options));
            Assert.Equal(profile.BundleFiles.Select(f => f.Path).Order(StringComparer.Ordinal),
                Directory.EnumerateFiles(directory).Select(Path.GetFileName).Order(StringComparer.Ordinal));
        }
        finally
        {
            Directory.Delete(cache, recursive: true);
        }
    }
}
