using Starlings.Acquisition;
using Starlings.Models.Qwen35;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>
/// Real downloads from the hosting repository (<see cref="Qwen35ModelAssets.HostingRepository"/>, private for now).
/// Skipped unless <c>STARLING_NETWORK_TESTS=1</c>; the authorized test also needs <c>HF_TOKEN</c> with read
/// access to the repository. The fp16x download is 1.57 GB.
/// </summary>
public sealed class HostingDownloadTests
{
    private const string EnableVariable = "STARLING_NETWORK_TESTS";

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static void RequireNetwork() =>
        Assert.SkipUnless(Environment.GetEnvironmentVariable(EnableVariable) == "1", $"Set {EnableVariable}=1 to run real downloads.");

    [Fact]
    public async Task Fp16xDownloadsFromTheHostingRepositoryAndVerifies()
    {
        RequireNetwork();
        Assert.SkipWhen(
            string.IsNullOrEmpty(Environment.GetEnvironmentVariable(ModelAcquisitionOptions.TokenEnvironmentVariable)),
            $"Set {ModelAcquisitionOptions.TokenEnvironmentVariable} to a token with read access to {Qwen35ModelAssets.HostingRepository}.");
        string cache = Directory.CreateTempSubdirectory("df-qwen35-hosting-").FullName;
        try
        {
            var options = new ModelAcquisitionOptions { CacheDirectory = cache };
            var started = DateTime.UtcNow;
            var files = await Qwen35ModelAssets.DownloadAsync(Qwen35Variant.Fp16x, options, cancellationToken: Ct);
            TestContext.Current.SendDiagnosticMessage(
                $"fp16x lock ({Qwen35ModelAssets.GetLock(Qwen35Variant.Fp16x).Files.Sum(f => f.Size)} bytes) downloaded and verified in {(DateTime.UtcNow - started).TotalSeconds:F0} s");

            // Offline: the cache verifies by size and SHA-256 without the network or a token.
            var local = Qwen35ModelAssets.OpenLocal(Qwen35Variant.Fp16x, new ModelAcquisitionOptions { CacheDirectory = cache, UseEnvironmentToken = false });
            Assert.Equal(files.ModelDirectory, local.ModelDirectory);
            Assert.Equal(Qwen35ChatTemplate.TemplateSha256, Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(File.ReadAllBytes(local.ChatTemplatePath))));
        }
        finally
        {
            Directory.Delete(cache, recursive: true);
        }
    }

    [Fact]
    public async Task WithoutATokenThePrivateRepositoryIsAccessDenied()
    {
        RequireNetwork();
        string cache = Directory.CreateTempSubdirectory("df-qwen35-hosting-").FullName;
        try
        {
            var options = new ModelAcquisitionOptions { CacheDirectory = cache, UseEnvironmentToken = false, RetryDelay = TimeSpan.Zero };
            var ex = await Assert.ThrowsAsync<ModelAssetAccessDeniedException>(() => Qwen35ModelAssets.DownloadAsync(Qwen35Variant.Fp16x, options, cancellationToken: Ct));
            Assert.False(ex.TokenProvided);
            Assert.Equal(Qwen35ModelAssets.ChatTemplateFile, ex.AssetPath);
            Assert.Contains("private", ex.Message, StringComparison.Ordinal);
            Assert.Contains(ModelAcquisitionOptions.TokenEnvironmentVariable, ex.Message, StringComparison.Ordinal);
            TestContext.Current.SendDiagnosticMessage($"{(int)ex.StatusCode}: {ex.Message}");
        }
        finally
        {
            Directory.Delete(cache, recursive: true);
        }
    }
}
