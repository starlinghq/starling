using Starlings.Acquisition;
using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>The asset lock against the registry manifest, and the capability descriptor. Runs in CI.</summary>
public class AssetAndCapabilityTests
{
    private static string Manifest => File.ReadAllText(Gliner2TestAssets.ManifestPath);

    [Fact]
    public void LockPinsTheManifestFilesAndCommit()
    {
        var assetLock = Gliner2ModelAssets.Lock;
        Assert.Equal("fastino/gliner2-base-v1", assetLock.Repository);
        Assert.Equal(40, assetLock.Revision.Length);
        Assert.Contains($"revision: \"{assetLock.Revision}\"", Manifest, StringComparison.Ordinal);
        foreach (var file in assetLock.Files)
        {
            // origin.files / processing assets list path, sha256 and size of every locked file.
            Assert.Contains($"- path: {file.Path}\n      sha256: \"{file.Sha256}\"\n      size: {file.Size}", Manifest.Replace("\r\n", "\n", StringComparison.Ordinal), StringComparison.Ordinal);
        }
    }

    [Fact]
    public void TheGraphHasAnExplicitHttpsSourceAndTheTokenizerComesFromTheHub()
    {
        var graph = Gliner2ModelAssets.Lock.GetFile(Gliner2ModelAssets.ModelFile);
        Assert.Equal(Uri.UriSchemeHttps, graph.Source!.Scheme);
        Assert.Equal(Gliner2ModelAssets.ModelSource, graph.Source.AbsoluteUri);
        Assert.Null(Gliner2ModelAssets.Lock.GetFile(Gliner2ModelAssets.TokenizerFile).Source);
        string json = Gliner2ModelAssets.Lock.ToJson();
        Assert.Contains("\"schema_version\": 2", json, StringComparison.Ordinal);
        Assert.Equal(Gliner2ModelAssets.Lock.Files, ModelAssetLock.Parse(json).Files);
        Assert.Contains(Gliner2ModelAssets.ModelSource, Manifest, StringComparison.Ordinal);
    }

    [Fact]
    public void OpenDirectoryNeverDownloadsAndNamesTheMissingFiles()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"df-gliner2-empty-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var ex = Assert.Throws<ModelAssetMissingException>(() => Gliner2ModelAssets.OpenDirectory(dir));
            Assert.Equal([Gliner2ModelAssets.ModelFile, Gliner2ModelAssets.TokenizerFile], ex.MissingPaths);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void CapabilitiesMatchTheManifestAndTheContract()
    {
        var all = Gliner2Capabilities.Statuses;
        Assert.Equal(Status.Supported, all["entity_extraction.cpu"]);
        Assert.Equal(Status.Supported, all["entity_extraction.cuda"]);
        Assert.Equal(Status.Experimental, all["label_descriptions"]);
        Assert.Equal(Status.Unsupported, all["concurrent_calls.cuda.unserialized"]);
        foreach (var key in new[] { "classification", "relation_extraction", "structured_extraction", "plain_deduplicated_output", "input_over_512_tokens", "truncation", "tf32", "extensions_ai", "mlnet" })
        {
            Assert.Equal(Status.Unsupported, all[key]);
        }

    }

    [Fact]
    public void ProfileDefaults()
    {
        var options = new Gliner2ExtractorOptions();
        Assert.Equal(0.5f, options.Threshold);
        Assert.Equal(512, options.MaxSequenceLength);
        Assert.Equal(8, options.MaxBatchSize);
    }
}
