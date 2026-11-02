using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.Kev;

namespace Starlings.Models.Kev.Tests;

/// <summary>The Hub checkpoint locks of the definition package: Kev's own repository and its frozen base (no model needed).</summary>
public sealed class CheckpointLockTests
{
    [Theory]
    [InlineData(KevModel.Kev08B, "kev-0.8b", "qwen3.5-0.8b-base")]
    [InlineData(KevModel.Kev4B, "kev-4b", "qwen3.5-4b-base")]
    public void CheckpointLocksEqualThePackagedJsonAndTheCandidateContract(KevModel model, string candidate, string baseName)
    {
        KevModelProfile p = KevModelProfile.For(model);
        ModelAssetLock kev = p.CheckpointLock;
        ModelAssetLock bas = p.BaseCheckpointLock;
        Assert.Equal(($"{candidate}-checkpoint", p.Repository, p.Revision), (kev.Name, kev.Repository, kev.Revision));
        Assert.Equal(($"{baseName}-checkpoint", p.BaseRepository, p.BaseRevision), (bas.Name, bas.Repository, bas.Revision));
        Assert.Equal($"{candidate}.checkpoint.lock.json", p.CheckpointLockFileName);
        Assert.Equal($"{baseName}.checkpoint.lock.json", p.BaseCheckpointLockFileName);
        Assert.All(kev.Files.Concat(bas.Files), f => Assert.Null(f.Source));
        Assert.Equal(
            ["adapter_config.json", "adapter_model.safetensors", "head.pt", "provenance.json", "tokenizer.json", "tokenizer_config.json"],
            kev.Files.Select(f => f.Path));
        Assert.Equal(p.AdapterSha256, kev.GetFile("adapter_model.safetensors").Sha256);
        Assert.Equal(p.HeadPtSha256, kev.GetFile("head.pt").Sha256);
        Assert.Contains(bas.Files, f => f.Path.StartsWith("model.safetensors-", StringComparison.Ordinal));
        Assert.Contains(bas.Files, f => f.Path == "config.json");

        // Each packaged JSON lock reads back as an equal lock.
        foreach ((ModelAssetLock l, string fileName) in new[] { (kev, p.CheckpointLockFileName), (bas, p.BaseCheckpointLockFileName) })
        {
            string path = Path.Combine(KevTestAssets.RepositoryRoot, "ai-dotnet", "src", "Starlings.Models.Kev", "locks", fileName);
            using (var doc = JsonDocument.Parse(File.ReadAllText(path)))
            {
                Assert.Equal(ModelAssetLock.SourcesSchemaVersion, doc.RootElement.GetProperty("schema_version").GetInt32());
            }

            ModelAssetLock file = ModelAssetLock.Load(path);
            Assert.Equal((l.Name, l.Repository, l.Revision), (file.Name, file.Repository, file.Revision));
            Assert.Equal(l.Files.Select(f => (f.Path, f.Size, f.Sha256, f.Source)), file.Files.Select(f => (f.Path, f.Size, f.Sha256, f.Source)));
        }

        // The pipeline's candidate contract pins the same checkpoints: each lock is origin.files (origin.base.files)
        // without the card and license.
        using var profile = JsonDocument.Parse(File.ReadAllText(Path.Combine(KevTestAssets.RepositoryRoot, "model-pipeline", "candidates", candidate, "profile.json")));
        Assert.Equal(p.Id, profile.RootElement.GetProperty("id").GetString());
        JsonElement origin = profile.RootElement.GetProperty("origin");
        AssertMatches(origin, kev);
        AssertMatches(origin.GetProperty("base"), bas);
    }

    private static void AssertMatches(JsonElement origin, ModelAssetLock l)
    {
        Assert.Equal((origin.GetProperty("repo").GetString(), origin.GetProperty("revision").GetString()), (l.Repository, l.Revision));
        Assert.Equal(
            origin.GetProperty("files").EnumerateArray()
                .Select(f => (Path: f.GetProperty("path").GetString()!, Size: f.GetProperty("size").GetInt64(), Sha256: f.GetProperty("sha256").GetString()!))
                .Where(f => f.Path is not ("LICENSE" or "README.md"))
                .OrderBy(f => f.Path, StringComparer.Ordinal),
            l.Files.Select(f => (f.Path, f.Size, f.Sha256)));
    }
}
