using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.QwenChat;

namespace Starlings.Models.QwenChat.Tests;

/// <summary>The Hub checkpoint locks of the definition package (no model needed).</summary>
public sealed class CheckpointLockTests
{
    public static TheoryData<string, string> Checkpoints => new()
    {
        { QwenChatCheckpoint.Qwen3.Id, "qwen3" },
        { QwenChatCheckpoint.Qwen25.Id, "qwen25" },
    };

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void CheckpointLockEqualsThePackagedJsonAndTheCandidateContract(string id, string candidate)
    {
        QwenChatCheckpoint checkpoint = QwenChatCheckpoint.FromId(id);
        ModelAssetLock l = checkpoint.Lock;
        Assert.Equal($"{id}-checkpoint", l.Name);
        Assert.Equal((checkpoint.Repository, checkpoint.Revision), (l.Repository, l.Revision));
        Assert.Equal((checkpoint.Profile.Repository, checkpoint.Profile.Revision), (l.Repository, l.Revision));
        Assert.All(l.Files, f => Assert.Null(f.Source));
        Assert.Equal(
            ["config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
            l.Files.Select(f => f.Path));
        Assert.Equal(checkpoint.TokenizerSha256, l.GetFile("tokenizer.json").Sha256);

        // The packaged JSON lock reads back as an equal lock.
        Assert.Equal($"{id}.checkpoint.lock.json", checkpoint.LockFileName);
        string path = Path.Combine(TemplateCorpus.RepositoryRoot, "ai-dotnet", "src", "Starlings.Models.QwenChat", "locks", checkpoint.LockFileName);
        using (var doc = JsonDocument.Parse(File.ReadAllText(path)))
        {
            Assert.Equal(ModelAssetLock.SourcesSchemaVersion, doc.RootElement.GetProperty("schema_version").GetInt32());
        }

        ModelAssetLock file = ModelAssetLock.Load(path);
        Assert.Equal((l.Name, l.Repository, l.Revision), (file.Name, file.Repository, file.Revision));
        Assert.Equal(l.Files.Select(f => (f.Path, f.Size, f.Sha256, f.Source)), file.Files.Select(f => (f.Path, f.Size, f.Sha256, f.Source)));

        // The pipeline's candidate contract pins the same checkpoint: the lock is its origin.files without the card and license.
        using var profile = JsonDocument.Parse(File.ReadAllText(Path.Combine(TemplateCorpus.RepositoryRoot, "model-pipeline", "candidates", candidate, "profile.json")));
        JsonElement origin = profile.RootElement.GetProperty("origin");
        Assert.Equal(id, profile.RootElement.GetProperty("id").GetString()!.Split('/')[0]);
        Assert.Equal((origin.GetProperty("repo").GetString(), origin.GetProperty("revision").GetString()), (l.Repository, l.Revision));
        Assert.Equal(
            origin.GetProperty("files").EnumerateArray()
                .Select(f => (Path: f.GetProperty("path").GetString()!, Size: f.GetProperty("size").GetInt64(), Sha256: f.GetProperty("sha256").GetString()!))
                .Where(f => f.Path is not ("LICENSE" or "README.md"))
                .OrderBy(f => f.Path, StringComparer.Ordinal),
            l.Files.Select(f => (f.Path, f.Size, f.Sha256)));
    }
}
