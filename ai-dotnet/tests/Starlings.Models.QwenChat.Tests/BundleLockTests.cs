using System.Text.Json.Nodes;
using Starlings.Models.QwenChat;

namespace Starlings.Models.QwenChat.Tests;

public sealed class BundleLockTests
{
    public static TheoryData<string> Profiles => [QwenChatProfile.Qwen3.Id, QwenChatProfile.Qwen25.Id];

    [Theory]
    [MemberData(nameof(Profiles))]
    public void PinsEqualTheRegistryManifest(string id)
    {
        QwenChatProfile profile = QwenChatProfile.FromId(id);
        string manifest = File.ReadAllText(Path.Combine(RepositoryRoot(), "model-pipeline", "registry", id, "manifest.yaml"));
        Assert.Contains($"id: {id}", manifest);
        Assert.Contains($"revision: \"{profile.Revision}\"", manifest);
        string[] required = ["genai_config.json", "model.onnx", "model.onnx.data", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"];
        Assert.Equal(required.Order(StringComparer.Ordinal), profile.BundleFiles.Select(f => f.Path).Order(StringComparer.Ordinal));
        // Every pinned file is a path/sha256/size entry of the manifest (origin.files; the graph files also under
        // execution.graph.assets), compared line by line without indentation.
        string entries = string.Join('\n', manifest.Split('\n').Select(line => line.Trim()));
        foreach (QwenChatBundleFile file in profile.BundleFiles)
        {
            Assert.Contains($"- path: {file.Path}\nsha256: \"{file.Sha256}\"\nsize: {file.Size}\n", entries, StringComparison.Ordinal);
        }

        Assert.Equal(profile.TokenizerSha256, profile.BundleFiles.Single(f => f.Path == "tokenizer.json").Sha256);
        Assert.Equal(profile.TemplateSha256, profile.BundleFiles.Single(f => f.Path == "chat_template.jinja").Sha256);
    }

    [Fact]
    public void RejectsALockThatDiffersFromThePin()
    {
        QwenChatProfile profile = QwenChatProfile.Qwen3;
        using var bundle = new TempBundle(profile, files => files[0]!["sha256"] = new string('0', 64));
        var error = Assert.Throws<InvalidDataException>(() => QwenChatModel.Load(profile, bundle.Path));
        Assert.Contains("pinned digest", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RejectsAFileOutsideThePin()
    {
        QwenChatProfile profile = QwenChatProfile.Qwen25;
        using var bundle = new TempBundle(profile, files => files[0]!["path"] = "extra.bin");
        var error = Assert.Throws<InvalidDataException>(() => QwenChatModel.Load(profile, bundle.Path));
        Assert.Contains("not in the profile's lock", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void VerifiesEveryPinnedFileWithoutABuildLock()
    {
        // A downloaded bundle has no bundle-lock.json; the files themselves must still equal the pin.
        string directory = Directory.CreateTempSubdirectory("qwenchat-nolock-").FullName;
        try
        {
            File.WriteAllText(Path.Combine(directory, "genai_config.json"), "{}");
            var error = Assert.Throws<InvalidDataException>(() => QwenChatModel.Load(QwenChatProfile.Qwen3, directory));
            Assert.Contains("Bundle asset mismatch", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    internal static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "registry")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException("Repository root not found above the test output.");
    }

    /// <summary>A directory holding only a bundle-lock.json built from the profile's pins, then altered.</summary>
    private sealed class TempBundle : IDisposable
    {
        public TempBundle(QwenChatProfile profile, Action<JsonArray> alter)
        {
            Path = Directory.CreateTempSubdirectory("qwenchat-lock-").FullName;
            var files = new JsonArray([.. profile.BundleFiles.Select(f => (JsonNode)new JsonObject { ["path"] = f.Path, ["sha256"] = f.Sha256, ["size"] = f.Size })]);
            alter(files);
            var lockJson = new JsonObject
            {
                ["format_version"] = 1,
                ["profile"] = profile.Id,
                ["origin"] = new JsonObject { ["repo"] = profile.Repository, ["revision"] = profile.Revision },
                ["builder"] = new JsonObject { ["version"] = "0.17.1", ["source_commit"] = "83de55a1f3886cddbb32a5517e26adb3f9d65159" },
                ["precision"] = "fp32",
                ["provider"] = "CPUExecutionProvider",
                ["files"] = files,
            };
            File.WriteAllText(System.IO.Path.Combine(Path, "bundle-lock.json"), lockJson.ToJsonString());
        }

        public string Path { get; }

        public void Dispose() => Directory.Delete(Path, recursive: true);
    }
}
