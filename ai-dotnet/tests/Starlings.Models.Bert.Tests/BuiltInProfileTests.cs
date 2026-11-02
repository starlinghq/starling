namespace Starlings.Models.Bert.Tests;

/// <summary>The profiles embedded in the package equal the repository's candidate contracts.</summary>
public sealed class BuiltInProfileTests
{
    public static TheoryData<string> BuiltInNames => new(BertModelProfile.BuiltInIds);

    [Theory]
    [MemberData(nameof(BuiltInNames))]
    public void BuiltInEqualsLoadOfTheCandidateFile(string name)
    {
        string path = Path.Combine(RepositoryRoot(), "model-pipeline", "candidates", name, "profile.json");
        Assert.Equal(File.ReadAllText(path), BertModelProfile.BuiltInJson(name));

        var builtIn = BertModelProfile.BuiltIn(name);
        var loaded = BertModelProfile.Load(path);
        Assert.Equal(loaded.Id, builtIn.Id);
        Assert.Equal(loaded.Task, builtIn.Task);
        Assert.Equal(loaded.MaxTokens, builtIn.MaxTokens);
        Assert.Equal(loaded.Pooling, builtIn.Pooling);
        Assert.Equal(loaded.HiddenSize, builtIn.HiddenSize);
        Assert.Equal(loaded.Normalize, builtIn.Normalize);
        Assert.Equal(loaded.TokenizerProfile, builtIn.TokenizerProfile);
        Assert.Equal(loaded.Prompts, builtIn.Prompts);
        Assert.Equal(Describe(loaded.SourceLock), Describe(builtIn.SourceLock));
        Assert.Equal(Describe(loaded.RuntimeLock), Describe(builtIn.RuntimeLock));
    }

    [Fact]
    public void BuiltInIdsAreEveryAdmittedCandidateTheRouteAccepts()
    {
        // Built in = the route accepts the candidate and the registry admits it (a manifest at
        // model-pipeline/registry/<profile id>/). A draft candidate the route accepts but the registry does not
        // admit yet (paraphrase-multilingual) stays out of the package: BertModelProfile.Load still reads it.
        var accepted = new List<string>();
        foreach (string dir in Directory.EnumerateDirectories(Path.Combine(RepositoryRoot(), "model-pipeline", "candidates")))
        {
            string file = Path.Combine(dir, "profile.json");
            if (!File.Exists(file))
            {
                continue;
            }

            try
            {
                var profile = BertModelProfile.Load(file);
                if (Directory.Exists(Path.Combine(RepositoryRoot(), "model-pipeline", "registry", profile.Id)))
                {
                    accepted.Add(Path.GetFileName(dir));
                }
            }
            catch (NotSupportedException)
            {
                // Another route's candidate (Kev, Qwen, Whisper).
            }
        }

        accepted.Sort(StringComparer.Ordinal);
        Assert.Equal(accepted, BertModelProfile.BuiltInIds);
    }

    [Fact]
    public void DraftCandidateIsAcceptedButNotBuiltIn()
    {
        string path = Path.Combine(RepositoryRoot(), "model-pipeline", "candidates", "paraphrase-multilingual", "profile.json");
        var profile = BertModelProfile.Load(path);
        Assert.False(Directory.Exists(Path.Combine(RepositoryRoot(), "model-pipeline", "registry", profile.Id)));
        Assert.DoesNotContain("paraphrase-multilingual", BertModelProfile.BuiltInIds);
        Assert.Throws<ArgumentException>(() => BertModelProfile.BuiltIn("paraphrase-multilingual"));
    }

    [Theory]
    [InlineData("bge-m3", 3)]
    [InlineData("bge-reranker-v2-m3", 3)]
    [InlineData("bge-small", 2)]
    public void BuiltInRuntimeLocksNameTheInferenceFiles(string name, int files)
    {
        var profile = BertModelProfile.BuiltIn(name);
        Assert.Equal(files, profile.RuntimeLock.Files.Count);
        Assert.Equal(40, profile.RuntimeLock.Revision.Length);
    }

    [Theory]
    [InlineData("bge-small-en-v1.5/onnx-fp32-cpu")]
    [InlineData("kev-0.8b")]
    [InlineData("BGE-SMALL")]
    [InlineData("")]
    public void UnknownBuiltInNameThrows(string name) =>
        Assert.Throws<ArgumentException>(() => BertModelProfile.BuiltIn(name));

    private static string[] Describe(Starlings.Acquisition.ModelAssetLock assetLock) =>
        [$"{assetLock.Name} {assetLock.Repository} {assetLock.Revision}",
            .. assetLock.Files.Select(f => $"{f.Path} {f.Size} {f.Sha256}")];

    private static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "candidates")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException("Repository root not found above the test output.");
    }
}
