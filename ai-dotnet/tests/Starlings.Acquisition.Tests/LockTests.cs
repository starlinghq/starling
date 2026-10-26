namespace Starlings.Acquisition.Tests;

public sealed class LockTests
{
    private const string Sha = FakeHub.Sha;
    private static readonly string Digest = new('a', 64);

    [Theory]
    [InlineData("main")]
    [InlineData("v1.0")]
    [InlineData("614241f6")]
    [InlineData("614241F622F53C4EEFF9890BDC4F31CFECC418B3")]
    public void RejectsAnythingButAFullLowercaseCommitSha(string revision) =>
        Assert.Throws<ArgumentException>(() => new ModelAssetLock("x", "a/b", revision, [new ModelAssetFile("f", 1, Digest)]));

    [Theory]
    [InlineData("../escape")]
    [InlineData("a/../../escape")]
    [InlineData("/etc/passwd")]
    [InlineData("a\\b")]
    [InlineData("C:/x")]
    [InlineData("a//b")]
    [InlineData("./a")]
    [InlineData("")]
    public void RejectsPathsThatCouldLeaveTheSnapshot(string path) =>
        Assert.Throws<ArgumentException>(() => new ModelAssetFile(path, 1, Digest));

    [Theory]
    [InlineData("noslash")]
    [InlineData("a/b/c")]
    [InlineData("../b")]
    [InlineData("a/b c")]
    public void RejectsMalformedRepositories(string repository) =>
        Assert.Throws<ArgumentException>(() => new ModelAssetLock("x", repository, Sha, [new ModelAssetFile("f", 1, Digest)]));

    [Fact]
    public void RejectsBadDigestsDuplicatesAndEmptyLocks()
    {
        Assert.Throws<ArgumentException>(() => new ModelAssetFile("f", 1, "ABC"));
        Assert.Throws<ArgumentException>(() => new ModelAssetFile("f", 1, new string('A', 64)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModelAssetFile("f", -1, Digest));
        Assert.Throws<ArgumentException>(() => new ModelAssetLock("x", "a/b", Sha, []));
        Assert.Throws<ArgumentException>(() => new ModelAssetLock("x", "a/b", Sha, [new ModelAssetFile("f", 1, Digest), new ModelAssetFile("f", 2, Digest)]));
    }

    [Fact]
    public void JsonRoundTrips()
    {
        var original = new ModelAssetLock("tiny", "acme/tiny-model", Sha,
            [new ModelAssetFile("onnx/model.onnx", 123, Digest), new ModelAssetFile("tokenizer.json", 0, new string('0', 64))]);
        string json = original.ToJson();
        var parsed = ModelAssetLock.Parse(json);
        Assert.Equal(original.Name, parsed.Name);
        Assert.Equal(original.Repository, parsed.Repository);
        Assert.Equal(original.Revision, parsed.Revision);
        Assert.Equal(original.Files, parsed.Files);
        Assert.Equal(123, parsed.TotalSize);
        Assert.Contains("\"schema_version\": 1", json, StringComparison.Ordinal);
    }

    [Fact]
    public void ExplicitSourcesRoundTripAsSchemaVersion2()
    {
        var source = new Uri("https://github.com/acme/models/releases/download/tiny-v1/converted.onnx");
        var original = new ModelAssetLock("tiny", "acme/tiny-model", Sha,
            [new ModelAssetFile("onnx/converted.onnx", 123, Digest, source), new ModelAssetFile("tokenizer.json", 7, new string('0', 64))]);
        string json = original.ToJson();
        Assert.Contains("\"schema_version\": 2", json, StringComparison.Ordinal);
        Assert.Contains($"\"url\": \"{source.AbsoluteUri}\"", json, StringComparison.Ordinal);
        var parsed = ModelAssetLock.Parse(json);
        Assert.Equal(original.Files, parsed.Files);
        Assert.Equal(source, parsed.GetFile("onnx/converted.onnx").Source);
        Assert.Null(parsed.GetFile("tokenizer.json").Source);
    }

    [Fact]
    public void HostingAndDisplayNameRoundTripAsSchemaVersion3()
    {
        const string hostingSha = "fedcba9876543210fedcba9876543210fedcba98";
        var source = new Uri("https://github.com/acme/models/releases/download/tiny-v1/converted.onnx");
        var original = new ModelAssetLock("tiny", "acme/tiny-model", Sha,
            [new ModelAssetFile("model.onnx", 123, Digest), new ModelAssetFile("extra.onnx", 7, Digest, source)])
        {
            DisplayName = "Tiny Model 1.0",
            Hosting = new ModelAssetHosting("hosts/tiny-model-onnx", hostingSha),
        };
        string json = original.ToJson();
        Assert.Contains("\"schema_version\": 3", json, StringComparison.Ordinal);
        Assert.Contains("\"display_name\": \"Tiny Model 1.0\"", json, StringComparison.Ordinal);
        var parsed = ModelAssetLock.Parse(json);
        Assert.Equal("Tiny Model 1.0", parsed.DisplayName);
        Assert.Equal(original.Hosting, parsed.Hosting);
        Assert.Equal("hosts/tiny-model-onnx", parsed.HostingRepository);
        Assert.Equal(hostingSha, parsed.HostingRevision);
        Assert.Equal(original.Files, parsed.Files);
        Assert.Equal("acme/tiny-model", parsed.Repository);
        Assert.Equal(json, parsed.ToJson());

        // Each newer field alone needs version 3; without them the lowest version is still written.
        Assert.Contains("\"schema_version\": 3", new ModelAssetLock("t", "a/b", Sha, [new ModelAssetFile("f", 1, Digest)]) { DisplayName = "T" }.ToJson(), StringComparison.Ordinal);
        var plain = new ModelAssetLock("t", "a/b", Sha, [new ModelAssetFile("f", 1, Digest)]);
        Assert.Null(plain.Hosting);
        Assert.Equal("a/b", plain.HostingRepository);
        Assert.Equal(Sha, plain.HostingRevision);
        Assert.Contains("\"schema_version\": 1", plain.ToJson(), StringComparison.Ordinal);

        // The URL of a hosted file uses the hosting repository and commit.
        using var store = new ModelAssetStore(new ModelAcquisitionOptions { CacheDirectory = Path.GetTempPath(), Endpoint = new Uri("https://hub.test/") });
        Assert.Equal($"https://hub.test/hosts/tiny-model-onnx/resolve/{hostingSha}/model.onnx", store.BuildUri(original, original.Files[0]).AbsoluteUri);
        Assert.Equal(source, store.BuildUri(original, original.Files[1]));
    }

    [Theory]
    [InlineData("""{"schema_version":2,"name":"x","display_name":"X","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}""")]
    [InlineData("""{"schema_version":1,"name":"x","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","hosting":{"repository":"c/d","revision":"0123456789abcdef0123456789abcdef01234567"},"files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}""")]
    [InlineData("""{"schema_version":3,"name":"x","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","hosting":{"repository":"c/d","revision":"main"},"files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}""")]
    [InlineData("""{"schema_version":3,"name":"x","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","hosting":{"repository":"c"},"files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}""")]
    [InlineData("""{"schema_version":3,"name":"x","display_name":" ","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}""")]
    public void VersionThreeFieldsAreRejectedInOlderVersionsOrWhenInvalid(string json) =>
        Assert.Throws<FormatException>(() => ModelAssetLock.Parse(json));

    [Fact]
    public void VersionTwoLocksStayReadable()
    {
        const string json = """{"schema_version":2,"name":"x","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","url":"https://example.test/f"}]}""";
        var parsed = ModelAssetLock.Parse(json);
        Assert.Null(parsed.Hosting);
        Assert.Null(parsed.DisplayName);
        Assert.Contains("\"schema_version\": 2", parsed.ToJson(), StringComparison.Ordinal);
    }

    /// <summary>
    /// The interchange examples that model-pipeline's lock schema validates (model-pipeline/tests/data/asset-locks) are
    /// read and written back byte for byte, so both sides agree on versions 1 to 3.
    /// </summary>
    [Theory]
    [InlineData("v1-hub.json", 1)]
    [InlineData("v2-sources.json", 2)]
    [InlineData("v3-hosting.json", 3)]
    public void InterchangeExamplesRoundTripExactly(string name, int version)
    {
        string text = File.ReadAllText(Path.Combine(RepositoryRoot(), "model-pipeline", "tests", "data", "asset-locks", name)).ReplaceLineEndings("\n");
        var parsed = ModelAssetLock.Parse(text);
        Assert.Equal(text, parsed.ToJson().ReplaceLineEndings("\n"));
        Assert.Contains($"\"schema_version\": {version}", text, StringComparison.Ordinal);
    }

    private static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "schemas")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException("Repository root (with model-pipeline/schemas) not found above " + AppContext.BaseDirectory);
    }

    [Theory]
    [InlineData("c/d", "main")]
    [InlineData("c", "0123456789abcdef0123456789abcdef01234567")]
    public void HostingIsValidated(string repository, string revision) =>
        Assert.Throws<ArgumentException>(() => new ModelAssetHosting(repository, revision));

    [Theory]
    [InlineData("")]
    [InlineData("  ")]
    [InlineData("a\nb")]
    public void DisplayNameIsValidated(string displayName) =>
        Assert.Throws<ArgumentException>(() => new ModelAssetLock("t", "a/b", Sha, [new ModelAssetFile("f", 1, Digest)]) { DisplayName = displayName });

    [Theory]
    [InlineData("http://example.test/model.onnx")]
    [InlineData("https://user:secret@example.test/model.onnx")]
    [InlineData("https://example.test/model.onnx#fragment")]
    [InlineData("file:///tmp/model.onnx")]
    public void RejectsUnsafeExplicitSources(string url) =>
        Assert.Throws<ArgumentException>(() => new ModelAssetFile("f", 1, Digest, new Uri(url)));

    [Theory]
    [InlineData("{}")]
    [InlineData("""{"schema_version":1,"name":"x","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","url":"https://example.test/f"}]}""")]
    [InlineData("""{"schema_version":2,"name":"x","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","url":"not a url"}]}""")]
    [InlineData("""{"schema_version":4,"name":"x","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}""")]
    [InlineData("not json")]
    [InlineData("""{"schema_version":2,"name":"x","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","files":[]}""")]
    [InlineData("""{"schema_version":1,"name":"x","repository":"a/b","revision":"main","files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}""")]
    public void InvalidJsonIsAFormatException(string json) =>
        Assert.Throws<FormatException>(() => ModelAssetLock.Parse(json));

    [Fact]
    public void ResolveUrlIsPinnedToTheCommitAndEscaped()
    {
        using var store = new ModelAssetStore(new ModelAcquisitionOptions { CacheDirectory = Path.GetTempPath(), Endpoint = new Uri("https://hub.test/mirror") });
        var assetLock = new ModelAssetLock("x", "acme/tiny-model", Sha, [new ModelAssetFile("dir/a b#.json", 1, Digest)]);
        Assert.Equal(
            $"https://hub.test/mirror/acme/tiny-model/resolve/{Sha}/dir/a%20b%23.json",
            store.BuildUri(assetLock, assetLock.Files[0]).AbsoluteUri);
    }

    [Fact]
    public void SnapshotLayoutIsOwnerNameCommit()
    {
        using var cache = new TempCache();
        using var store = new ModelAssetStore(new ModelAcquisitionOptions { CacheDirectory = cache.Root });
        var assetLock = new ModelAssetLock("x", "acme/tiny-model", Sha, [new ModelAssetFile("f", 1, Digest)]);
        Assert.Equal(Path.Combine(cache.Root, "acme", "tiny-model", Sha), store.GetSnapshotDirectory(assetLock));
        Assert.Equal(cache.Root, store.CacheDirectory);
    }

    [Fact]
    public void OptionsToStringNeverShowsTheToken()
    {
        var options = new ModelAcquisitionOptions { HuggingFaceToken = "hf_secret_value" };
        Assert.DoesNotContain("hf_secret_value", options.ToString(), StringComparison.Ordinal);
        Assert.Contains("Token=(set)", options.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void DefaultCacheDirectoryIsAbsoluteAndProductScoped()
    {
        string dir = ModelAcquisitionOptions.DefaultCacheDirectory;
        Assert.True(Path.IsPathRooted(dir));
        Assert.Contains("starlings", dir, StringComparison.Ordinal);
    }
}
