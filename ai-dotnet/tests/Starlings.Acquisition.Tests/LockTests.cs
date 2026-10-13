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
    [InlineData("""{"schema_version":3,"name":"x","repository":"a/b","revision":"0123456789abcdef0123456789abcdef01234567","files":[{"path":"f","size":1,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}""")]
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
