using System.Text;
using static Starlings.Interventions.Tests.TestData;

namespace Starlings.Interventions.Tests;

/// <summary>Cache key and the read side of the derived-artifact cache, on hand-built entries with synthetic files.</summary>
public sealed class InterventionArtifactTests
{
    private const string Model = "tiny-mlp";

    // The documented example (docs/interventions/cache.md); the Python test test_intervention_cache.py uses the same.
    private const string ExampleText =
        "starling-intervention-cache-key 1\n" +
        "compiler_version\t0.1.0\n" +
        "site_mapping_version\tlfm3/fp32/1\n" +
        "opset\t17\n" +
        "precision\tfp32\n" +
        "profile_sha256\t2222222222222222222222222222222222222222222222222222222222222222\n" +
        "base\tgraph\tonnx/model.onnx\t0000000000000000000000000000000000000000000000000000000000000000\t1234\n" +
        "base\texternal_data\tonnx/model.onnx_data\t1111111111111111111111111111111111111111111111111111111111111111\t4096\n" +
        "input\tintervention_strength\tstrength\tfloat32\t[1]\n";

    private const string ExampleKey = "560773f98c03601aa3bd35112beebdbd3b91198e8dbac03f94eb1134499e6579";

    private static InterventionManifest ExampleManifest(
        string compilerVersion = "0.1.0",
        string mappingVersion = "lfm3/fp32/1",
        int opset = 17,
        string precision = "fp32",
        string profileSha = "2222222222222222222222222222222222222222222222222222222222222222",
        long dataSize = 4096,
        string inputName = "intervention_strength",
        long[]? shape = null,
        bool graphFirst = false)
    {
        InterventionBaseArtifact graph = new(InterventionArtifactRole.Graph, new("onnx/model.onnx", new string('0', 64), 1234));
        InterventionBaseArtifact data = new(InterventionArtifactRole.ExternalData, new("onnx/model.onnx_data", new string('1', 64), dataSize));
        return new(
            compilerVersion,
            mappingVersion,
            graphFirst ? [graph, data] : [data, graph],
            "p",
            profileSha,
            InterventionOperatorKind.DirectionalProjection,
            opset,
            precision,
            [new(inputName, InterventionInputRole.Strength, InterventionDType.Fp32, shape ?? [1])],
            [new(0, InterventionTarget.ResidualPost, "t")],
            new("model.onnx", new string('5', 64), 10),
            []);
    }

    [Fact]
    public void KeyTextMatchesTheDocumentedExample()
    {
        InterventionManifest manifest = ExampleManifest();
        Assert.Equal(ExampleText, InterventionCacheKey.GetText(manifest));
        Assert.Equal(ExampleKey, InterventionCacheKey.Compute(manifest));
        Assert.Equal(ExampleKey, InterventionHashes.ComputeSha256(Encoding.UTF8.GetBytes(ExampleText)));
    }

    [Fact]
    public void KeyIgnoresDocumentOrderAndCoversEveryComponent()
    {
        Assert.Equal(ExampleKey, InterventionCacheKey.Compute(ExampleManifest(graphFirst: true)));
        string[] variants =
        [
            InterventionCacheKey.Compute(ExampleManifest(compilerVersion: "0.1.1")),
            InterventionCacheKey.Compute(ExampleManifest(mappingVersion: "lfm3/fp32/2")),
            InterventionCacheKey.Compute(ExampleManifest(opset: 18)),
            InterventionCacheKey.Compute(ExampleManifest(precision: "fp16")),
            InterventionCacheKey.Compute(ExampleManifest(profileSha: new string('3', 64))),
            InterventionCacheKey.Compute(ExampleManifest(dataSize: 4097)),
            InterventionCacheKey.Compute(ExampleManifest(inputName: "alpha")),
            InterventionCacheKey.Compute(ExampleManifest(shape: [])),
        ];
        Assert.Equal(variants.Length, variants.Distinct(StringComparer.Ordinal).Count());
        Assert.DoesNotContain(ExampleKey, variants);
    }

    [Fact]
    public void KeySortsPathsByUtf8Bytes()
    {
        // U+FF5E (3 UTF-8 bytes, EF BD 9E) sorts before U+1F600 (F0 9F 98 80) in UTF-8 byte order, but after its
        // surrogate pair (D83D DE00) in UTF-16 code-unit order; the key uses the UTF-8 order, as Python does.
        InterventionManifest manifest = new(
            "0.1.0",
            "m/1",
            [
                new(InterventionArtifactRole.Graph, new("model.onnx", new string('0', 64), 1)),
                new(InterventionArtifactRole.ExternalData, new("\U0001F600.bin", new string('1', 64), 1)),
                new(InterventionArtifactRole.ExternalData, new("～.bin", new string('2', 64), 1)),
            ],
            "p",
            new string('3', 64),
            InterventionOperatorKind.DirectionalProjection,
            17,
            "fp32",
            [new("s", InterventionInputRole.Strength, InterventionDType.Fp32, [1])],
            [new(0, InterventionTarget.ResidualPost, "t")],
            new("derived.onnx", new string('5', 64), 10),
            []);
        string text = InterventionCacheKey.GetText(manifest);
        Assert.True(text.IndexOf("～.bin", StringComparison.Ordinal) < text.IndexOf("\U0001F600.bin", StringComparison.Ordinal));
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Hand-built cache entries

    private sealed class Entry : IDisposable
    {
        public readonly TempDir Root = new();

        public Entry()
        {
            BaseDir = Path.Combine(Root.Path, "bundle");
            Graph = Utf8("derived graph bytes (not a real ONNX graph; the reader never parses it)");
            Data = Utf8("weights weights weights weights");
            float[] q = Orthonormal(2, 8, seed: 3);
            Directions = InterventionDirections.Create(2, 8, q).Encode();
            InterventionSite site = new(1, InterventionTarget.ResidualPost, 8, 2, InterventionDType.Fp32,
                "directions/layer1.f32", InterventionHashes.ComputeSha256(Directions), [1f, 0.5f]);
            BaseGraphSha = InterventionHashes.ComputeSha256(Utf8("base graph"));
            Profile = new InterventionProfile("synthetic-demo", BaseGraphSha, InterventionOperatorKind.DirectionalProjection, [site]);
            ProfileBytes = Profile.Serialize();
            Root.Write("bundle/onnx/model.onnx", Utf8("base graph"));
            string baseData = Root.Write("bundle/onnx/model.onnx_data", Data);
            // A whole-tick modification time, so tests can restore it exactly with the .NET API (100 ns precision).
            File.SetLastWriteTimeUtc(baseData, DataModified);
            CacheRoot = Path.Combine(Root.Path, "cache");
            string link = Path.Combine(Root.Path, "staging", "model.onnx_data");
            HardLink(baseData, link);
            Identity = Internal.FileIdentityProbe.TryGet(link)
                ?? new InterventionFileIdentity(0, 0, Internal.FileIdentityProbe.ModifiedNanoseconds(link));
            Manifest = new InterventionManifest(
                "0.1.0",
                "tiny-mlp/fp32/1",
                [
                    new(InterventionArtifactRole.Graph, new("onnx/model.onnx", BaseGraphSha, 10)),
                    new(InterventionArtifactRole.ExternalData, new("onnx/model.onnx_data", InterventionHashes.ComputeSha256(Data), Data.Length)),
                ],
                Profile.Id,
                InterventionHashes.ComputeSha256(ProfileBytes),
                InterventionOperatorKind.DirectionalProjection,
                17,
                "fp32",
                [new("intervention_strength", InterventionInputRole.Strength, InterventionDType.Fp32, [1])],
                [new(1, InterventionTarget.ResidualPost, "layer1/residual_post")],
                new("model.onnx", InterventionHashes.ComputeSha256(Graph), Graph.Length),
                [new("model.onnx_data", InterventionHashes.ComputeSha256(Data), Data.Length, Identity)],
                new InterventionManifestValidation(InterventionCheckerResult.Passed));
            Key = InterventionCacheKey.Compute(Manifest);
            Dir = InterventionArtifactCache.GetEntryDirectory(CacheRoot, Model, Key);
            Directory.CreateDirectory(Dir);
            File.Move(link, PathOf("model.onnx_data"));
            Write("model.onnx", Graph);
            Write("profile.json", ProfileBytes);
            Write("directions/layer1.f32", Directions);
            Write("compiler-manifest.json", Manifest.Serialize());
        }

        public string BaseDir { get; }

        public string BaseGraphSha { get; }

        public string CacheRoot { get; }

        public string Dir { get; }

        public string Key { get; }

        public byte[] Graph { get; }

        public byte[] Data { get; }

        public byte[] Directions { get; }

        public byte[] ProfileBytes { get; }

        public InterventionProfile Profile { get; }

        public InterventionManifest Manifest { get; }

        public InterventionFileIdentity Identity { get; }

        public static DateTime DataModified => new(2026, 1, 2, 3, 4, 5, DateTimeKind.Utc);

        public string PathOf(string relative) => Path.Combine(Dir, relative.Replace('/', Path.DirectorySeparatorChar));

        public void Write(string relative, byte[] bytes)
        {
            string full = PathOf(relative);
            Directory.CreateDirectory(Path.GetDirectoryName(full)!);
            File.WriteAllBytes(full, bytes);
        }

        public void Dispose() => Root.Dispose();
    }

    [Fact]
    public void OpensAVerifiedEntry()
    {
        using Entry e = new();
        InterventionArtifact artifact = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key);
        Assert.Equal(e.Dir, artifact.Directory);
        Assert.Equal(e.Key, artifact.CacheKey);
        Assert.Equal(Path.Combine(e.Dir, "model.onnx"), artifact.GraphPath);
        Assert.Equal("synthetic-demo", artifact.Profile.Profile.Id);
        Assert.Equal("intervention_strength", artifact.Manifest.StrengthInput.Name);
        artifact.Manifest.VerifyBaseModel(e.BaseGraphSha);
        Assert.Throws<InterventionMismatchException>(() => artifact.Manifest.VerifyBaseModel(OtherSha));

        InterventionArtifact full = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Full);
        Assert.Equal(e.Key, full.CacheKey);
        Assert.Equal([e.Dir], InterventionArtifactCache.ListEntries(e.CacheRoot, Model));
    }

    [Fact]
    public void ListEntriesSkipsTemporaryDirectoriesAndLocks()
    {
        using Entry e = new();
        string parent = InterventionArtifactCache.GetModelDirectory(e.CacheRoot, Model);
        Directory.CreateDirectory(Path.Combine(parent, $".tmp-{e.Key}-1-abcd"));
        File.WriteAllText(Path.Combine(parent, $"{e.Key}.lock"), "");
        Directory.CreateDirectory(Path.Combine(parent, "not-a-key"));
        Assert.Equal([e.Dir], InterventionArtifactCache.ListEntries(e.CacheRoot, Model));
        Assert.Empty(InterventionArtifactCache.ListEntries(e.CacheRoot, "other-model"));
    }

    [Theory]
    [InlineData("")]
    [InlineData("Upper")]
    [InlineData("../x")]
    [InlineData("a/b")]
    [InlineData(".hidden")]
    [InlineData("tiny\n")]
    public void RejectsBadModelNames(string model)
    {
        Assert.False(InterventionArtifactCache.IsModelName(model));
        Assert.Throws<ArgumentException>(() => InterventionArtifactCache.GetModelDirectory("cache", model));
    }

    [Fact]
    public void RejectsBadKeys()
    {
        Assert.Throws<ArgumentException>(() => InterventionArtifactCache.GetEntryDirectory("cache", Model, "ABC"));
        Assert.Throws<ArgumentException>(() => InterventionArtifactCache.GetEntryDirectory("cache", Model, new string('A', 64)));
    }

    [Fact]
    public void MissingEntryIsReported()
    {
        using Entry e = new();
        Assert.Throws<DirectoryNotFoundException>(() => InterventionArtifactCache.Open(e.CacheRoot, Model, new string('a', 64)));
    }

    [Fact]
    public void DirectoryNameMustBeTheManifestKey()
    {
        using Entry e = new();
        string renamed = Path.Combine(Path.GetDirectoryName(e.Dir)!, new string('e', 64));
        Directory.Move(e.Dir, renamed);
        Assert.Throws<InterventionMismatchException>(() => InterventionArtifactCache.OpenEntry(renamed));

        // The same directory under a non-key name opens as a plain artifact, but not as a cache entry.
        string plain = Path.Combine(Path.GetDirectoryName(e.Dir)!, "derived");
        Directory.Move(renamed, plain);
        Assert.Equal(e.Key, InterventionArtifact.Open(plain).CacheKey);
        Assert.Throws<InterventionMismatchException>(() => InterventionArtifactCache.OpenEntry(plain));
    }

    public static TheoryData<string> Corruptions() =>
    [
        "flip-graph", "truncate-graph", "missing-graph", "missing-manifest", "missing-profile", "edit-profile",
        "tamper-directions", "missing-data", "resize-data", "bad-manifest",
    ];

    [Theory]
    [MemberData(nameof(Corruptions))]
    public void CorruptEntryFailsClearly(string corruption)
    {
        using Entry e = new();
        Corrupt(e, corruption);
        Exception error = Assert.ThrowsAny<Exception>(() => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key));
        Assert.True(
            error is InterventionMismatchException or InterventionContractException or FileNotFoundException,
            $"{corruption}: {error.GetType().Name}: {error.Message}");
        Assert.ThrowsAny<Exception>(() => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Full));
    }

    private static void Corrupt(Entry e, string corruption)
    {
        switch (corruption)
        {
            case "flip-graph":
                byte[] g = (byte[])e.Graph.Clone();
                g[3] ^= 1;
                e.Write("model.onnx", g);
                break;
            case "truncate-graph":
                e.Write("model.onnx", e.Graph[..^1]);
                break;
            case "missing-graph":
                File.Delete(e.PathOf("model.onnx"));
                break;
            case "missing-manifest":
                File.Delete(e.PathOf("compiler-manifest.json"));
                break;
            case "missing-profile":
                File.Delete(e.PathOf("profile.json"));
                break;
            case "edit-profile":
                e.Write("profile.json", [.. e.ProfileBytes, (byte)' ']);
                break;
            case "tamper-directions":
                byte[] d = (byte[])e.Directions.Clone();
                d[0] ^= 1;
                e.Write("directions/layer1.f32", d);
                break;
            case "missing-data":
                File.Delete(e.PathOf("model.onnx_data"));
                break;
            case "resize-data":
                e.Write("model.onnx_data", [.. e.Data, 0]);
                break;
            case "bad-manifest":
                e.Write("compiler-manifest.json", Utf8("{\"schema_version\": 2}"));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(corruption));
        }
    }

    private static InterventionManifest WithDerivedFiles(InterventionManifest m, IReadOnlyList<InterventionArtifactFile> files) =>
        new(m.CompilerVersion, m.SiteMappingVersion, m.BaseArtifacts, m.ProfileId, m.ProfileSha256, m.Operator, m.Opset,
            m.Precision, m.Inputs, m.Sites, m.DerivedGraph, files, m.Validation);

    [Fact]
    public void DerivedFilesMustBeTheBaseExternalData()
    {
        // The key covers base_artifacts, not derived_files: a manifest that records other weights for the linked file
        // (and a matching swapped file) keeps the key and the directory name, so the reader must tie the two together.
        using Entry e = new();
        byte[] other = (byte[])e.Data.Clone();
        other[0] ^= 1;
        e.Write("model.onnx_data", other);
        InterventionManifest forged = WithDerivedFiles(
            e.Manifest, [new("model.onnx_data", InterventionHashes.ComputeSha256(other), other.Length)]);
        Assert.Equal(e.Key, InterventionCacheKey.Compute(forged));
        e.Write("compiler-manifest.json", forged.Serialize());
        foreach (InterventionArtifactVerification level in Enum.GetValues<InterventionArtifactVerification>())
        {
            var error = Assert.Throws<InterventionMismatchException>(
                () => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, level, e.BaseDir));
            Assert.Contains("base artifact", error.Message, StringComparison.Ordinal);
        }

        // A derived file that links nothing in the base, and base external data without its link, are rejected too.
        e.Write("model.onnx_data", e.Data);
        InterventionArtifactFile link = e.Manifest.DerivedFiles[0];
        e.Write("extra.bin", e.Data);
        e.Write("compiler-manifest.json", WithDerivedFiles(e.Manifest, [link, new("extra.bin", link.Sha256, link.Size)]).Serialize());
        Assert.Throws<InterventionMismatchException>(() => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key));
        e.Write("compiler-manifest.json", WithDerivedFiles(e.Manifest, []).Serialize());
        Assert.Throws<InterventionMismatchException>(() => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key));
        e.Write("compiler-manifest.json", e.Manifest.Serialize());
        _ = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Full, e.BaseDir);
    }

    [Fact]
    public void SameSizeWeightsRewriteIsCaughtByTheModificationTime()
    {
        using Entry e = new();
        Assert.Equal(e.Identity, e.Manifest.DerivedFiles[0].Identity);
        _ = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Fast, e.BaseDir);
        byte[] changed = (byte[])e.Data.Clone();
        changed[^1] ^= 1;
        using (FileStream stream = new(e.PathOf("model.onnx_data"), FileMode.Open, FileAccess.Write))
        {
            stream.Write(changed); // in place: the base file shares the inode and changes too
        }

        var error = Assert.Throws<InterventionMismatchException>(() => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key));
        Assert.Contains("modified", error.Message, StringComparison.Ordinal);
        Assert.Throws<InterventionMismatchException>(
            () => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Full));

        // A rewrite that restores the modification time is the documented limit of Fast; Full still catches it.
        File.SetLastWriteTimeUtc(e.PathOf("model.onnx_data"), Entry.DataModified);
        _ = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Fast, e.BaseDir);
        Assert.Throws<InterventionMismatchException>(
            () => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Full));
    }

    [Fact]
    public void ReplacedLinkIsCaughtByFastAndPassesFull()
    {
        Assert.SkipUnless(OperatingSystem.IsLinux(), "device and inode are checked on Linux only");
        using Entry e = new();
        string link = e.PathOf("model.onnx_data");
        File.Delete(link);
        File.WriteAllBytes(link, e.Data); // same bytes and size, new inode
        File.SetLastWriteTimeUtc(link, Entry.DataModified);
        var error = Assert.Throws<InterventionMismatchException>(() => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key));
        Assert.Contains("inode", error.Message, StringComparison.Ordinal);
        _ = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Full, e.BaseDir);
    }

    [Fact]
    public void ReplacedBaseFileIsCaughtByFastWithTheBaseDirectory()
    {
        Assert.SkipUnless(OperatingSystem.IsLinux(), "device and inode are checked on Linux only");
        using Entry e = new();
        string original = Path.Combine(e.BaseDir, "onnx", "model.onnx_data");
        string replacement = original + ".new";
        File.WriteAllBytes(replacement, e.Data);
        File.Move(replacement, original, overwrite: true); // replaced by rename: the entry keeps the old inode
        _ = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key); // the entry itself is unchanged
        var error = Assert.Throws<InterventionMismatchException>(
            () => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Fast, e.BaseDir));
        Assert.Contains("no longer a hard link", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void FastRequiresARecordedIdentity()
    {
        using Entry e = new();
        InterventionArtifactFile link = e.Manifest.DerivedFiles[0];
        e.Write("compiler-manifest.json", WithDerivedFiles(e.Manifest, [new(link.Path, link.Sha256, link.Size)]).Serialize());
        var error = Assert.Throws<InterventionMismatchException>(() => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key));
        Assert.Contains("no identity", error.Message, StringComparison.Ordinal);
        _ = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Full);
    }

    [Fact]
    public void SymlinkedEntryDirectoryIsRejected()
    {
        using Entry e = new();
        string alias = Path.Combine(Path.GetDirectoryName(e.Dir)!, new string('f', 64));
        Directory.CreateSymbolicLink(alias, e.Dir);
        var error = Assert.Throws<InterventionMismatchException>(() => InterventionArtifactCache.OpenEntry(alias));
        Assert.Contains("symbolic link", error.Message, StringComparison.Ordinal);
        Assert.Equal([e.Dir], InterventionArtifactCache.ListEntries(e.CacheRoot, Model));
    }

    [Fact]
    public void ProbeReportsSharedIdentityForHardLinks()
    {
        Assert.SkipUnless(OperatingSystem.IsLinux(), "statx is Linux only");
        using TempDir dir = new();
        string a = dir.Write("a.bin", [1, 2, 3]);
        string b = dir.Write("b.bin", [1, 2, 3]);
        string c = Path.Combine(dir.Path, "c.bin");
        HardLink(a, c);
        InterventionFileIdentity ia = Internal.FileIdentityProbe.TryGet(a)!;
        Assert.Equal(ia, Internal.FileIdentityProbe.TryGet(c));
        InterventionFileIdentity ib = Internal.FileIdentityProbe.TryGet(b)!;
        Assert.Equal(ia.Device, ib.Device);
        Assert.NotEqual(ia.Inode, ib.Inode);
        // statx's nanosecond mtime agrees with .NET's 100 ns value.
        Assert.Equal(Internal.FileIdentityProbe.ModifiedNanoseconds(a) / 100, ia.ModifiedNanoseconds / 100);
        Assert.Throws<FileNotFoundException>(() => Internal.FileIdentityProbe.TryGet(Path.Combine(dir.Path, "missing")));
    }

    [Theory]
    [InlineData(8u, 1u, 2049UL)]
    [InlineData(259u, 3u, 66307UL)]
    [InlineData(0u, 52u, 52UL)]
    [InlineData(4096u, 256u, (4096UL << 32) | (256UL << 12))]
    public void DeviceNumbersUseTheGlibcEncoding(uint major, uint minor, ulong expected) =>
        Assert.Equal(expected, Internal.FileIdentityProbe.MakeDevice(major, minor));

    [Fact]
    public void FullVerificationWithBaseDirectoryHashesTheBaseArtifacts()
    {
        using Entry e = new();
        _ = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Full, e.BaseDir);
        _ = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Fast, e.BaseDir);

        // A changed base graph is caught by Full with the base directory; Fast never hashes the bundle.
        File.WriteAllText(Path.Combine(e.BaseDir, "onnx", "model.onnx"), "base grapH");
        _ = InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Fast, e.BaseDir);
        Assert.Throws<InterventionMismatchException>(
            () => InterventionArtifactCache.Open(e.CacheRoot, Model, e.Key, InterventionArtifactVerification.Full, e.BaseDir));
    }
}
