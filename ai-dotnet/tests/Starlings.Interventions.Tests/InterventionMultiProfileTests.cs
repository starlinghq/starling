using System.Text;
using static Starlings.Interventions.Tests.TestData;

namespace Starlings.Interventions.Tests;

/// <summary>
/// Profile schema version 2 (<c>directional_addition</c>), compiler manifest schema version 2 (several profiles, one
/// strength vector), selections of several profiles, and schema 2 cache entries. Synthetic values only.
/// </summary>
public sealed class InterventionMultiProfileTests
{
    // The documented schema 2 example (docs/interventions/profile-format.md), written by the Python serializer.
    private const string CanonicalV2 = """
        {
          "schema_version": 2,
          "compiler_version": "0.1.0",
          "site_mapping_version": "tiny-mlp/fp32/1",
          "base_artifacts": [
            {
              "role": "graph",
              "path": "onnx/model.onnx",
              "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
              "size": 1234
            },
            {
              "role": "external_data",
              "path": "onnx/model.onnx_data",
              "sha256": "4444444444444444444444444444444444444444444444444444444444444444",
              "size": 5000000000
            }
          ],
          "profiles": [
            {
              "id": "synthetic-demo",
              "sha256": "2222222222222222222222222222222222222222222222222222222222222222",
              "operator": "directional_projection"
            },
            {
              "id": "synthetic-shift",
              "sha256": "5555555555555555555555555555555555555555555555555555555555555555",
              "operator": "directional_addition"
            }
          ],
          "opset": 17,
          "precision": "fp32",
          "inputs": [
            {
              "name": "intervention_strength",
              "role": "strength",
              "dtype": "float32",
              "shape": [
                2
              ]
            }
          ],
          "sites": [
            {
              "layer": 1,
              "target": "residual_post",
              "tensor": "/model/layers.1/Add_1_output_0",
              "profiles": [
                0
              ]
            },
            {
              "layer": 3,
              "target": "residual_post",
              "tensor": "/model/layers.3/Add_1_output_0",
              "profiles": [
                0,
                1
              ]
            }
          ],
          "derived_graph": {
            "path": "model.onnx",
            "sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "size": 2048
          },
          "derived_files": [
            {
              "path": "model.onnx_data",
              "sha256": "4444444444444444444444444444444444444444444444444444444444444444",
              "size": 5000000000,
              "identity": {
                "device": 66307,
                "inode": 1234567,
                "mtime_ns": 1761000000123456789
              }
            }
          ],
          "validation": {
            "onnx_checker": "passed"
          }
        }

        """;

    // The documented schema 2 key example (docs/interventions/cache.md); test_intervention_multi.py checks the same.
    private const string V2KeyText =
        "starling-intervention-cache-key 2\n" +
        "compiler_version\t0.1.0\n" +
        "site_mapping_version\tlfm3/fp32/1\n" +
        "opset\t17\n" +
        "precision\tfp32\n" +
        "profile\t0\t2222222222222222222222222222222222222222222222222222222222222222\n" +
        "profile\t1\t5555555555555555555555555555555555555555555555555555555555555555\n" +
        "base\tgraph\tonnx/model.onnx\t0000000000000000000000000000000000000000000000000000000000000000\t1234\n" +
        "base\texternal_data\tonnx/model.onnx_data\t1111111111111111111111111111111111111111111111111111111111111111\t4096\n" +
        "input\tintervention_strength\tstrength\tfloat32\t[2]\n";

    private const string V2Key = "e0f6313abf870678bfcecfaf7734e414b9d8543428204bbf1204782e2e62d37b";

    // ----------------------------------------------------------------------------------------------------------------
    // Profile schema 2

    private static InterventionProfile Addition(string id = "synthetic-shift", int schemaVersion = 2) => new(
        id, BaseSha, InterventionOperatorKind.DirectionalAddition, [Site(1, scales: [0.5f])], null, schemaVersion);

    [Fact]
    public void AdditionProfilesAreSchemaVersion2AndRoundTrip()
    {
        InterventionProfile profile = new("synthetic-shift", BaseSha, InterventionOperatorKind.DirectionalAddition, [Site(1, scales: [0.5f])]);
        Assert.Equal(2, profile.SchemaVersion);
        byte[] bytes = profile.Serialize();
        Assert.Contains("\"schema_version\": 2", Text(bytes), StringComparison.Ordinal);
        Assert.Contains("\"operator\": \"directional_addition\"", Text(bytes), StringComparison.Ordinal);
        InterventionProfile parsed = InterventionProfile.Parse(bytes);
        Assert.Equal(InterventionOperatorKind.DirectionalAddition, parsed.Operator);
        Assert.Equal(bytes, parsed.Serialize());
    }

    [Fact]
    public void Version1DocumentsStayVersion1AndAVersion2ProjectionKeepsItsVersion()
    {
        Assert.Equal(1, Profile().SchemaVersion);
        Assert.Equal(Utf8(CanonicalProfile), InterventionProfile.Parse(Utf8(CanonicalProfile)).Serialize());
        string v2 = CanonicalProfile.Replace("\"schema_version\": 1", "\"schema_version\": 2", StringComparison.Ordinal);
        InterventionProfile parsed = InterventionProfile.Parse(Utf8(v2));
        Assert.Equal(2, parsed.SchemaVersion);
        Assert.Equal(Utf8(v2), parsed.Serialize());
    }

    [Theory]
    [InlineData("\"schema_version\": 1", "\"operator\": \"directional_addition\"", "operator")]
    [InlineData("\"schema_version\": 2", "\"operator\": \"directional_scaling\"", "operator")]
    [InlineData("\"schema_version\": 2", "\"operator\": \"Directional_Addition\"", "operator")]
    [InlineData("\"schema_version\": 3", "\"operator\": \"directional_projection\"", "schema_version")]
    public void ProfileOperatorAndVersionRules(string version, string op, string path)
    {
        string text = CanonicalProfile
            .Replace("\"schema_version\": 1", version, StringComparison.Ordinal)
            .Replace("\"operator\": \"directional_projection\"", op, StringComparison.Ordinal);
        InterventionContractException e = Assert.Throws<InterventionContractException>(() => InterventionProfile.Parse(Utf8(text)));
        Assert.Equal(path, e.JsonPath);
    }

    [Fact]
    public void Version1CannotCarryTheAdditionOperator()
    {
        InterventionContractException e = Assert.Throws<InterventionContractException>(() => Addition(schemaVersion: 1));
        Assert.Equal("operator", e.JsonPath);
        Assert.Throws<InterventionContractException>(() => Addition(schemaVersion: 3));
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Manifest schema 2

    private static InterventionManifest V2(
        IEnumerable<InterventionManifestProfile>? profiles = null,
        IEnumerable<InterventionSiteBinding>? sites = null,
        long[]? shape = null,
        string inputName = "intervention_strength") => new(
        "0.1.0",
        "lfm3/fp32/1",
        [
            new(InterventionArtifactRole.ExternalData, new("onnx/model.onnx_data", new string('1', 64), 4096)),
            new(InterventionArtifactRole.Graph, new("onnx/model.onnx", new string('0', 64), 1234)),
        ],
        profiles ??
        [
            new("a", new string('2', 64), InterventionOperatorKind.DirectionalProjection),
            new("b", new string('5', 64), InterventionOperatorKind.DirectionalAddition),
        ],
        17,
        "fp32",
        [new(inputName, InterventionInputRole.Strength, InterventionDType.Fp32, shape ?? [2])],
        sites ??
        [
            new(2, InterventionTarget.ResidualPost, "t2", [0, 1]),
            new(1, InterventionTarget.ResidualPost, "t1", [0]),
        ],
        new("model.onnx", new string('5', 64), 10),
        []);

    [Fact]
    public void ParsesAndReserializesTheDocumentedExampleByteForByte()
    {
        InterventionManifest manifest = InterventionManifest.Parse(Utf8(CanonicalV2));
        Assert.Equal(2, manifest.SchemaVersion);
        Assert.Equal(["synthetic-demo", "synthetic-shift"], manifest.ProfileIds);
        Assert.Equal(InterventionOperatorKind.DirectionalAddition, manifest.Profiles[1].Operator);
        Assert.Equal([2L], manifest.StrengthInput.Shape);
        Assert.Equal([0, 1], manifest.Sites[1].Profiles);
        Assert.Equal(CanonicalV2, Text(manifest.Serialize()));
        Assert.Throws<InvalidOperationException>(() => manifest.ProfileId);
        Assert.Throws<InvalidOperationException>(() => manifest.ProfileSha256);
        Assert.Throws<InvalidOperationException>(() => manifest.Operator);
    }

    public static TheoryData<string, string, string> Replaced => new()
    {
        { "\"shape\": [\n        2\n      ]", "\"shape\": [\n        1\n      ]", "inputs" },
        { "\"shape\": [\n        2\n      ]", "\"shape\": []", "inputs" },
        { "\"id\": \"synthetic-shift\"", "\"id\": \"synthetic-demo\"", "profiles" },
        { "\"operator\": \"directional_addition\"", "\"operator\": \"directional_scaling\"", "profiles[1].operator" },
        { "\"profiles\": [\n        0,\n        1\n      ]", "\"profiles\": [\n        1,\n        0\n      ]", "sites[1].profiles" },
        { "\"profiles\": [\n        0,\n        1\n      ]", "\"profiles\": [\n        0,\n        0\n      ]", "sites[1].profiles" },
        { "\"profiles\": [\n        0,\n        1\n      ]", "\"profiles\": []", "sites[1].profiles" },
        { "\"profiles\": [\n        0,\n        1\n      ]", "\"profiles\": [\n        0,\n        2\n      ]", "sites[1].profiles[1]" },
        { "\"profiles\": [\n        0,\n        1\n      ]", "\"profiles\": [\n        0\n      ]", "sites" },
        { "\"schema_version\": 2", "\"schema_version\": 3", "schema_version" },
        { "\"profiles\": [\n    {\n      \"id\": \"synthetic-demo\",\n      \"sha256\": \"2222222222222222222222222222222222222222222222222222222222222222\",\n      \"operator\": \"directional_projection\"\n    },\n    {\n      \"id\": \"synthetic-shift\",\n      \"sha256\": \"5555555555555555555555555555555555555555555555555555555555555555\",\n      \"operator\": \"directional_addition\"\n    }\n  ]", "\"profiles\": []", "profiles" },
        { "\"schema_version\": 2", "\"schema_version\": 1", "profiles" }, // a schema 1 manifest has no "profiles"
        { "\"opset\": 17", "\"profile_id\": \"a\",\n  \"opset\": 17", "profile_id" },
    };

    [Theory]
    [MemberData(nameof(Replaced))]
    public void RejectsInvalidSchema2Documents(string find, string replace, string path)
    {
        Assert.Contains(find, CanonicalV2, StringComparison.Ordinal);
        string text = CanonicalV2.Replace(find, replace, StringComparison.Ordinal);
        InterventionContractException e = Assert.Throws<InterventionContractException>(() => InterventionManifest.Parse(Utf8(text)));
        Assert.Equal(path, e.JsonPath);
    }

    [Fact]
    public void ConstructorRulesMatchTheParser()
    {
        Assert.Equal(2, V2().SchemaVersion);
        Assert.Throws<InterventionContractException>(() => V2(shape: [1]));
        Assert.Throws<InterventionContractException>(() => V2(profiles: []));
        Assert.Throws<InterventionContractException>(() => V2(sites: [new(1, InterventionTarget.ResidualPost, "t1", [0])]));
        Assert.Throws<InterventionContractException>(() => V2(sites: [new(1, InterventionTarget.ResidualPost, "t1")]));
        Assert.Throws<InterventionContractException>(() => V2(sites: [new(1, InterventionTarget.ResidualPost, "t1", [0, 1, 2])]));
        Assert.Throws<InterventionContractException>(() => new InterventionSiteBinding(1, InterventionTarget.ResidualPost, "t", [1, 0]));
        Assert.Throws<InterventionContractException>(() => V2(profiles:
        [
            new("a", new string('2', 64), InterventionOperatorKind.DirectionalProjection),
            new("a", new string('5', 64), InterventionOperatorKind.DirectionalAddition),
        ]));

        // Schema 1: one projection profile, sites without profile lists.
        Assert.Throws<InterventionContractException>(() => new InterventionManifest(
            "0.1.0", "m/1", [new(InterventionArtifactRole.Graph, new("model.onnx", new string('0', 64), 1))], "p", new string('2', 64),
            InterventionOperatorKind.DirectionalAddition, 17, "fp32",
            [new("s", InterventionInputRole.Strength, InterventionDType.Fp32, [1])],
            [new(0, InterventionTarget.ResidualPost, "t")], new("d.onnx", new string('5', 64), 1), []));
        Assert.Throws<InterventionContractException>(() => new InterventionManifest(
            "0.1.0", "m/1", [new(InterventionArtifactRole.Graph, new("model.onnx", new string('0', 64), 1))], "p", new string('2', 64),
            InterventionOperatorKind.DirectionalProjection, 17, "fp32",
            [new("s", InterventionInputRole.Strength, InterventionDType.Fp32, [1])],
            [new(0, InterventionTarget.ResidualPost, "t", [0])], new("d.onnx", new string('5', 64), 1), []));
    }

    [Fact]
    public void SitesOfAndStrengthResolution()
    {
        InterventionManifest manifest = V2();
        Assert.Equal([1, 2], manifest.SitesOf(0).Select(s => s.Layer));
        Assert.Equal([2], manifest.SitesOf(1).Select(s => s.Layer));
        Assert.Equal([0f, 0f], manifest.ResolveStrengths(null));
        Assert.Equal([0.5f, 0f], manifest.ResolveStrengths(new InterventionSelection("a", 0.5f)));
        Assert.Equal([0f, -2f], manifest.ResolveStrengths(new InterventionSelection("b", -2f)));
        Assert.Equal([1f, 3f], manifest.ResolveStrengths(new InterventionSelection([new("b", 3f), new("a", 1f)])));
        Assert.Throws<InterventionMismatchException>(() => manifest.ResolveStrengths(new InterventionSelection("c", 1f)));
        Assert.Throws<InterventionMismatchException>(() => manifest.ResolveStrengths(new InterventionSelection([new("a", 1f), new("c", 1f)])));
    }

    [Fact]
    public void Schema2KeyTextMatchesTheDocumentedExample()
    {
        InterventionManifest manifest = V2();
        Assert.Equal(V2KeyText, InterventionCacheKey.GetText(manifest));
        Assert.Equal(V2Key, InterventionCacheKey.Compute(manifest));
        Assert.Equal(V2Key, InterventionHashes.ComputeSha256(Encoding.UTF8.GetBytes(V2KeyText)));

        // The profile order is part of the key; a schema 1 graph of the same files has another key.
        InterventionManifest swapped = V2(
            profiles:
            [
                new("b", new string('5', 64), InterventionOperatorKind.DirectionalAddition),
                new("a", new string('2', 64), InterventionOperatorKind.DirectionalProjection),
            ],
            sites: [new(2, InterventionTarget.ResidualPost, "t2", [0, 1]), new(1, InterventionTarget.ResidualPost, "t1", [1])]);
        Assert.NotEqual(V2Key, InterventionCacheKey.Compute(swapped));
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Selection

    [Fact]
    public void SelectionOfSeveralProfiles()
    {
        InterventionSelection one = new("a", 0.5f);
        Assert.Equal("a", one.ProfileId);
        Assert.Equal(0.5f, one.Strength);
        Assert.Equal(one, new InterventionSelection("a", 0.5f));
        Assert.Equal(one.GetHashCode(), new InterventionSelection([new("a", 0.5f)]).GetHashCode());

        InterventionSelection two = new([new("a", 1f), new("b", 0f)]);
        Assert.Equal(2, two.Entries.Count);
        Assert.Throws<InvalidOperationException>(() => two.ProfileId);
        Assert.Throws<InvalidOperationException>(() => two.Strength);
        Assert.Equal(0f, two.StrengthOf("b"));
        Assert.Equal(0f, two.StrengthOf("c"));
        Assert.Equal([0f, 1f, 0f], two.Resolve(["c", "a", "b"]));
        Assert.NotEqual(two, new InterventionSelection([new("b", 0f), new("a", 1f)]));

        Assert.Throws<InterventionContractException>(() => new InterventionSelection([]));
        Assert.Throws<InterventionContractException>(() => new InterventionSelection([new("a", 1f), new("a", 2f)]));
        Assert.Throws<InterventionContractException>(() => new InterventionSelection([new("a", 1f), new("B", 2f)]));
        Assert.Throws<InterventionContractException>(() => new InterventionSelection([new("a", float.NaN)]));
        Assert.Throws<InterventionMismatchException>(() => two.EnsureMatches(Profile()));
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Schema 2 cache entries

    private sealed class MultiEntry : IDisposable
    {
        public readonly TempDir Root = new();

        public MultiEntry()
        {
            BaseDir = Path.Combine(Root.Path, "bundle");
            byte[] graph = Utf8("derived graph bytes (never parsed by the reader)");
            byte[] data = Utf8("weights weights weights");
            string baseGraphSha = InterventionHashes.ComputeSha256(Utf8("base graph"));
            Root.Write("bundle/onnx/model.onnx", Utf8("base graph"));
            string baseData = Root.Write("bundle/onnx/model.onnx_data", data);
            string link = Path.Combine(Root.Path, "staging", "model.onnx_data");
            HardLink(baseData, link);
            InterventionFileIdentity identity = Internal.FileIdentityProbe.TryGet(link)
                ?? new InterventionFileIdentity(0, 0, Internal.FileIdentityProbe.ModifiedNanoseconds(link));

            // Both profiles use the same directions path; schema 2 entries keep them apart under profiles/<i>/.
            byte[] q1 = InterventionDirections.Create(2, 8, Orthonormal(2, 8, seed: 3)).Encode();
            byte[] q2 = InterventionDirections.Create(1, 8, Orthonormal(1, 8, seed: 4)).Encode();
            InterventionProfile a = new("synthetic-a", baseGraphSha, InterventionOperatorKind.DirectionalProjection,
                [Site(1, rank: 2, file: "directions/layer1.f32", sha: InterventionHashes.ComputeSha256(q1))]);
            InterventionProfile b = new("synthetic-b", baseGraphSha, InterventionOperatorKind.DirectionalAddition,
                [Site(1, file: "directions/layer1.f32", sha: InterventionHashes.ComputeSha256(q2)),
                 Site(2, file: "directions/layer1.f32", sha: InterventionHashes.ComputeSha256(q2))]);
            ProfileA = a.Serialize();
            ProfileB = b.Serialize();
            Manifest = new InterventionManifest(
                "0.1.0",
                "tiny-mlp/fp32/1",
                [
                    new(InterventionArtifactRole.Graph, new("onnx/model.onnx", baseGraphSha, 10)),
                    new(InterventionArtifactRole.ExternalData, new("onnx/model.onnx_data", InterventionHashes.ComputeSha256(data), data.Length)),
                ],
                [
                    new InterventionManifestProfile(a.Id, InterventionHashes.ComputeSha256(ProfileA), a.Operator),
                    new InterventionManifestProfile(b.Id, InterventionHashes.ComputeSha256(ProfileB), b.Operator),
                ],
                17,
                "fp32",
                [new("intervention_strength", InterventionInputRole.Strength, InterventionDType.Fp32, [2])],
                [new(1, InterventionTarget.ResidualPost, "layer1/residual_post", [0, 1]), new(2, InterventionTarget.ResidualPost, "layer2/residual_post", [1])],
                new("model.onnx", InterventionHashes.ComputeSha256(graph), graph.Length),
                [new("model.onnx_data", InterventionHashes.ComputeSha256(data), data.Length, identity)],
                new InterventionManifestValidation(InterventionCheckerResult.Passed));
            Key = InterventionCacheKey.Compute(Manifest);
            CacheRoot = Path.Combine(Root.Path, "cache");
            Dir = InterventionArtifactCache.GetEntryDirectory(CacheRoot, "tiny-mlp", Key);
            Directory.CreateDirectory(Dir);
            File.Move(link, Path.Combine(Dir, "model.onnx_data"));
            Write("model.onnx", graph);
            Write("profiles/0/profile.json", ProfileA);
            Write("profiles/0/directions/layer1.f32", q1);
            Write("profiles/1/profile.json", ProfileB);
            Write("profiles/1/directions/layer1.f32", q2);
            Write("compiler-manifest.json", Manifest.Serialize());
        }

        public string BaseDir { get; }

        public string CacheRoot { get; }

        public string Dir { get; }

        public string Key { get; }

        public byte[] ProfileA { get; }

        public byte[] ProfileB { get; }

        public InterventionManifest Manifest { get; }

        public void Write(string relative, byte[] bytes)
        {
            string full = Path.Combine(Dir, relative.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(full)!);
            File.WriteAllBytes(full, bytes);
        }

        public void Dispose() => Root.Dispose();
    }

    [Fact]
    public void OpensASchema2EntryWithItsProfilesInStrengthOrder()
    {
        using MultiEntry e = new();
        InterventionArtifact artifact = InterventionArtifactCache.Open(e.CacheRoot, "tiny-mlp", e.Key, InterventionArtifactVerification.Fast, e.BaseDir);
        Assert.Equal(["synthetic-a", "synthetic-b"], artifact.Profiles.Select(p => p.Profile.Id));
        Assert.Equal("profiles/1/profile.json", InterventionArtifact.ProfilePath(artifact.Manifest, 1));
        Assert.Throws<InvalidOperationException>(() => artifact.Profile);
        InterventionArtifact full = InterventionArtifactCache.Open(e.CacheRoot, "tiny-mlp", e.Key, InterventionArtifactVerification.Full, e.BaseDir);
        Assert.Equal(e.Key, full.CacheKey);
    }

    [Fact]
    public void Schema2EntryWithSwappedProfilesFails()
    {
        using MultiEntry e = new();
        e.Write("profiles/0/profile.json", e.ProfileB);
        e.Write("profiles/1/profile.json", e.ProfileA);
        Assert.Throws<InterventionMismatchException>(() => InterventionArtifactCache.Open(e.CacheRoot, "tiny-mlp", e.Key));
    }

    [Fact]
    public void Schema2EntryWithAMissingProfileFails()
    {
        using MultiEntry e = new();
        File.Delete(Path.Combine(e.Dir, "profiles", "1", "profile.json"));
        Assert.Throws<FileNotFoundException>(() => InterventionArtifactCache.Open(e.CacheRoot, "tiny-mlp", e.Key));
    }
}
