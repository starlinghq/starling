using Starlings.Models;
using static Starlings.Interventions.Tests.TestData;

namespace Starlings.Interventions.Tests;

public sealed class InterventionManifestTests
{
    private const string ProfileSha = "2222222222222222222222222222222222222222222222222222222222222222";
    private const string GraphSha = "3333333333333333333333333333333333333333333333333333333333333333";
    private const string DataSha = "4444444444444444444444444444444444444444444444444444444444444444";

    private const string Canonical = """
        {
          "schema_version": 1,
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
          "profile_id": "synthetic-demo",
          "profile_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
          "operator": "directional_projection",
          "opset": 17,
          "precision": "fp32",
          "inputs": [
            {
              "name": "intervention_strength",
              "role": "strength",
              "dtype": "float32",
              "shape": [
                1
              ]
            }
          ],
          "sites": [
            {
              "layer": 1,
              "target": "residual_post",
              "tensor": "/model/layers.1/Add_1_output_0"
            },
            {
              "layer": 3,
              "target": "residual_post",
              "tensor": "/model/layers.3/Add_1_output_0"
            }
          ],
          "derived_graph": {
            "path": "model.onnx",
            "sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "size": 2048
          },
          "derived_files": []
        }

        """;

    private static InterventionManifest Manifest(
        IEnumerable<InterventionBaseArtifact>? baseArtifacts = null,
        IEnumerable<InterventionGraphInput>? inputs = null,
        IEnumerable<InterventionSiteBinding>? sites = null,
        InterventionArtifactFile? derivedGraph = null,
        IEnumerable<InterventionArtifactFile>? derivedFiles = null,
        string profileId = "synthetic-demo",
        string profileSha = ProfileSha,
        InterventionManifestValidation? validation = null) => new(
        "0.1.0",
        "tiny-mlp/fp32/1",
        baseArtifacts ??
        [
            new(InterventionArtifactRole.Graph, new("onnx/model.onnx", BaseSha, 1234)),
            new(InterventionArtifactRole.ExternalData, new("onnx/model.onnx_data", DataSha, 5_000_000_000)),
        ],
        profileId,
        profileSha,
        InterventionOperatorKind.DirectionalProjection,
        17,
        "fp32",
        inputs ?? [Strength()],
        sites ??
        [
            new(3, InterventionTarget.ResidualPost, "/model/layers.3/Add_1_output_0"),
            new(1, InterventionTarget.ResidualPost, "/model/layers.1/Add_1_output_0"),
        ],
        derivedGraph ?? new("model.onnx", GraphSha, 2048),
        derivedFiles ?? [],
        validation);

    private static InterventionGraphInput Strength(string name = "intervention_strength", params long[] shape) =>
        new(name, InterventionInputRole.Strength, InterventionDType.Fp32, shape.Length == 0 && name == "intervention_strength" ? [1] : shape);

    [Fact]
    public void SerializesToTheDocumentedCanonicalForm()
    {
        Assert.Equal(Canonical, Text(Manifest().Serialize()));
    }

    [Fact]
    public void RoundTripsByteIdentically()
    {
        byte[] bytes = Utf8(Canonical);
        InterventionManifest manifest = InterventionManifest.Parse(bytes);
        Assert.Equal(bytes, manifest.Serialize());
        Assert.Equal(BaseSha, manifest.BaseModelSha256);
        Assert.Equal("intervention_strength", manifest.StrengthInput.Name);
        Assert.Equal([1L], manifest.StrengthInput.Shape);
        Assert.Equal(5_000_000_000, manifest.BaseArtifacts[1].File.Size);
        Assert.Equal([1, 3], manifest.Sites.Select(s => s.Layer));
    }

    public static TheoryData<string, string, string> Replaced => new()
    {
        { "\"schema_version\": 1", "\"schema_version\": 2", "schema_version" },
        { "\"opset\": 17,", "\"opset\": 17,\n  \"extra\": true,", "extra" },
        { "\"opset\": 17,", "\"opset\": 0,", "opset" },
        { "\"precision\": \"fp32\"", "\"precision\": \"FP32\"", "precision" },
        { "\"compiler_version\": \"0.1.0\"", "\"compiler_version\": \"0.1 beta\"", "compiler_version" },
        { "\"role\": \"external_data\"", "\"role\": \"graph\"", "base_artifacts" },
        { "\"role\": \"graph\"", "\"role\": \"external_data\"", "base_artifacts" },
        { "\"role\": \"graph\"", "\"role\": \"weights\"", "base_artifacts[0].role" },
        { "onnx/model.onnx_data", "onnx/model.onnx", "base_artifacts" },
        { "\"size\": 1234", "\"size\": -1", "base_artifacts[0].size" },
        { "\"size\": 1234", "\"size\": 12.5", "base_artifacts[0].size" },
        { "\"role\": \"strength\"", "\"role\": \"bias\"", "inputs[0].role" },
        { "\"shape\": [\n        1\n      ]", "\"shape\": [\n        2\n      ]", "inputs" },
        { "\"shape\": [\n        1\n      ]", "\"shape\": [\n        -1\n      ]", "inputs[0].shape" },
        { "\"dtype\": \"float32\"", "\"dtype\": \"float16\"", "inputs[0].dtype" },
        { "\"layer\": 3,", "\"layer\": 1,", "sites" },
        { "\"tensor\": \"/model/layers.1/Add_1_output_0\"", "\"tensor\": \"\"", "sites[0].tensor" },
        { "\"derived_files\": []", "\"derived_files\": [{\"path\": \"model.onnx\", \"sha256\": \"3333333333333333333333333333333333333333333333333333333333333333\", \"size\": 1}]", "derived_files" },
        { "\"derived_files\": []", "\"derived_files\": [{\"path\": \"../escape\", \"sha256\": \"3333333333333333333333333333333333333333333333333333333333333333\", \"size\": 1}]", "derived_files[0].path" },
        { "\"path\": \"model.onnx\",", "\"path\": \"model.onnx\",\n    \"role\": \"graph\",", "derived_graph.role" },
        { "\"profile_sha256\": \"2222", "\"profile_sha256\": \"222", "profile_sha256" },
        { "  \"derived_files\": []\n", "  \"derived_files\": null\n", "derived_files" },
        { "\"shape\": [\n        1\n      ]", "\"shape\": [1, 1]", "inputs" },
        { "\"shape\": [\n        1\n      ]", "\"shape\": [4611686018427387905, 4611686018427387905, 4611686018427387905, 4611686018427387905]", "inputs[0].shape" },
        { "\"shape\": [\n        1\n      ]", "\"shape\": [1e0]", "inputs[0].shape[0]" },
        { "\"opset\": 17,", "\"opset\": 1.7e1,", "opset" },
        { "\"tensor\": \"/model/layers.3/Add_1_output_0\"", "\"tensor\": \"/model/layers.1/Add_1_output_0\"", "sites" },
    };

    // The tail of the documented example (docs/interventions/profile-format.md), which the Python tests parse.
    private const string IdentityTail = """
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
            "onnx_checker": "skipped_base_rejected",
            "detail": "No Op registered for SimplifiedLayerNormalization with domain_version of 17"
          }
        }

        """;

    private static string WithIdentity => Canonical.Replace("  \"derived_files\": []\n}\n", IdentityTail, StringComparison.Ordinal);

    [Fact]
    public void IdentityAndValidationRoundTrip()
    {
        Assert.NotEqual(Canonical, WithIdentity);
        byte[] bytes = Utf8(WithIdentity);
        InterventionManifest manifest = InterventionManifest.Parse(bytes);
        Assert.Equal(bytes, manifest.Serialize());
        Assert.Equal(new InterventionFileIdentity(66307, 1234567, 1761000000123456789), manifest.DerivedFiles[0].Identity);
        Assert.Equal(InterventionCheckerResult.SkippedBaseRejected, manifest.Validation!.OnnxChecker);
        Assert.StartsWith("No Op registered", manifest.Validation.Detail, StringComparison.Ordinal);

        InterventionManifest built = Manifest(
            derivedFiles: [new("model.onnx_data", DataSha, 5_000_000_000, new InterventionFileIdentity(66307, 1234567, 1761000000123456789))],
            validation: new InterventionManifestValidation(InterventionCheckerResult.SkippedBaseRejected, manifest.Validation.Detail));
        Assert.Equal(WithIdentity, Text(built.Serialize()));
        Assert.Null(InterventionManifest.Parse(Utf8(Canonical)).Validation);
    }

    public static TheoryData<string, string, string> ReplacedWithIdentity => new()
    {
        { "\"inode\": 1234567", "\"inode\": -1", "derived_files[0].identity.inode" },
        { "\"device\": 66307", "\"device\": 1.5", "derived_files[0].identity.device" },
        { "\"mtime_ns\": 1761000000123456789", "\"ctime_ns\": 1", "derived_files[0].identity.ctime_ns" },
        { ",\n        \"mtime_ns\": 1761000000123456789", "", "derived_files[0].identity.mtime_ns" },
        { "\"onnx_checker\": \"skipped_base_rejected\"", "\"onnx_checker\": \"skipped\"", "validation.onnx_checker" },
        { "\"detail\": \"No Op", "\"detail\": \"\\nNo Op", "validation.detail" },
        { "\"detail\": \"No Op registered for SimplifiedLayerNormalization with domain_version of 17\"", "\"detail\": \"\"", "validation.detail" },
        { "\"onnx_checker\": \"skipped_base_rejected\",", "\"result\": 1,", "validation.result" },
        { "\"path\": \"model.onnx\",\n    \"sha256\"", "\"path\": \"model.onnx\",\n    \"identity\": {},\n    \"sha256\"", "derived_graph.identity" },
    };

    [Theory]
    [MemberData(nameof(ReplacedWithIdentity))]
    public void RejectsInvalidIdentityOrValidation(string find, string replace, string path)
    {
        Assert.Contains(find, WithIdentity, StringComparison.Ordinal);
        string json = WithIdentity.Replace(find, replace, StringComparison.Ordinal);
        InterventionContractException e = Assert.Throws<InterventionContractException>(() => InterventionManifest.Parse(Utf8(json)));
        Assert.Equal(path, e.JsonPath);
    }

    [Fact]
    public void OnlyDerivedFilesCarryAnIdentity()
    {
        InterventionFileIdentity identity = new(1, 2, 3);
        Assert.Throws<InterventionContractException>(() => Manifest(derivedGraph: new("model.onnx", GraphSha, 2048, identity)));
        Assert.Throws<InterventionContractException>(() => Manifest(baseArtifacts:
        [
            new(InterventionArtifactRole.Graph, new("onnx/model.onnx", BaseSha, 1234, identity)),
        ]));
        Assert.Throws<InterventionContractException>(() => new InterventionFileIdentity(-1, 0, 0));
        Assert.Throws<InterventionContractException>(() => new InterventionManifestValidation(InterventionCheckerResult.Passed, new string('x', 513)));
        Assert.Throws<InterventionContractException>(() => new InterventionManifestValidation(InterventionCheckerResult.Passed, "a\tb"));
    }

    [Theory]
    [MemberData(nameof(Replaced))]
    public void RejectsInvalidValue(string find, string replace, string path)
    {
        Assert.Contains(find, Canonical, StringComparison.Ordinal);
        string json = Canonical.Replace(find, replace, StringComparison.Ordinal);
        InterventionContractException e = Assert.Throws<InterventionContractException>(() => InterventionManifest.Parse(Utf8(json)));
        Assert.Equal(path, e.JsonPath);
    }

    [Fact]
    public void RejectsMissingStrengthAndDuplicateInputs()
    {
        Assert.Throws<InterventionContractException>(() => Manifest(inputs: []));
        Assert.Throws<InterventionContractException>(() => Manifest(inputs: [Strength(), Strength()]));
        Assert.Throws<InterventionContractException>(() => Manifest(inputs: [Strength("a", 1), Strength("b", 1)]));
        Assert.Throws<InterventionContractException>(() => Manifest(inputs: [Strength("a", 1, 2)]));
        Assert.Throws<InterventionContractException>(() => Manifest(inputs: [Strength("a", 1, 1)]));
        Assert.Throws<InterventionContractException>(() => Manifest(inputs: [Strength("a", 0)]));

        // (2^62 + 1)^4 wraps to 1 in unchecked 64-bit arithmetic; the shape must be rejected, not counted as one element.
        const long Wrap = (1L << 62) + 1;
        Assert.Throws<InterventionContractException>(() => Strength("a", Wrap, Wrap, Wrap, Wrap));
        Assert.Equal(0, Strength("a", 0, long.MaxValue).ElementCount);
        Assert.Equal(1, Manifest(inputs: [Strength("a", 1)]).StrengthInput.ElementCount);
        Assert.Equal([], Manifest(inputs: [Strength("scalar_strength")]).StrengthInput.Shape);
    }

    [Fact]
    public void VerifiesAgainstItsProfile()
    {
        InterventionProfile profile = Profile();
        InterventionManifest manifest = Manifest();
        manifest.VerifyProfile(profile, ProfileSha);
        manifest.VerifyBaseModel(BaseSha);

        Assert.Throws<InterventionMismatchException>(() => manifest.VerifyBaseModel(OtherSha));
        Assert.Throws<InterventionMismatchException>(() => manifest.VerifyProfile(profile, OtherSha));
        Assert.Throws<InterventionMismatchException>(() => Manifest(profileId: "other").VerifyProfile(profile, ProfileSha));
        Assert.Throws<InterventionMismatchException>(() => Manifest(baseArtifacts: [new(InterventionArtifactRole.Graph, new("model.onnx", OtherSha, 1))]).VerifyProfile(profile, ProfileSha));
        Assert.Throws<InterventionMismatchException>(() => Manifest(sites: [new(1, InterventionTarget.ResidualPost, "t")]).VerifyProfile(profile, ProfileSha));
        Assert.Throws<InterventionMismatchException>(() => Manifest(sites: [new(1, InterventionTarget.ResidualPost, "t"), new(2, InterventionTarget.ResidualPost, "u")]).VerifyProfile(profile, ProfileSha));
        Assert.Throws<InterventionContractException>(() => Manifest(sites: [new(1, InterventionTarget.ResidualPost, "t"), new(3, InterventionTarget.ResidualPost, "t")]));
    }

    [Fact]
    public void VerifiesDerivedAndBaseFilesOnDisk()
    {
        using TempDir derivedDir = new();
        using TempDir baseDir = new();
        byte[] graph = Utf8("derived graph bytes");
        byte[] data = Utf8("derived initializers");
        byte[] baseGraph = Utf8("base graph bytes");
        byte[] baseData = Utf8("base weights");
        derivedDir.Write("model.onnx", graph);
        derivedDir.Write("data/interventions.bin", data);
        baseDir.Write("onnx/model.onnx", baseGraph);
        string baseDataPath = baseDir.Write("onnx/model.onnx_data", baseData);

        InterventionManifest manifest = Manifest(
            baseArtifacts:
            [
                new(InterventionArtifactRole.Graph, new("onnx/model.onnx", InterventionHashes.ComputeSha256(baseGraph), baseGraph.Length)),
                new(InterventionArtifactRole.ExternalData, new("onnx/model.onnx_data", InterventionHashes.ComputeSha256(baseData), baseData.Length)),
            ],
            derivedGraph: new("model.onnx", InterventionHashes.ComputeSha256(graph), graph.Length),
            derivedFiles: [new("data/interventions.bin", InterventionHashes.ComputeSha256(data), data.Length)]);

        manifest.VerifyDerivedFiles(derivedDir.Path);
        manifest.VerifyBaseArtifacts(baseDir.Path);

        // Same size, different content.
        derivedDir.Write("data/interventions.bin", Utf8("derived initializerZ"));
        Assert.Throws<InterventionMismatchException>(() => manifest.VerifyDerivedFiles(derivedDir.Path));

        File.AppendAllText(baseDataPath, "x");
        Assert.Throws<InterventionMismatchException>(() => manifest.VerifyBaseArtifacts(baseDir.Path));

        File.Delete(Path.Combine(derivedDir.Path, "model.onnx"));
        Assert.Throws<FileNotFoundException>(() => manifest.VerifyDerivedFiles(derivedDir.Path));
    }

    [Fact]
    public void LoadsFromDisk()
    {
        using TempDir dir = new();
        string path = dir.Write("compiler-manifest.json", Utf8(Canonical));
        Assert.Equal(Utf8(Canonical), InterventionManifest.Load(path).Serialize());
    }

    [Fact]
    public void HashesUseTheContractSpelling()
    {
        Assert.Equal("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", InterventionHashes.ComputeSha256(Utf8("abc")));
        Assert.True(InterventionHashes.IsSha256(BaseSha));
        Assert.False(InterventionHashes.IsSha256(BaseSha.ToUpperInvariant()));
        Assert.False(InterventionHashes.IsSha256(BaseSha[1..]));
        Assert.False(InterventionHashes.IsSha256(null));
    }

    [Fact]
    public void CapabilitiesAreExplicitAndNothingIsSupported()
    {
        Assert.DoesNotContain(CapabilityStatus.Supported, InterventionCapabilities.All.Values);
        Assert.Equal(CapabilityStatus.Experimental, InterventionCapabilities.All["operator.directional_projection"]);
        Assert.Equal(CapabilityStatus.Untested, InterventionCapabilities.All["aot"]);
        Assert.Equal(CapabilityStatus.Unsupported, InterventionCapabilities.All["profiles.hot_swap"]);
    }
}
