using System.Text;
using Starlings.Interventions;

namespace Starlings.Models.Lfm3.Interventions.Tests;

/// <summary>
/// The LFM3 checks of a compiler manifest, run before any session exists. <c>Data/compiler-manifest.json</c> is the
/// manifest the Python compiler wrote for the synthetic profile on the real fp32 graph (sites after layers 8 and 16);
/// <c>Data/compiler-manifest-multi.json</c> the schema version 2 manifest of the two synthetic profiles in one graph
/// (projection after layers 8 and 16, addition after layers 12 and 16, strength vector [2]). Machine-local file
/// identities are removed from both.
/// </summary>
public sealed class ManifestCheckTests
{
    private static readonly string Text = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Data", "compiler-manifest.json")).ReplaceLineEndings("\n");

    private static readonly string MultiText = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Data", "compiler-manifest-multi.json")).ReplaceLineEndings("\n");

    private static InterventionManifest Parse(string text) => InterventionManifest.Parse(Encoding.UTF8.GetBytes(text));

    [Fact]
    public void TheCompiledMultiProfileManifestPassesForFp32()
    {
        InterventionManifest manifest = Parse(MultiText);
        Lfm3Interventions.CheckManifest(manifest, Lfm3Precision.Fp32);
        Assert.Equal(2, manifest.SchemaVersion);
        Assert.Equal(["lfm3-synthetic-test", "lfm3-synthetic-add"], manifest.ProfileIds);
        Assert.Equal([2L], manifest.StrengthInput.Shape);
        Assert.Equal([8, 12, 16], manifest.Sites.Select(s => s.Layer));
        Assert.Equal([0, 1], manifest.Sites[2].Profiles);
        Assert.Equal(MultiText, Encoding.UTF8.GetString(manifest.Serialize()));

        // Profile 0 is the profile of the single-profile manifest, byte for byte.
        Assert.Equal(Parse(Text).ProfileSha256, manifest.Profiles[0].Sha256);
    }

    [Fact]
    public void AMultiProfileStrengthOfTheWrongLengthIsRejected()
    {
        // [1] for two profiles breaks the manifest contract itself; the LFM3 check would reject it too.
        string other = MultiText.Replace("\"shape\": [\n        2\n      ]", "\"shape\": [\n        1\n      ]", StringComparison.Ordinal);
        Assert.NotEqual(MultiText, other);
        Assert.Throws<InterventionContractException>(() => Parse(other));
    }

    [Fact]
    public void AMultiProfileManifestForAnotherGraphIsRejected()
    {
        string other = MultiText.Replace("add6653fc8b3cf2c7ee9fa0205a4a9a833fcb85d3449a5fea06f8aa494b5ef77", new string('0', 64), StringComparison.Ordinal);
        Assert.Throws<InterventionMismatchException>(() => Lfm3Interventions.CheckManifest(Parse(other), Lfm3Precision.Fp32));
        Assert.Throws<InterventionMismatchException>(() => Lfm3Interventions.CheckManifest(Parse(MultiText), Lfm3Precision.Fp16));
    }

    [Fact]
    public void TheCompiledManifestPassesForFp32()
    {
        InterventionManifest manifest = Parse(Text);
        Lfm3Interventions.CheckManifest(manifest, Lfm3Precision.Fp32);
        Assert.Equal(Lfm3Interventions.BaseModelSha256(Lfm3Precision.Fp32), manifest.BaseModelSha256);
        Assert.Equal("intervention_strength", manifest.StrengthInput.Name);
    }

    [Fact]
    public void BaseModelShaIsTheLockedGraph()
    {
        Assert.Equal("add6653fc8b3cf2c7ee9fa0205a4a9a833fcb85d3449a5fea06f8aa494b5ef77", Lfm3Interventions.BaseModelSha256(Lfm3Precision.Fp32));
        Assert.Equal("d3093d69f6f954bd74f1dc9e59452ec5e9766e62d9c409456fb8ea5698a830fe", Lfm3Interventions.BaseModelSha256(Lfm3Precision.Fp16));
    }

    [Fact]
    public void AWrongBaseGraphIsRejected()
    {
        string other = Text.Replace("add6653fc8b3cf2c7ee9fa0205a4a9a833fcb85d3449a5fea06f8aa494b5ef77", new string('0', 64), StringComparison.Ordinal);
        Assert.Throws<InterventionMismatchException>(() => Lfm3Interventions.CheckManifest(Parse(other), Lfm3Precision.Fp32));
    }

    [Fact]
    public void TheFp32ArtifactIsRejectedForTheFp16Graph()
    {
        // The fp16 graph has another base SHA-256: the artifact was compiled for a different graph.
        Assert.Throws<InterventionMismatchException>(() => Lfm3Interventions.CheckManifest(Parse(Text), Lfm3Precision.Fp16));
    }

    [Fact]
    public void AnotherPrecisionLabelIsRejected()
    {
        string other = Text.Replace("\"precision\": \"fp32\"", "\"precision\": \"int8\"", StringComparison.Ordinal);
        Assert.Throws<InterventionMismatchException>(() => Lfm3Interventions.CheckManifest(Parse(other), Lfm3Precision.Fp32));
    }

    [Fact]
    public void AScalarStrengthIsRejected()
    {
        // The manifest contract allows shape [] or [1]; the LFM3 sequence passes [1], so [] would fail at the first call.
        string other = Text.Replace("\"shape\": [\n        1\n      ]", "\"shape\": []", StringComparison.Ordinal);
        Assert.NotEqual(Text, other);
        Assert.Throws<InterventionContractException>(() => Lfm3Interventions.CheckManifest(Parse(other), Lfm3Precision.Fp32));
    }

    [Theory]
    [InlineData("input_ids")]
    [InlineData("past_conv.0")]
    [InlineData("past_key_values.2.key")]
    public void AStrengthNamedLikeAnLfm3InputIsRejected(string name)
    {
        string other = Text.Replace("\"name\": \"intervention_strength\"", $"\"name\": \"{name}\"", StringComparison.Ordinal);
        Assert.Throws<InterventionContractException>(() => Lfm3Interventions.CheckManifest(Parse(other), Lfm3Precision.Fp32));
    }

    [Fact]
    public void ASiteOutsideTheDecoderLayersIsRejected()
    {
        string other = Text
            .Replace("\"layer\": 16", "\"layer\": 24", StringComparison.Ordinal)
            .Replace("/model/layers.16/Add_2/output_0", "/model/layers.24/Add_2/output_0", StringComparison.Ordinal);
        Assert.Throws<InterventionContractException>(() => Lfm3Interventions.CheckManifest(Parse(other), Lfm3Precision.Fp32));
    }

    [Fact]
    public void ExternalDataThatIsNotTheLockedWeightsIsRejected()
    {
        // The same graph file with other weights of the same size: model.onnx holds only offsets, so its hash matches the
        // lock while the weights do not. Base artifact and derived link are changed together, as a compiler would record them.
        const string locked = "bb631a02237e4f8a491e12b25c333849046332b6db60fe0a2120f278c982fa76";
        string other = Text.Replace(locked, new string('1', 64), StringComparison.Ordinal);
        Assert.NotEqual(Text, other);
        Parse(other); // still a consistent manifest
        Assert.Throws<InterventionMismatchException>(() => Lfm3Interventions.CheckManifest(Parse(other), Lfm3Precision.Fp32));
    }

    [Fact]
    public void AMissingLockedWeightFileIsRejected()
    {
        // Drop model.onnx_data_3 from base_artifacts and derived_files.
        InterventionManifest manifest = Parse(Text);
        InterventionManifest other = Parse(Text
            .Replace(",\n    {\n      \"role\": \"external_data\",\n      \"path\": \"onnx/model.onnx_data_3\",\n      \"sha256\": \"1538827aa1c279442457028c4e3d784b7ea82bcf03abd4b55e45fa4facf9fe62\",\n      \"size\": 784818176\n    }", string.Empty, StringComparison.Ordinal)
            .Replace(",\n    {\n      \"path\": \"model.onnx_data_3\",\n      \"sha256\": \"1538827aa1c279442457028c4e3d784b7ea82bcf03abd4b55e45fa4facf9fe62\",\n      \"size\": 784818176\n    }", string.Empty, StringComparison.Ordinal));
        Assert.Equal(manifest.BaseArtifacts.Count - 1, other.BaseArtifacts.Count);
        Assert.Throws<InterventionMismatchException>(() => Lfm3Interventions.CheckManifest(other, Lfm3Precision.Fp32));
    }

    [Fact]
    public void AnUnknownSiteMappingVersionIsRejected()
    {
        string other = Text.Replace("\"lfm3-1.7b/fp32/1\"", "\"lfm3-1.7b/fp32/2\"", StringComparison.Ordinal);
        Assert.NotEqual(Text, other);
        Assert.Throws<InterventionContractException>(() => Lfm3Interventions.CheckManifest(Parse(other), Lfm3Precision.Fp32));
    }

    [Fact]
    public void NothingIsSupported()
    {
        Assert.DoesNotContain(CapabilityStatus.Supported, Lfm3InterventionCapabilities.All.Values);
        Assert.Equal(CapabilityStatus.Unsupported, Lfm3InterventionCapabilities.All["graph.fp16"]);
        Assert.Equal(CapabilityStatus.Unsupported, Lfm3InterventionCapabilities.All["selection.mid_sequence_change"]);
        Assert.Equal(CapabilityStatus.Experimental, Lfm3InterventionCapabilities.All["graph.fp32"]);
        Assert.Equal(CapabilityStatus.Experimental, Lfm3InterventionCapabilities.All["profiles.multiple_per_graph"]);
    }
}
