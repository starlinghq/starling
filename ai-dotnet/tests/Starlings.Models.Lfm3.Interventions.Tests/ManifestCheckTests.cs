using System.Text;
using Starlings.Interventions;

namespace Starlings.Models.Lfm3.Interventions.Tests;

/// <summary>
/// The LFM3 checks of a compiler manifest, run before any session exists. <c>Data/compiler-manifest.json</c> is the
/// manifest the Python compiler wrote for the synthetic profile on the real fp32 graph (sites after layers 8 and 16).
/// </summary>
public sealed class ManifestCheckTests
{
    private static readonly string Text = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Data", "compiler-manifest.json")).ReplaceLineEndings("\n");

    private static InterventionManifest Parse(string text) => InterventionManifest.Parse(Encoding.UTF8.GetBytes(text));

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
    public void NothingIsSupported()
    {
        Assert.DoesNotContain(CapabilityStatus.Supported, Lfm3InterventionCapabilities.All.Values);
        Assert.Equal(CapabilityStatus.Unsupported, Lfm3InterventionCapabilities.All["graph.fp16"]);
        Assert.Equal(CapabilityStatus.Unsupported, Lfm3InterventionCapabilities.All["selection.mid_sequence_change"]);
        Assert.Equal(CapabilityStatus.Experimental, Lfm3InterventionCapabilities.All["graph.fp32"]);
    }
}
