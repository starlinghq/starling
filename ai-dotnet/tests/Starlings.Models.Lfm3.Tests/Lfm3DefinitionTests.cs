using Starlings.Acquisition;

namespace Starlings.Models.Lfm3.Tests;

/// <summary>Weight-free checks of the checkpoint definition, the asset locks and the capability table.</summary>
public sealed class Lfm3DefinitionTests
{
    [Fact]
    public void LayerLayoutIsThatOfTheConfig()
    {
        Assert.Equal(Lfm3Checkpoint.Layers, Lfm3Checkpoint.LayerTypes.Count);
        Assert.Equal(13, Lfm3Checkpoint.LayerTypes.Count(t => t == "conv"));
        Assert.Equal(5, Lfm3Checkpoint.LayerTypes.Count(t => t == "sliding_attention"));
        Assert.Equal(6, Lfm3Checkpoint.LayerTypes.Count(t => t == "full_attention"));
    }

    [Fact]
    public void ProfileNamesThePinnedCheckpoint()
    {
        Assert.Equal("LiquidAI/LFM3-1.7B@40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd", Lfm3Checkpoint.Profile.Checkpoint);
        Assert.Equal([Lfm3Checkpoint.ImEndTokenId], Lfm3Checkpoint.Profile.EosTokenIds);
        Assert.Equal("lfm3-bpe-chat", Lfm3Checkpoint.Profile.TokenizerProfile);
    }

    [Theory]
    [InlineData(Lfm3Precision.Fp32, "onnx/model.onnx", 6)]
    [InlineData(Lfm3Precision.Fp16, "onnx/model_fp16.onnx", 4)]
    public void LocksHoldTheGraphItsDataAndTheTokenizer(Lfm3Precision precision, string graph, int files)
    {
        ModelAssetLock assetLock = Lfm3ModelAssets.LockFor(precision);
        Assert.Equal(Lfm3ModelAssets.Repository, assetLock.Repository);
        Assert.Equal(Lfm3ModelAssets.Revision, assetLock.Revision);
        Assert.Equal(graph, Lfm3ModelAssets.ModelFile(precision));
        Assert.Equal(files, assetLock.Files.Count);
        Assert.Contains(assetLock.Files, f => f.Path == graph);
        Assert.Contains(assetLock.Files, f => f.Path == Lfm3ModelAssets.TokenizerFile && f.Sha256 == Lfm3ModelAssets.TokenizerSha256);
        Assert.All(assetLock.Files.Where(f => f.Path != graph && f.Path != Lfm3ModelAssets.TokenizerFile), f => Assert.StartsWith(graph + "_data", f.Path, StringComparison.Ordinal));
    }

    [Fact]
    public void NothingIsSupportedAndTheUnimplementedIsUnsupported()
    {
        Assert.DoesNotContain(CapabilityStatus.Supported, Lfm3Capabilities.All.Values);
        foreach (string name in (string[])["sampling", "logprobs", "guided_decoding", "tool_calling", "vision", "audio", "prefix_reuse"])
        {
            Assert.Equal(CapabilityStatus.Unsupported, Lfm3Capabilities.All[name]);
        }

        Assert.Equal(CapabilityStatus.Untested, Lfm3Capabilities.All["generation.cuda"]);
    }

    [Fact]
    public void OptionsRejectAContextBeyondTheCheckpoint()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Lfm3Options { ContextLength = Lfm3Checkpoint.ContextLength + 1 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new Lfm3Options { ContextLength = 0 }.Validate());
        new Lfm3Options { ContextLength = Lfm3Checkpoint.ContextLength }.Validate();
    }

    [Fact]
    public void ArgMaxTakesTheFirstMaximum()
    {
        Assert.Equal(1, Lfm3Sequence.ArgMax([0f, 3f, 3f, -1f]));
        Assert.Equal(0, Lfm3Sequence.ArgMax([float.NegativeInfinity, float.NegativeInfinity]));
    }
}
