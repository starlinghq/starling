using Starlings.Acquisition;

namespace Starlings.Models.Clover.Tests;

/// <summary>Weight-free checks of the checkpoint definition, the asset lock and the capability table.</summary>
public sealed class CloverDefinitionTests
{
    [Fact]
    public void ProfileNamesThePinnedCheckpoint()
    {
        Assert.Equal("Diffuspace/Clover-35B-A3B-Preview@4b0402b65e523d0183a14c431955438714cb16ff", CloverCheckpoint.Profile.Checkpoint);
        Assert.Equal([CloverCheckpoint.EndOfTextTokenId, CloverCheckpoint.ImEndTokenId], CloverCheckpoint.Profile.EosTokenIds);
        Assert.Equal("clover-bpe-chat", CloverCheckpoint.Profile.TokenizerProfile);
        Assert.Equal(CloverChatTemplate.TemplateSha256, CloverCheckpoint.Profile.ChatTemplateSha256);
    }

    [Fact]
    public void LockHoldsEveryShardTheGdnFilesAndTheTokenizer()
    {
        ModelAssetLock assetLock = CloverCheckpoint.Lock;
        Assert.Equal(CloverCheckpoint.Repository, assetLock.Repository);
        Assert.Equal(CloverCheckpoint.Revision, assetLock.Revision);
        Assert.Equal(34, assetLock.Files.Count);
        for (int i = 1; i <= 26; i++)
        {
            Assert.Contains(assetLock.Files, f => f.Path == $"model-{i:D5}-of-00026.safetensors");
        }

        Assert.Contains(assetLock.Files, f => f.Path == "bd_gdn_conv_right_weight.safetensors");
        Assert.Contains(assetLock.Files, f => f.Path == "bd_gdn_softmax_scalars.safetensors");
        Assert.Contains(assetLock.Files, f => f.Path == "tokenizer.json" && f.Sha256 == CloverCheckpoint.TokenizerSha256);
        Assert.Contains(assetLock.Files, f => f.Path == "chat_template.jinja" && f.Sha256 == CloverChatTemplate.TemplateSha256);
        Assert.DoesNotContain(assetLock.Files, f => f.Path.EndsWith(".py", StringComparison.Ordinal));
        Assert.Equal(69_342_873_290, assetLock.Files.Sum(f => f.Size));
    }

    [Fact]
    public void NothingIsSupportedAndGenerationIsUnsupported()
    {
        Assert.DoesNotContain(CapabilityStatus.Supported, CloverCapabilities.All.Values);
        foreach (string name in (string[])["generation", "generation.self_speculative", "sampling", "tool_calling", "vision"])
        {
            Assert.Equal(CapabilityStatus.Unsupported, CloverCapabilities.All[name]);
        }

        Assert.Equal(CapabilityStatus.Experimental, CloverCapabilities.All["chat_template.text"]);
    }
}
