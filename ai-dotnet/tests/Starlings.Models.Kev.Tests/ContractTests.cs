using Starlings.Models.Kev;

namespace Starlings.Models.Kev.Tests;

/// <summary>The profile contracts, locks and capability table (no model needed).</summary>
public sealed class ContractTests
{
    [Theory]
    [InlineData(KevModel.Kev08B, "kev-0.8b/onnx-fp32-cpu", "jaredpalmer/kev-0.8b", "9a45d25eb2ab761841196625383fa1dff0e56c1e", 1024, 24, 2, 16, 2.3510958125672174)]
    [InlineData(KevModel.Kev4B, "kev-4b/onnx-fp32-cpu", "jaredpalmer/kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101", 2560, 32, 4, 32, 2.406050072164233)]
    public void ProfilesPinTheContract(KevModel model, string id, string repository, string revision, int hidden, int layers, int kvHeads, int linearValueHeads, double temperature)
    {
        var p = KevModelProfile.For(model);
        Assert.Equal(id, p.Id);
        Assert.Equal(repository, p.Repository);
        Assert.Equal(revision, p.Revision);
        Assert.Equal(hidden, p.HiddenSize);
        Assert.Equal(layers, p.Layers);
        Assert.Equal(kvHeads, p.KeyValueHeads);
        Assert.Equal(linearValueHeads, p.LinearValueHeads);
        Assert.Equal(temperature, p.Temperature);
        Assert.Same(p, KevModelProfile.ForRepository(repository));
        foreach (string sha in new[] { p.Revision, p.BaseRevision, p.HeadPtSha256, p.AdapterSha256 })
        {
            Assert.Matches("^[0-9a-f]{40}$|^[0-9a-f]{64}$", sha);
        }
    }

    [Theory]
    [InlineData("jaredpalmer/kev-9b")]
    [InlineData("jaredpalmer/kev-27b")]
    [InlineData("jaredpalmer/kev-0.6b")]
    [InlineData("midudev/kev-4b-ONNX")]
    [InlineData("JaredPalmer/kev-4b")]
    public void RefusesOtherRepositories(string repository) => Assert.Null(KevModelProfile.ForRepository(repository));

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void LocksPinEveryBundleFile(KevModel model)
    {
        var assetLock = KevModelAssets.Lock(model);
        Assert.Equal(
            [KevModelAssets.ModelFile, KevModelAssets.ModelDataFile, KevModelAssets.HeadFile, KevModelAssets.HeadMetadataFile, KevModelAssets.TokenizerFile],
            assetLock.Files.Select(f => f.Path));
        Assert.All(assetLock.Files, f => Assert.Matches("^[0-9a-f]{64}$", f.Sha256));
        Assert.All(assetLock.Files, f => Assert.DoesNotMatch("^0{64}$", f.Sha256));
        Assert.Equal(KevModelAssets.TokenizerSha256, assetLock.GetFile(KevModelAssets.TokenizerFile).Sha256);
    }

    [Fact]
    public void FullAttentionEveryFourthLayer()
    {
        Assert.Equal([3, 7, 11, 15, 19, 23, 27, 31], Enumerable.Range(0, 32).Where(KevModelProfile.IsFullAttention));
    }

    [Fact]
    public void EveryCapabilityHasAnExplicitStatus()
    {
        Assert.All(KevCapabilities.All.Values, s => Assert.True(Enum.IsDefined(s)));
        Assert.DoesNotContain(KevCapabilityStatus.Supported, KevCapabilities.All.Values); // Experimental profiles
        foreach (string key in new[] { "decision.noul", "decision.choice", "decision.score", "state_over_8192_tokens", "truncation", "bf16", "quantized", "cuda", "asset_acquisition.download" })
        {
            Assert.True(KevCapabilities.All.ContainsKey(key), key);
        }
    }

    [Fact]
    public void CudaProfilesShareTheBundleAndKeepTf32Off()
    {
        foreach (var profile in KevModelProfile.All)
        {
            Assert.Equal(profile.Id.Replace("-cpu", "-cuda", StringComparison.Ordinal), profile.CudaId);
        }

        Assert.Equal(KevCapabilityStatus.Experimental, KevCapabilities.All["cuda"]);
        Assert.Equal(KevCapabilityStatus.Experimental, KevCapabilities.All["concurrent_calls.cuda.serialized"]);
        Assert.Equal(KevCapabilityStatus.Untested, KevCapabilities.All["concurrent_calls.cuda.unserialized"]);
        Assert.Equal(KevCapabilityStatus.Unsupported, KevCapabilities.All["tf32"]);
        Assert.Null(new KevOptions().SerializeNativeCalls); // provider default: serialized on CUDA only
    }

    [Fact]
    public void OptionsRejectOutOfContractSettings()
    {
        new KevOptions { Session = new() { Provider = Starlings.Runtime.Onnx.OnnxExecutionProvider.Cuda } }.Validate();
        Assert.Throws<NotSupportedException>(() => new KevOptions { Session = new() { Provider = (Starlings.Runtime.Onnx.OnnxExecutionProvider)99 } }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new KevOptions { MaxStateTokens = KevModelProfile.MaxStateTokens + 1 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new KevOptions { MaxRowTokens = KevModelProfile.MaxRowTokens + 1 }.Validate());
        new KevOptions().Validate();
    }

    [Fact]
    public void AnswerMappingFollowsTheSystemOneAdapter()
    {
        var noul = new KevAnswer("u", KevQuestion.Noul("q"), [0.2f, 0.8f]);
        Assert.Equal(0.8f, noul.Yes);
        Assert.Equal("true", noul.Choice);
        Assert.Equal((0.8 - 0.5) / 0.5, noul.Confidence, 6);

        var choice = new KevAnswer("c", KevQuestion.Choice("q", "a", "b", "c"), [0.5f, 0.5f, 0f]);
        Assert.Equal("a", choice.Choice); // first on ties, as Python max(range, key=...)
        Assert.Equal(1.0, new KevAnswer("o", KevQuestion.Choice("q", "only"), [1f]).Confidence);

        var score = new KevAnswer("s", KevQuestion.Score("q", "lo", "mid", "hi"), [0f, 0f, 1f]);
        Assert.Equal(2.0, score.ExpectedScore, 6);
        Assert.Equal(1.0, score.Confidence, 6);
        var uniform = new KevAnswer("s", KevQuestion.Score("q", "lo", "mid", "hi"), [1f / 3, 1f / 3, 1f / 3]);
        Assert.Equal(0.0, uniform.Confidence, 6);
        Assert.Equal(["lo", "mid", "hi"], score.Legend!);
    }
}
