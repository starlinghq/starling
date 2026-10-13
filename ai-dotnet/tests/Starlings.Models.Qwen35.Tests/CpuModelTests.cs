using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>Loads the fp32x bundle on CPU once (skips without it: not in Git, 3.1 GB).</summary>
public sealed class Qwen35CpuFixture : IDisposable
{
    public Qwen35CpuFixture()
    {
        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Fp32x);
        Reason = reason;
        if (files is not null)
        {
            Model = Qwen35ChatModel.Load(files);
        }
    }

    public Qwen35ChatModel? Model { get; }

    public string Reason { get; }

    public void Dispose() => Model?.Dispose();
}

/// <summary>
/// fp32x on CPU against the transformers fp32 reference (prompts-v1). The full 22-prompt run is the fidelity runner's
/// job (ai-dotnet/tools/Starlings.Qwen35.Fidelity); these tests keep to short outputs.
/// </summary>
public sealed class CpuModelTests(Qwen35CpuFixture fixture) : IClassFixture<Qwen35CpuFixture>
{
    private Qwen35ChatModel Model
    {
        get
        {
            Assert.SkipWhen(fixture.Model is null, fixture.Reason);
            return fixture.Model!;
        }
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static GenerationCase Case(string id) => Qwen35TestAssets.LoadCases().Single(c => c.Id == id);

    [Theory]
    [InlineData("en-short")]
    [InlineData("hi")]
    [InlineData("special-imend")]
    [InlineData("special-endoftext")]
    [InlineData("special-broken")]
    [InlineData("system-pirate")]
    public void GreedyIdsEqualTheReferenceUpToTheStopToken(string id) => Qwen35ModelChecks.GreedyFromReferenceIds(Model, Case(id));

    [Theory]
    [InlineData("thinking-on")]
    [InlineData("multi-turn")]
    [InlineData("special-think-tool")]
    public void TextPathEqualsTheReference(string id) => Qwen35ModelChecks.TextPathMatches(Model, Case(id), 12);

    [Fact]
    public void StreamingReassemblesNonAsciiOutput() => Qwen35ModelChecks.StreamingReassembles(Model, Case("emoji"), 24);

    [Fact]
    public void MaxNewTokensBoundary()
    {
        var c = Case("en-short");
        int content = c.ContentIds.Length;
        var at = Model.Generate(Qwen35Prompt.FromTokenIds(c.PromptIds), new Qwen35ChatOptions { MaxNewTokens = content }, Ct);
        Assert.Equal(GenerationFinishReason.Length, at.FinishReason);
        Assert.Equal(c.ContentIds, at.TokenIds);
        var past = Model.Generate(Qwen35Prompt.FromTokenIds(c.PromptIds), new Qwen35ChatOptions { MaxNewTokens = content + 1 }, Ct);
        Assert.Equal(GenerationFinishReason.Stop, past.FinishReason);
        Assert.Equal(Qwen35ChatModel.ImEndTokenId, past.StopTokenId);
        Assert.Equal(c.ContentIds, past.TokenIds);
        Assert.Equal(content, past.Usage.CompletionTokens);
    }

    [Fact]
    public void StopSequenceCutsTheReferenceText()
    {
        var c = Case("en-short");
        string full = Model.Tokenizer.Decode(c.ContentIds, false);
        string stop = Model.Tokenizer.Decode(c.ContentIds.AsSpan(2, 2), false);
        var r = Model.Generate(Qwen35Prompt.FromTokenIds(c.PromptIds), new Qwen35ChatOptions { MaxNewTokens = 32, StopSequences = [stop] }, Ct);
        Assert.Equal(GenerationFinishReason.StopSequence, r.FinishReason);
        Assert.Equal(full[..full.IndexOf(stop, StringComparison.Ordinal)], r.Text);
    }

    [Fact]
    public void ConcurrentRequestsAreIndependent()
    {
        Assert.False(Model.Info.SerializesNativeCalls);
        Qwen35ModelChecks.ConcurrentRequestsAreIndependent(Model, [Case("zh"), Case("code-py"), Case("ja"), Case("en-math")], 12, rounds: 2, threads: 4);
    }

    [Fact]
    public void CancellationBetweenStepsThenReuse() => Qwen35ModelChecks.CancellationThenReuse(Model, Case("en-list"), Case("en-short"));

    [Fact]
    public void SeededSamplingIsReproducibleAndTopKOneIsGreedy()
    {
        var c = Case("en-list");
        var prompt = Qwen35Prompt.FromTokenIds(c.PromptIds);
        var s = new SamplingOptions { Seed = 7, Temperature = 0.7, TopK = 20, TopP = 0.8 };
        var a = Model.Generate(prompt, new Qwen35ChatOptions { MaxNewTokens = 16, Sampling = s }, Ct);
        var b = Model.Generate(prompt, new Qwen35ChatOptions { MaxNewTokens = 16, Sampling = s }, Ct);
        Assert.Equal(a.TokenIds, b.TokenIds);
        var k1 = Model.Generate(prompt, new Qwen35ChatOptions { MaxNewTokens = 16, Sampling = new SamplingOptions { Seed = 1, TopK = 1 } }, Ct);
        Assert.Equal(c.ContentIds.Take(16), k1.TokenIds);
    }

    [Fact]
    public void InfoNamesEveryIdentity()
    {
        var info = Model.Info;
        Assert.Equal("qwen3.5-0.8b/ortgenai-fp32x", info.Profile);
        Assert.Equal("Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17", info.Checkpoint);
        Assert.Equal("0.17.1", info.GenAIVersion);
        Assert.Contains(Qwen35ModelAssets.TokenizerSha256, info.Tokenizer, StringComparison.Ordinal);
        Assert.Contains(Qwen35ChatTemplate.TemplateSha256, info.ChatTemplate, StringComparison.Ordinal);
        Assert.Contains(info.ConfigOverlays, o => o.Contains("248319", StringComparison.Ordinal));
        Assert.Equal([248044, 248046], info.StopTokenIds);
        Assert.Equal(262144, info.ContextLength);
    }
}
