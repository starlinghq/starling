using Starlings.Generation;
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

    private static Qwen35Tolerances Tolerances => Qwen35TestAssets.Tolerances("ortgenai-fp32x-cpu");

    [Theory]
    [InlineData("en-short")]
    [InlineData("special-endoftext")]
    [InlineData("ru-de-mix")]
    [InlineData("thinking-on")]
    [InlineData("long-license")]
    public void Step0LogitsMatchTheReference(string id) => Qwen35LogitChecks.Step0LogitsMatchTheReference(Model, Case(id), Tolerances);

    [Fact]
    public void CachedDecodingEqualsAFreshPrefill() => Qwen35LogitChecks.CachedDecodingEqualsAFreshPrefill(Model, Case("en-list"), 12, Tolerances);

    [Fact]
    public void AdditionalStopTokenIdEndsAtItsFirstOccurrence() => Qwen35LogitChecks.AdditionalStopTokenIdEndsAtItsFirstOccurrence(Model, Case("en-list"), 3);

    /// <summary>
    /// Long context above the 1,666-token prompt of prompts-v1: greedy ids, step-0 logits and cached vs uncached
    /// decoding against the transformers reference on prompts-long-v1 (proposed fixture). The CPU default prefill is one
    /// call, which allocates about 1 MB of logits per prompt token, so these use 512-token chunks (bit-identical on CPU).
    /// Longer prompts (up to 32,844 tokens) are in docs/evidence/r8-qwen35-long-context/.
    /// </summary>
    [Theory]
    [InlineData("long-4096")]
    public void LongContextGreedyAndLogitsEqualTheReference(string id)
    {
        var c = Qwen35TestAssets.LoadLongContextCases().Single(x => x.Id == id);
        Qwen35ModelChecks.GreedyFromReferenceIds(Model, c, prefillChunkTokens: 512);
        Qwen35LogitChecks.Step0LogitsMatchTheReference(Model, c, Tolerances, prefillChunkTokens: 512);
        Qwen35LogitChecks.CachedDecodingEqualsAFreshPrefill(Model, c, 16, Tolerances, prefillChunkTokens: 512);
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

    /// <summary>
    /// Per-token logprobs on the real model: under greedy decoding every token is the most likely of its step (the
    /// logits read are those that selected it), the ids are unchanged by the option, and the first step's top 5 are the
    /// reference's step-0 top 5 within the profile's step-0 top-5 bound.
    /// </summary>
    [Fact]
    public void GreedyLogprobsAreTheStepArgmaxAndStepZeroMatchesTheReference()
    {
        var c = Case("en-list");
        var prompt = Qwen35Prompt.FromTokenIds(c.PromptIds);
        var plain = Model.Generate(prompt, new Qwen35ChatOptions { MaxNewTokens = 12 }, Ct);
        var result = Model.Generate(prompt, new Qwen35ChatOptions { MaxNewTokens = 12, ReturnLogprobs = true, TopLogprobs = 5 }, Ct);
        Assert.Equal(plain.TokenIds, result.TokenIds);
        Assert.Equal(c.ContentIds.Take(12), result.TokenIds);
        Assert.Equal(result.TokenIds, result.Logprobs.Select(l => l.TokenId));
        Assert.All(result.Logprobs, l =>
        {
            Assert.Equal(l.TokenId, l.TopLogprobs[0].TokenId);
            Assert.Equal(l.Logprob, l.TopLogprobs[0].Logprob);
            Assert.True(l.Logprob <= 0 && double.IsFinite(l.Logprob));
        });

        var tolerances = Tolerances;
        var first = result.Logprobs[0].TopLogprobs;
        Assert.Equal(c.Step0Top.Take(5).Select(t => t.Id), first.Select(t => t.TokenId));
        double top5 = c.Step0Top.Take(5).Zip(first).Max(p => Math.Abs(p.First.Logprob - p.Second.Logprob));
        Assert.True(top5 <= tolerances.Step0Top5LogprobMaxAbs, $"step-0 top-5 logprob max abs {top5} > {tolerances.Step0Top5LogprobMaxAbs} ({tolerances.Source})");
    }

    /// <summary>
    /// An abandoned streaming enumerator (not disposed after its first update) no longer pins the model: after a
    /// garbage collection its native sequence is released on a thread-pool thread and the model's reference count
    /// returns to its own reference.
    /// </summary>
    [Fact]
    public void AnAbandonedStreamReleasesItsSequenceAfterCollection()
    {
        var model = Model;
        Assert.Equal(1, model.Backend.References);
        long before = NativeLifetime.FinalizerReleases;
        StartAndAbandon(model, Qwen35Prompt.FromTokenIds(Case("en-short").PromptIds));
        Assert.Equal(2, model.Backend.References);

        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (model.Backend.References != 1 && DateTime.UtcNow < deadline)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            Thread.Sleep(50);
        }

        Assert.Equal(1, model.Backend.References);
        Assert.True(NativeLifetime.FinalizerReleases > before);

        // The model stays usable.
        Assert.Equal(c0(), model.Generate(Qwen35Prompt.FromTokenIds(Case("en-short").PromptIds), new Qwen35ChatOptions { MaxNewTokens = 4 }, Ct).TokenIds);

        IEnumerable<int> c0() => Case("en-short").ContentIds.Take(4);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void StartAndAbandon(Qwen35ChatModel model, Qwen35Prompt prompt)
    {
        var enumerator = model.GenerateStreaming(prompt, new Qwen35ChatOptions { MaxNewTokens = 8 }, CancellationToken.None).GetEnumerator();
        Assert.True(enumerator.MoveNext());
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
        Assert.Equal(Qwen35ChatModel.MaxContextTokens, info.ContextLength);
        Assert.Equal(262144, info.ModelContextLength);
    }
}
