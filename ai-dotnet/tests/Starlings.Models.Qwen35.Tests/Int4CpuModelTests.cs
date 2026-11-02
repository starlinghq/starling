using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>Loads the int4 variant on CPU once, or records why it cannot (the bundle is a local build, never hosted).</summary>
public sealed class Qwen35Int4CpuFixture : IDisposable
{
    public Qwen35Int4CpuFixture()
    {
        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Int4);
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
/// The Experimental int4 CPU profile (qwen3.5-0.8b/ortgenai-int4-cpu, issue #163) against its own reference: the same
/// int4 graph run by the upstream Python binding on the prompts-v1 prompt ids (proposed fixture
/// model-pipeline/proposed/qwen3.5-0.8b/ortgenai-int4-cpu). No maintainer tolerances exist for this profile yet, so
/// these tests check discrete equality only (greedy ids, finish reason, step-0 argmax); the float differences are
/// reported in docs/evidence/qwen35-int4-cpu-2026-11-01/README.md. They skip when the local int4 build is absent.
/// </summary>
public sealed class Int4CpuModelTests(Qwen35Int4CpuFixture fixture) : IClassFixture<Qwen35Int4CpuFixture>
{
    private Qwen35ChatModel Model
    {
        get
        {
            Assert.SkipWhen(fixture.Model is null, fixture.Reason);
            return fixture.Model!;
        }
    }

    private static GenerationCase Case(string id) => Qwen35TestAssets.LoadInt4Cases().Single(c => c.Id == id);

    [Fact]
    public void TheFixtureSharesThePromptIdsOfTheAcceptedReference()
    {
        var accepted = Qwen35TestAssets.LoadCases().ToDictionary(c => c.Id);
        var int4 = Qwen35TestAssets.LoadInt4Cases();
        Assert.Equal(accepted.Keys.Order(StringComparer.Ordinal), int4.Select(c => c.Id).Order(StringComparer.Ordinal));
        foreach (var c in int4)
        {
            Assert.Equal(accepted[c.Id].Rendered, c.Rendered);
            Assert.Equal(accepted[c.Id].PromptIds, c.PromptIds);
        }
    }

    [Theory]
    [InlineData("en-short")]
    [InlineData("hi")]
    [InlineData("special-imend")]
    [InlineData("special-endoftext")]
    [InlineData("system-pirate")]
    [InlineData("zh")]
    public void GreedyIdsEqualThePythonInt4Reference(string id) => Qwen35ModelChecks.GreedyFromReferenceIds(Model, Case(id));

    [Theory]
    [InlineData("en-short")]
    [InlineData("special-endoftext")]
    [InlineData("ru-de-mix")]
    [InlineData("long-license")]
    public void Step0ArgmaxEqualsThePythonInt4Reference(string id)
    {
        var c = Case(id);
        var r = Model.Generate(Qwen35Prompt.FromTokenIds(c.PromptIds), new Qwen35ChatOptions { MaxNewTokens = 1 }, TestContext.Current.CancellationToken);
        int first = r.StopTokenId ?? r.TokenIds[0];
        Assert.Equal(c.Step0Top[0].Id, first);
        Assert.Equal(c.OutputIds[0], first);
    }

    [Fact]
    public void ModelInfoNamesTheInt4Variant()
    {
        Assert.Equal(Qwen35Variant.Int4, Model.Info.Variant);
        Assert.Equal("qwen3.5-0.8b/ortgenai-int4", Model.Info.Profile);
        Assert.Equal(GenAIExecutionProvider.Cpu, Model.Info.Provider);
    }
}
