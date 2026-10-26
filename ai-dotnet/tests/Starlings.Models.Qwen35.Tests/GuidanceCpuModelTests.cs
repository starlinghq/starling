using System.Text.Json;
using System.Text.RegularExpressions;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>
/// The fp32x bundle on CPU in a temporary directory with ONNX Runtime GenAI's guidance files added
/// (<see cref="Qwen35GuidedDirectory"/>). Skips without the model or a verified <c>tokenizer_config.json</c>.
/// </summary>
public sealed class Qwen35GuidedCpuFixture : IDisposable
{
    private readonly Qwen35GuidedDirectory? _directory;

    public Qwen35GuidedCpuFixture()
    {
        _directory = Qwen35GuidedDirectory.TryCreate(Qwen35Variant.Fp32x, out string reason);
        Reason = reason;
        if (_directory is not null)
        {
            try
            {
                Model = Qwen35ChatModel.Load(_directory.Files);
            }
            catch
            {
                // A constructor that throws gets no Dispose; remove the temporary directory here.
                _directory.Dispose();
                throw;
            }
        }
    }

    public Qwen35ChatModel? Model { get; }

    public string Reason { get; }

    public void Dispose()
    {
        Model?.Dispose();
        _directory?.Dispose();
    }
}

/// <summary>
/// Guided decoding on the real fp32x model on CPU (greedy; outputs are checked against the constraint, not a reference).
/// Runs alone: its fixture loads another 3.1 GB model.
/// </summary>
[Collection(OwnModelTestGroup.Name)]
public sealed class GuidanceCpuModelTests(Qwen35GuidedCpuFixture fixture) : IClassFixture<Qwen35GuidedCpuFixture>
{
    private static readonly Qwen35Tool Weather = Qwen35Tool.Function(
        "get_weather",
        "Current weather for a city.",
        JsonDocument.Parse("""
        {"type": "object", "properties": {
          "city": {"type": "string"},
          "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }, "required": ["city"]}
        """).RootElement.Clone());

    private static readonly Qwen35Tool Timer = Qwen35Tool.Function(
        "set_timer",
        "Starts a countdown timer.",
        JsonDocument.Parse("""{"type": "object", "properties": {"minutes": {"type": "integer"}}, "required": ["minutes"]}""").RootElement.Clone());

    private Qwen35ChatModel Model
    {
        get
        {
            Assert.SkipWhen(fixture.Model is null, fixture.Reason);
            return fixture.Model!;
        }
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void TheDirectoryHasTheGuidanceFiles() => Assert.Empty(Model.Info.GuidanceMissingFiles);

    [Fact]
    public void UnguidedGreedyIdsAreUnchanged() =>
        Qwen35ModelChecks.GreedyFromReferenceIds(Model, Qwen35TestAssets.LoadCases().Single(c => c.Id == "en-short"));

    [Fact]
    public void ARegexIsFollowed()
    {
        var r = Model.Generate(
            [Qwen35Message.User("Write a product code.")],
            new Qwen35ChatOptions { MaxNewTokens = 16, Guidance = GenerationGuidance.Regex("FORCED-[0-9]{3}") },
            Ct);
        Report("regex", r);
        Assert.Matches("^FORCED-[0-9]{3}$", r.Text);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
    }

    [Fact]
    public void AJsonSchemaIsFollowed()
    {
        const string Schema = """{"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "additionalProperties": false}""";
        var r = Model.Generate(
            [Qwen35Message.User("Name the capital of France and its country as JSON.")],
            new Qwen35ChatOptions { MaxNewTokens = 48, Guidance = GenerationGuidance.JsonSchema(Schema) },
            Ct);
        Report("json_schema", r);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
        using var json = JsonDocument.Parse(r.Text);
        Assert.Equal(JsonValueKind.String, json.RootElement.GetProperty("city").ValueKind);
        Assert.Equal(JsonValueKind.String, json.RootElement.GetProperty("country").ValueKind);
        Assert.Equal(2, json.RootElement.EnumerateObject().Count());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheRequiredToolGrammarYieldsOneParsedCall(bool fastForward)
    {
        Qwen35Tool[] tools = [Weather, Timer];
        var options = new Qwen35ChatOptions
        {
            MaxNewTokens = 64,
            Tools = tools,
            Guidance = Qwen35ToolGrammar.Build(tools, new Qwen35ToolGrammarOptions { FastForward = fastForward }),
        };
        var r = Model.Generate([Qwen35Message.User("What's the weather in Porto?")], options, Ct);
        Report(fastForward ? "tool grammar, fast-forward" : "tool grammar", r);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
        var parsed = Qwen35ToolCallParser.Parse(r.Text, options);
        var call = Assert.Single(parsed.ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal(string.Empty, parsed.Text);
        Assert.Contains("porto", call.Arguments.GetProperty("city").GetString()!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FastForwardKeepsTheTextOfTheOneTokenPerStepRun()
    {
        // Forced tokens are tokenized by GenAI rather than chosen by the model, so the ids may differ; the text is what
        // the parser sees.
        Qwen35Tool[] tools = [Weather, Timer];
        Qwen35Message[] conversation = [Qwen35Message.User("Set a timer for 25 minutes.")];
        var plain = Model.Generate(conversation, new Qwen35ChatOptions { MaxNewTokens = 64, Tools = tools, Guidance = Qwen35ToolGrammar.Build(tools) }, Ct);
        var ff = Model.Generate(
            conversation,
            new Qwen35ChatOptions { MaxNewTokens = 64, Tools = tools, Guidance = Qwen35ToolGrammar.Build(tools, new Qwen35ToolGrammarOptions { FastForward = true }) },
            Ct);
        Report("timer", plain);
        Report("timer, fast-forward", ff);
        Assert.Equal(plain.Text, ff.Text);
        Assert.Equal("set_timer", Assert.Single(Qwen35ToolCallParser.Parse(ff.Text, tools, enableThinking: false).ToolCalls).Name);
    }

    private static void Report(string name, GenerationResult r) =>
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"{name}: {r.Usage.CompletionTokens} tokens, {r.FinishReason}, decode {r.Timings.Decode.TotalMilliseconds:F0} ms: {Regex.Escape(r.Text)}");
}
