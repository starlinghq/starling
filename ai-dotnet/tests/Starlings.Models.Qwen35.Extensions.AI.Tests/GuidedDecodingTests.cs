using System.Text.Json;
using Starlings.Generation.OnnxGenAI;
using Starlings.TestSupport;
using Starlings.Tokenizers;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>
/// The IChatClient's guided requests with a scripted sequence: <see cref="ChatToolMode.RequireAny"/>,
/// <see cref="ChatToolMode.RequireSpecific"/> and JSON response formats map to a constraint only when the model
/// directory has the guidance files, fail with <see cref="NotSupportedException"/> naming the missing files otherwise,
/// and leave unguided requests unchanged.
/// </summary>
public sealed class GuidedDecodingTests
{
    private const string WeatherCall = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static readonly AIFunction Weather = AIFunctionFactory.Create((string city) => $"Sunny in {city}", "get_weather", "Current weather for a city.");

    private static readonly AIFunction Time = AIFunctionFactory.Create(() => "12:00", "get_time", "The current time.");

    private static Tokenizer Tokenizer
    {
        get
        {
            var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
            Assert.SkipWhen(tokenizer is null, reason);
            return tokenizer!;
        }
    }

    private static ScriptedBackend Backend(string reply, Qwen35ModelInfo? info = null)
    {
        var tokenizer = Tokenizer;
        int[] ids = [.. tokenizer.Encode(reply, addSpecialTokens: false).Ids, Qwen35ChatModel.ImEndTokenId];
        return new ScriptedBackend(tokenizer, _ => ids, info ?? ScriptedBackend.GuidedInfo);
    }

    private static ChatMessage[] Question => [new(ChatRole.User, "Weather in Paris?")];

    [Theory]
    [InlineData("require any")]
    [InlineData("require specific")]
    [InlineData("json")]
    [InlineData("json schema")]
    public async Task WithoutTheGuidanceFilesGuidedRequestsNameTheMissingFiles(string name)
    {
        var backend = Backend("x", ScriptedBackend.Info);
        using var client = new Qwen35ChatClient(backend.Model);
        ChatOptions options = name switch
        {
            "require any" => new() { Tools = [Weather], ToolMode = ChatToolMode.RequireAny },
            "require specific" => new() { Tools = [Weather], ToolMode = ChatToolMode.RequireSpecific("get_weather") },
            "json" => new() { ResponseFormat = ChatResponseFormat.Json },
            _ => new() { ResponseFormat = ChatResponseFormat.ForJsonSchema(JsonDocument.Parse("""{"type":"object"}""").RootElement) },
        };
        var ex = await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(Question, options, Ct));
        Assert.Contains("tokenizer.json and tokenizer_config.json are missing", ex.Message, StringComparison.Ordinal);
        Assert.Empty(backend.Sequences);
    }

    [Fact]
    public async Task RequireAnyConstrainsToTheOfferedToolsAndReturnsTheCall()
    {
        var backend = Backend(WeatherCall);
        using var client = new Qwen35ChatClient(backend.Model);
        var response = await client.GetResponseAsync(Question, new ChatOptions { Tools = [Weather, Time], ToolMode = ChatToolMode.RequireAny }, Ct);

        var guidance = Assert.IsType<GenerationGuidance>(backend.Single().Options.Guidance);
        Assert.Equal(GenerationGuidanceKind.Lark, guidance.Kind);
        Assert.Contains("<function=get_weather>", guidance.Data, StringComparison.Ordinal);
        Assert.Contains("<function=get_time>", guidance.Data, StringComparison.Ordinal);
        Assert.StartsWith("start: call (", guidance.Data, StringComparison.Ordinal);
        Assert.False(guidance.FastForward);

        // Both tools are rendered into the prompt, as with ChatToolMode.Auto.
        var prompt = backend.Model.CreatePrompt([Qwen35Message.User("Weather in Paris?")], ToTools([Weather, Time]));
        Assert.Equal(prompt.TokenIds, backend.Single().Prompt);

        var call = Assert.IsType<FunctionCallContent>(Assert.Single(response.Messages.Single().Contents));
        Assert.Equal("get_weather", call.Name);
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
    }

    [Fact]
    public async Task RequireSpecificAdmitsOnlyTheNamedToolAndSingleCallsWhenAsked()
    {
        var backend = Backend(WeatherCall);
        using var client = new Qwen35ChatClient(backend.Model);
        await client.GetResponseAsync(Question, new ChatOptions { Tools = [Weather, Time], ToolMode = ChatToolMode.RequireSpecific("get_weather"), AllowMultipleToolCalls = false }, Ct);
        string grammar = backend.Single().Options.Guidance!.Data;
        Assert.Contains("<function=get_weather>", grammar, StringComparison.Ordinal);
        Assert.DoesNotContain("get_time", grammar, StringComparison.Ordinal);
        Assert.StartsWith("start: call\n", grammar, StringComparison.Ordinal);
    }

    [Fact]
    public async Task StreamingRequireAnyEmitsTheCall()
    {
        var backend = Backend(WeatherCall);
        using var client = new Qwen35ChatClient(backend.Model);
        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingResponseAsync(Question, new ChatOptions { Tools = [Weather], ToolMode = ChatToolMode.RequireAny }, Ct))
        {
            updates.Add(u);
        }

        Assert.NotNull(backend.Single().Options.Guidance);
        var call = Assert.Single(updates.SelectMany(u => u.Contents).OfType<FunctionCallContent>());
        Assert.Equal("get_weather", call.Name);
        Assert.Equal(ChatFinishReason.ToolCalls, updates[^1].FinishReason);
    }

    [Fact]
    public async Task GuidanceFastForwardBuildsTheGrammarWithFastForward()
    {
        var backend = Backend(WeatherCall);
        using var client = new Qwen35ChatClient(backend.Model, new Qwen35ChatClientOptions { GuidanceFastForward = true });
        await client.GetResponseAsync(Question, new ChatOptions { Tools = [Weather], ToolMode = ChatToolMode.RequireAny }, Ct);
        Assert.True(backend.Single().Options.Guidance!.FastForward);
    }

    [Fact]
    public async Task JsonSchemaAndJsonModeMapToJsonSchemaGuidance()
    {
        const string Schema = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""";
        var backend = Backend("""{"city": "Paris"}""");
        using var client = new Qwen35ChatClient(backend.Model);
        var response = await client.GetResponseAsync(
            Question,
            new ChatOptions { ResponseFormat = ChatResponseFormat.ForJsonSchema(JsonDocument.Parse(Schema).RootElement) },
            Ct);
        var guidance = backend.Sequences.Last().Options.Guidance!;
        Assert.Equal(GenerationGuidanceKind.JsonSchema, guidance.Kind);
        Assert.Equal(Schema, guidance.Data);
        Assert.Equal("""{"city": "Paris"}""", response.Text);

        await client.GetResponseAsync(Question, new ChatOptions { ResponseFormat = ChatResponseFormat.Json }, Ct);
        Assert.Equal("""{"type": "object"}""", backend.Sequences.Last().Options.Guidance!.Data);
    }

    [Theory]
    [InlineData("json with tools")]
    [InlineData("json with require any")]
    [InlineData("require any without tools")]
    [InlineData("require specific of an unknown tool")]
    [InlineData("guidance with thinking")]
    [InlineData("guidance with logprobs")]
    [InlineData("single call in auto mode")]
    public async Task UnservableGuidedRequestsAreRejected(string name)
    {
        var backend = Backend("x");
        using var client = new Qwen35ChatClient(backend.Model);
        ChatOptions options = name switch
        {
            "json with tools" => new() { Tools = [Weather], ResponseFormat = ChatResponseFormat.Json },
            "json with require any" => new() { Tools = [Weather], ToolMode = ChatToolMode.RequireAny, ResponseFormat = ChatResponseFormat.Json },
            "require any without tools" => new() { ToolMode = ChatToolMode.RequireAny },
            "require specific of an unknown tool" => new() { Tools = [Weather], ToolMode = ChatToolMode.RequireSpecific("send_email") },
            "guidance with thinking" => new() { Tools = [Weather], ToolMode = ChatToolMode.RequireAny, AdditionalProperties = new() { ["enable_thinking"] = true } },
            "guidance with logprobs" => new() { ResponseFormat = ChatResponseFormat.Json, AdditionalProperties = new() { ["logprobs"] = true } },
            "single call in auto mode" => new() { Tools = [Weather], AllowMultipleToolCalls = false },
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
        await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(Question, options, Ct));
        Assert.Empty(backend.Sequences);
    }

    [Fact]
    public async Task UnguidedRequestsCarryNoGuidance()
    {
        var backend = Backend(WeatherCall);
        using var client = new Qwen35ChatClient(backend.Model);
        await client.GetResponseAsync(Question, null, Ct);
        await client.GetResponseAsync(Question, new ChatOptions { Tools = [Weather] }, Ct);
        await client.GetResponseAsync(Question, new ChatOptions { Tools = [Weather], ToolMode = ChatToolMode.Auto, ResponseFormat = ChatResponseFormat.Text }, Ct);
        Assert.All(backend.Sequences, s => Assert.Null(s.Options.Guidance));
    }

    private static List<Qwen35Tool> ToTools(IEnumerable<AIFunction> functions) =>
        [.. functions.Select(f => Qwen35Tool.Function(f.Name, f.Description, f.JsonSchema))];
}
