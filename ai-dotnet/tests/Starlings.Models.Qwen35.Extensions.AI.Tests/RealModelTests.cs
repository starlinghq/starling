using System.Globalization;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>Loads the fp32x bundle on CPU once (skips without it: not in Git, 3.1 GB).</summary>
public sealed class Qwen35ChatClientFixture : IDisposable
{
    public Qwen35ChatClientFixture()
    {
        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Fp32x);
        Reason = reason;
        if (files is not null)
        {
            Client = Qwen35ChatClient.Create(files);
        }
    }

    public Qwen35ChatClient? Client { get; }

    public string Reason { get; }

    public void Dispose() => Client?.Dispose();
}

/// <summary>
/// The IChatClient on the fp32x model on CPU against the transformers fp32 reference (prompts-v1). Completion length
/// per prompt: <c>$STARLING_QWEN35_CHAT_TOKENS</c>, default 32 (the fixture has 128; 32 keeps the run near
/// three minutes on a 6-core CPU).
/// </summary>
public sealed class RealModelTests(Qwen35ChatClientFixture fixture) : IClassFixture<Qwen35ChatClientFixture>
{
    private Qwen35ChatClient Client
    {
        get
        {
            Assert.SkipWhen(fixture.Client is null, fixture.Reason);
            return fixture.Client!;
        }
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static int Tokens => Environment.GetEnvironmentVariable("STARLING_QWEN35_CHAT_TOKENS") is { Length: > 0 } v
        ? int.Parse(v, CultureInfo.InvariantCulture)
        : 32;

    private static GenerationCase Case(string id) => Qwen35TestAssets.LoadCases().Single(c => c.Id == id);

    public static TheoryData<string> TextCaseIds() => [.. ChatClientModelChecks.TextCases().Select(c => c.Id)];

    [Theory]
    [MemberData(nameof(TextCaseIds))]
    public Task StreamingEqualsResponseAndTheReference(string id) =>
        ChatClientModelChecks.StreamingEqualsResponseAndReference(Client, Client.Model, Case(id), Math.Min(Tokens, Case(id).MaxNewTokens), Ct);

    [Fact]
    public async Task CancellationBetweenStepsThenReuse()
    {
        var latency = await ChatClientModelChecks.CancelBetweenStepsThenReuse(Client, Client.Model, Case("en-list"), Case("en-short"), 3, Ct);
        TestContext.Current.TestOutputHelper?.WriteLine($"cpu cancel between steps: {latency.TotalMilliseconds:F1} ms from Cancel() to OperationCanceledException");
        Assert.True(latency < TimeSpan.FromSeconds(5), $"{latency}");
    }

    [Fact]
    public async Task CancellationDuringChunkedPrefillThenReuse()
    {
        using var chunked = new Qwen35ChatClient(Client.Model, new Qwen35ChatClientOptions { Defaults = new Qwen35ChatOptions { PrefillChunkTokens = 64 } });
        var (latency, total) = await ChatClientModelChecks.CancelDuringPrefillThenReuse(chunked, Client.Model, Case("long-license"), Case("en-short"), TimeSpan.FromSeconds(1), Ct);
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"cpu cancel during prefill (64-token chunks, 1,666-token prompt, cancelled 1 s in): {latency.TotalMilliseconds:F1} ms from Cancel() to OperationCanceledException, call {total.TotalMilliseconds:F0} ms");
        Assert.True(latency < TimeSpan.FromSeconds(5), $"{latency}");
    }

    /// <summary>
    /// The prompts-v1 tools prompt (one get_weather tool) through ChatOptions.Tools: the prompt is the reference prompt,
    /// the greedy completion is the reference completion (transformers fp32: a get_weather call for Paris), and it comes
    /// back as one FunctionCallContent with finish reason ToolCalls.
    /// </summary>
    [Fact]
    public async Task ToolsPromptGivesTheReferenceCallAsFunctionCallContent()
    {
        var c = Case("tools");
        var tool = c.Tools.Single();
        var function = tool.Json.GetProperty("function");
        var options = new ChatOptions
        {
            MaxOutputTokens = c.MaxNewTokens,
            Tools = [AIFunctionFactory.CreateDeclaration(tool.Name, function.GetProperty("description").GetString(), tool.Parameters)],
        };
        Assert.Equal(c.PromptIds, Client.Model.CreatePrompt(c.Messages!, c.Tools).TokenIds);

        var response = await Client.GetResponseAsync(ChatClientModelChecks.ToChatMessages(c), options, Ct);
        var raw = Assert.IsType<Starlings.Generation.OnnxGenAI.GenerationResult>(response.RawRepresentation);
        Assert.Equal(c.PromptIds.Length, response.Usage!.InputTokenCount);
        Assert.Equal(c.ContentIds, raw.TokenIds);
        var call = Assert.IsType<FunctionCallContent>(Assert.Single(response.Messages.Single().Contents));
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", Assert.IsType<System.Text.Json.JsonElement>(call.Arguments!["city"]).GetString());
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);

        var streamed = new List<ChatResponseUpdate>();
        await foreach (var u in Client.GetStreamingResponseAsync(ChatClientModelChecks.ToChatMessages(c), options, Ct))
        {
            streamed.Add(u);
        }

        var combined = streamed.ToChatResponse();
        Assert.Equal(string.Empty, combined.Text);
        Assert.Equal("Paris", ((System.Text.Json.JsonElement)combined.Messages.Single().Contents.OfType<FunctionCallContent>().Single().Arguments!["city"]!).GetString());
        Assert.Equal(ChatFinishReason.ToolCalls, combined.FinishReason);
    }

    /// <summary>
    /// The Microsoft.Extensions.AI function-invocation loop on the real model (greedy): the model calls get_weather, the
    /// middleware invokes it and sends the result back, and the model answers in text. This records what the 0.8B model
    /// did on this one prompt; it does not qualify its tool choice in general.
    /// </summary>
    [Fact]
    public async Task FunctionInvocationLoopCompletesOnTheRealModel()
    {
        var invoked = new List<string>();
        var getWeather = AIFunctionFactory.Create(
            (string city) =>
            {
                invoked.Add(city);
                return $"18 C and sunny in {city}.";
            },
            "get_weather",
            "Get the current weather for a city.");
        using IChatClient pipeline = new ChatClientBuilder(Client).UseFunctionInvocation().Build();
        var response = await pipeline.GetResponseAsync(
            [new ChatMessage(ChatRole.User, "What is the weather in Paris today?")],
            new ChatOptions { Tools = [getWeather], MaxOutputTokens = 128 },
            Ct);

        var output = TestContext.Current.TestOutputHelper;
        output?.WriteLine($"invoked: [{string.Join(", ", invoked)}]");
        foreach (var m in response.Messages)
        {
            output?.WriteLine($"{m.Role}: {string.Join(" | ", m.Contents.Select(Describe))}");
        }

        Assert.Equal(["Paris"], invoked);
        Assert.Equal([ChatRole.Assistant, ChatRole.Tool, ChatRole.Assistant], response.Messages.Select(m => m.Role));
        Assert.Empty(response.Messages[^1].Contents.OfType<FunctionCallContent>());
        Assert.False(string.IsNullOrWhiteSpace(response.Messages[^1].Text));
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
    }

    private static string Describe(AIContent content) => content switch
    {
        FunctionCallContent f => $"call {f.Name}({string.Join(", ", f.Arguments?.Select(a => $"{a.Key}={a.Value}") ?? [])})",
        FunctionResultContent r => $"result {r.Result}",
        TextContent t => $"text {t.Text}",
        _ => content.GetType().Name,
    };

    [Fact]
    public async Task ConcurrentResponsesEqualSequential()
    {
        Assert.False(Client.Model.Info.SerializesNativeCalls);
        await ChatClientModelChecks.ConcurrentResponsesEqualSequential(Client, Client.Model, [Case("zh"), Case("code-py"), Case("ja"), Case("en-math")], 12, rounds: 2, Ct);
    }
}
