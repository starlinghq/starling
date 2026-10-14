using System.Text.Json;
using Starlings.TestSupport;
using Starlings.Tokenizers;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>
/// Tool calling through the IChatClient with a scripted native sequence (the real template, tokenizer, decode loop and
/// detokenizer; only the model's token choice is scripted): ChatOptions.Tools reach the prompt, the model's tool-call
/// blocks come back as <see cref="FunctionCallContent"/> with <see cref="ChatFinishReason.ToolCalls"/>, streaming
/// reassembles to the same response without leaking markup, malformed blocks come back as text, and the
/// Microsoft.Extensions.AI function-invocation loop completes.
/// </summary>
public sealed class ToolCallingTests
{
    private const string Call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
    private const string Answer = "It is 18 C and sunny in Paris.";

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static Tokenizer Tokenizer
    {
        get
        {
            var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
            Assert.SkipWhen(tokenizer is null, reason);
            return tokenizer!;
        }
    }

    private static AIFunction WeatherFunction(List<string>? calls = null) => AIFunctionFactory.Create(
        (string city) =>
        {
            calls?.Add(city);
            return $"18 C, sunny in {city}";
        },
        "get_weather",
        "Current weather for a city.");

    private static ChatOptions WithWeather(List<string>? calls = null) => new() { Tools = [WeatherFunction(calls)] };

    private static ChatMessage[] Question => [new(ChatRole.User, "What is the weather in Paris?")];

    // Replies with the completion for the prompt: a tool call until a tool response is in the prompt, then the answer.
    private static ScriptedBackend Backend(Func<string, string> reply)
    {
        var tokenizer = Tokenizer;
        return new ScriptedBackend(tokenizer, prompt =>
            [.. tokenizer.Encode(reply(tokenizer.Decode(prompt, false)), addSpecialTokens: false).Ids, Qwen35ChatModel.ImEndTokenId]);
    }

    private static ScriptedBackend Backend(string completion) => Backend(_ => completion);

    [Fact]
    public async Task ToolsReachThePromptAsTheTypedRenderingOfTheFunctionSchema()
    {
        var backend = Backend("Hello");
        using var client = new Qwen35ChatClient(backend.Model);
        var function = WeatherFunction();
        await client.GetResponseAsync(Question, new ChatOptions { Tools = [function] }, Ct);

        var tool = Qwen35Tool.Function("get_weather", "Current weather for a city.", function.JsonSchema);
        var expected = backend.Model.CreatePrompt([Qwen35Message.User("What is the weather in Paris?")], [tool]);
        Assert.Equal(expected.TokenIds, backend.Single().Prompt);
        Assert.Contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Current weather for a city.\", \"parameters\": {", expected.Text, StringComparison.Ordinal);
        Assert.Contains("\"city\": {\"type\": \"string\"}", expected.Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACallComesBackAsFunctionCallContentWithFinishReasonToolCalls()
    {
        using var client = new Qwen35ChatClient(Backend("I will check.\n\n" + Call).Model);
        var response = await client.GetResponseAsync(Question, WithWeather(), Ct);

        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
        var message = Assert.Single(response.Messages);
        Assert.Equal(2, message.Contents.Count);
        Assert.Equal("I will check.", Assert.IsType<TextContent>(message.Contents[0]).Text);
        var call = Assert.IsType<FunctionCallContent>(message.Contents[1]);
        Assert.Equal("get_weather", call.Name);
        Assert.StartsWith("call_", call.CallId, StringComparison.Ordinal);
        Assert.Equal("Paris", Assert.IsType<JsonElement>(call.Arguments!["city"]).GetString());
        Assert.IsType<Qwen35ToolCall>(call.RawRepresentation);
        Assert.DoesNotContain("<tool_call>", response.Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SeveralCallsInOneTurn()
    {
        using var client = new Qwen35ChatClient(Backend(Call + "\n" + Call.Replace("Paris", "Tokyo", StringComparison.Ordinal)).Model);
        var response = await client.GetResponseAsync(Question, WithWeather(), Ct);
        var calls = response.Messages.Single().Contents.OfType<FunctionCallContent>().ToList();
        Assert.Equal(["Paris", "Tokyo"], calls.Select(c => ((JsonElement)c.Arguments!["city"]!).GetString()));
        Assert.Equal(2, calls.Select(c => c.CallId).Distinct().Count());
        Assert.Equal(2, response.Messages.Single().Contents.Count);
    }

    public static TheoryData<string> Completions() =>
    [
        Call,
        "I will check.\n\n" + Call,
        Call + "\n" + Call.Replace("Paris", "Tokyo", StringComparison.Ordinal),
        "No call: 中文 🙂🧪 and a <tool literal.",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis",
        "<tool_call>\nnot a call\n</tool_call>\n" + Call,
    ];

    [Theory]
    [MemberData(nameof(Completions))]
    public async Task StreamingReassemblesToTheResponseWithoutLeakingMarkup(string completion)
    {
        using var client = new Qwen35ChatClient(Backend(completion).Model);
        var response = await client.GetResponseAsync(Question, WithWeather(), Ct);
        var updates = await ToListAsync(client.GetStreamingResponseAsync(Question, WithWeather(), Ct));
        var combined = updates.ToChatResponse();

        Assert.Equal(response.Text, combined.Text);
        Assert.Equal(response.FinishReason, combined.FinishReason);
        Assert.Equal(response.FinishReason, updates[^1].FinishReason);
        Assert.Equal(Describe(response), Describe(combined));
        Assert.Equal(response.Usage!.OutputTokenCount, combined.Usage!.OutputTokenCount);
        Assert.All(updates.SkipLast(1), u => Assert.NotEmpty(u.Contents));
        if (response.Messages.Single().Contents.OfType<FunctionCallContent>().Any())
        {
            Assert.All(updates, u => Assert.DoesNotContain("<function=", u.Text, StringComparison.Ordinal));
            Assert.Equal(response.Messages.Single().Contents.OfType<FunctionCallContent>().Count(), updates.SelectMany(u => u.Contents).OfType<FunctionCallContent>().Count());
        }
    }

    [Fact]
    public async Task TextIsStreamedBeforeTheCallCompletes()
    {
        using var client = new Qwen35ChatClient(Backend("Let me look that up for you.\n\n" + Call).Model);
        var updates = await ToListAsync(client.GetStreamingResponseAsync(Question, WithWeather(), Ct));
        int firstCall = updates.FindIndex(u => u.Contents.OfType<FunctionCallContent>().Any());
        Assert.True(firstCall > 1, $"call at update {firstCall}");
        Assert.Equal("Let me look that up for you.", string.Concat(updates.Take(firstCall).Select(u => u.Text)));
    }

    [Theory]
    [InlineData("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis")]
    [InlineData("<tool_call>\nget_weather(city=\"Paris\")\n</tool_call>")]
    [InlineData("<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n<parameter=city>\nB\n</parameter>\n</function>\n</tool_call>")]
    public async Task MalformedCallsComeBackAsTextWithoutThrowing(string completion)
    {
        using var client = new Qwen35ChatClient(Backend(completion).Model);
        var response = await client.GetResponseAsync(Question, WithWeather(), Ct);
        Assert.Equal(completion, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Empty(response.Messages.Single().Contents.OfType<FunctionCallContent>());

        var streamed = (await ToListAsync(client.GetStreamingResponseAsync(Question, WithWeather(), Ct))).ToChatResponse();
        Assert.Equal(completion, streamed.Text);
        Assert.Equal(ChatFinishReason.Stop, streamed.FinishReason);
    }

    [Fact]
    public async Task ACallCutByTheTokenLimitIsTextWithFinishReasonLength()
    {
        using var client = new Qwen35ChatClient(Backend(Call).Model);
        var options = WithWeather();
        options.MaxOutputTokens = 8;
        var response = await client.GetResponseAsync(Question, options, Ct);
        Assert.Equal(ChatFinishReason.Length, response.FinishReason);
        Assert.StartsWith("<tool_call>", response.Text, StringComparison.Ordinal);
        Assert.Empty(response.Messages.Single().Contents.OfType<FunctionCallContent>());
    }

    [Fact]
    public async Task FunctionInvocationLoopCompletes()
    {
        var invoked = new List<string>();
        var backend = Backend(prompt => prompt.Contains("<tool_response>", StringComparison.Ordinal) ? Answer : Call);
        using var client = new Qwen35ChatClient(backend.Model);
        using IChatClient pipeline = new ChatClientBuilder(client).UseFunctionInvocation().Build();

        var response = await pipeline.GetResponseAsync(Question, WithWeather(invoked), Ct);

        Assert.Equal(Answer, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Equal(["Paris"], invoked);
        Assert.Equal(2, backend.Sequences.Count);

        // The second request is the template's rendering of the question, the call and its result.
        var tool = Qwen35Tool.Function("get_weather", "Current weather for a city.", WeatherFunction().JsonSchema);
        var expected = backend.Model.CreatePrompt(
            [
                Qwen35Message.User("What is the weather in Paris?"),
                Qwen35Message.Assistant(string.Empty, [Qwen35ToolCall.Create("get_weather", [new("city", JsonSerializer.SerializeToElement("Paris"))])]),
                Qwen35Message.Tool("18 C, sunny in Paris"),
            ],
            [tool]);
        Assert.Equal(expected.TokenIds, backend.Sequences.Last().Prompt);

        // The history the loop returns: the call, its result and the answer.
        Assert.Equal([ChatRole.Assistant, ChatRole.Tool, ChatRole.Assistant], response.Messages.Select(m => m.Role));
        var result = Assert.IsType<FunctionResultContent>(Assert.Single(response.Messages[1].Contents));
        Assert.Equal(response.Messages[0].Contents.OfType<FunctionCallContent>().Single().CallId, result.CallId);
    }

    [Fact]
    public async Task FunctionInvocationLoopCompletesWhenStreaming()
    {
        var invoked = new List<string>();
        var backend = Backend(prompt => prompt.Contains("<tool_response>", StringComparison.Ordinal) ? Answer : "Checking.\n\n" + Call);
        using var client = new Qwen35ChatClient(backend.Model);
        using IChatClient pipeline = new ChatClientBuilder(client).UseFunctionInvocation().Build();

        var updates = await ToListAsync(pipeline.GetStreamingResponseAsync(Question, WithWeather(invoked), Ct));
        var response = updates.ToChatResponse();

        Assert.Equal(["Paris"], invoked);
        Assert.EndsWith(Answer, response.Text, StringComparison.Ordinal);
        Assert.DoesNotContain("<tool_call>", response.Text, StringComparison.Ordinal);
        Assert.Equal(2, backend.Sequences.Count);
    }

    [Fact]
    public async Task SeveralResultsInOneToolMessageRenderAsConsecutiveToolResponses()
    {
        var invoked = new List<string>();
        var backend = Backend(prompt => prompt.Contains("<tool_response>", StringComparison.Ordinal) ? "Done." : Call + "\n" + Call.Replace("Paris", "Tokyo", StringComparison.Ordinal));
        using var client = new Qwen35ChatClient(backend.Model);
        using IChatClient pipeline = new ChatClientBuilder(client).UseFunctionInvocation().Build();

        await pipeline.GetResponseAsync(Question, WithWeather(invoked), Ct);

        Assert.Equal(["Paris", "Tokyo"], invoked);
        string second = backend.Tokenizer.Decode(backend.Sequences.Last().Prompt.AsSpan(), false);
        Assert.EndsWith(
            "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18 C, sunny in Paris\n</tool_response>\n<tool_response>\n18 C, sunny in Tokyo\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            second,
            StringComparison.Ordinal);
    }

    [Fact]
    public async Task WithThinkingTheReasoningStaysTextAndTheCallIsParsed()
    {
        var backend = Backend("The user wants the weather.\n</think>\n\n" + Call);
        using var client = new Qwen35ChatClient(backend.Model);
        var options = WithWeather();
        options.AdditionalProperties = new() { [Qwen35ChatClient.EnableThinkingPropertyName] = true };
        var response = await client.GetResponseAsync(Question, options, Ct);

        Assert.Equal("The user wants the weather.\n</think>", response.Text);
        Assert.Equal("get_weather", response.Messages.Single().Contents.OfType<FunctionCallContent>().Single().Name);
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n", backend.Tokenizer.Decode(backend.Single().Prompt.AsSpan(), false), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ToolModeNoneRendersNoToolsAndParsesNothing()
    {
        var backend = Backend(Call);
        using var client = new Qwen35ChatClient(backend.Model);
        var options = WithWeather();
        options.ToolMode = ChatToolMode.None;
        var response = await client.GetResponseAsync(Question, options, Ct);

        Assert.Equal(backend.Model.CreatePrompt([Qwen35Message.User("What is the weather in Paris?")]).TokenIds, backend.Single().Prompt);
        Assert.Equal(Call, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
    }

    [Fact]
    public async Task ToolModeAutoEqualsNoToolMode()
    {
        var backend = Backend(Call);
        using var client = new Qwen35ChatClient(backend.Model);
        var auto = WithWeather();
        auto.ToolMode = ChatToolMode.Auto;
        await client.GetResponseAsync(Question, auto, Ct);
        await client.GetResponseAsync(Question, WithWeather(), Ct);
        Assert.Equal(backend.Sequences.First().Prompt, backend.Sequences.Last().Prompt);
    }

    [Fact]
    public async Task FunctionResultsOfAnyTypeBecomeToolMessageText()
    {
        var backend = Backend("ok");
        using var client = new Qwen35ChatClient(backend.Model);
        ChatMessage[] history =
        [
            .. Question,
            new(ChatRole.Assistant, [new FunctionCallContent("c1", "get_weather", new Dictionary<string, object?> { ["city"] = "Paris", ["days"] = 2, ["detail"] = new List<string> { "wind" } })]),
            new(ChatRole.Tool, [new FunctionResultContent("c1", new { temp = 18, sky = "sunny" }), new FunctionResultContent("c2", 42), new FunctionResultContent("c3", JsonSerializer.SerializeToElement("text")), new FunctionResultContent("c4", null)]),
        ];
        await client.GetResponseAsync(history, WithWeather(), Ct);
        string prompt = backend.Tokenizer.Decode(backend.Single().Prompt.AsSpan(), false);
        Assert.Contains("<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n<parameter=detail>\n[\"wind\"]\n</parameter>", prompt, StringComparison.Ordinal);
        Assert.Contains("<tool_response>\n{\"temp\":18,\"sky\":\"sunny\"}\n</tool_response>\n<tool_response>\n42\n</tool_response>\n<tool_response>\ntext\n</tool_response>\n<tool_response>\nnull\n</tool_response>", prompt, StringComparison.Ordinal);
    }

    private static string[] Describe(ChatResponse response) =>
        [.. response.Messages.SelectMany(m => m.Contents).Where(c => c is TextContent or FunctionCallContent).Aggregate(new List<string>(), (list, c) =>
        {
            if (c is TextContent t)
            {
                if (list.Count > 0 && list[^1].StartsWith("text:", StringComparison.Ordinal))
                {
                    list[^1] += t.Text;
                }
                else
                {
                    list.Add("text:" + t.Text);
                }
            }
            else
            {
                var f = (FunctionCallContent)c;
                list.Add($"call:{f.Name}:{JsonSerializer.Serialize(f.Arguments)}");
            }

            return list;
        })];

    private static async Task<List<ChatResponseUpdate>> ToListAsync(IAsyncEnumerable<ChatResponseUpdate> updates)
    {
        var list = new List<ChatResponseUpdate>();
        await foreach (var u in updates)
        {
            list.Add(u);
        }

        return list;
    }
}
