using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Models.Qwen35.Extensions.AI;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests.Cuda;

/// <summary>
/// Tool calling through the IChatClient adapter on a CUDA model (capability <c>tool_calling</c> of the CUDA profiles):
/// the same two real-model checks as <c>Starlings.Models.Qwen35.Extensions.AI.Tests/RealModelTests</c> on fp32x CPU.
/// They record what the 0.8B model did on one prompt; they do not qualify its tool choice in general.
/// </summary>
internal static class CudaToolCallingChecks
{
    /// <summary>
    /// The prompts-v1 tools prompt with its one get_weather tool: the reference call as <see cref="FunctionCallContent"/>,
    /// non-streamed and streamed. With <paramref name="requireReferenceIds"/> the completion ids must equal the
    /// transformers fp32 reference (fp32x); otherwise they are reported (fp16x, whose greedy rule allows divergence).
    /// </summary>
    public static async Task ToolsPromptGivesTheReferenceCall(Qwen35ChatClient client, GenerationCase c, bool requireReferenceIds, CancellationToken ct)
    {
        var tool = c.Tools.Single();
        var function = tool.Json.GetProperty("function");
        var options = new ChatOptions
        {
            MaxOutputTokens = c.MaxNewTokens,
            Tools = [AIFunctionFactory.CreateDeclaration(tool.Name, function.GetProperty("description").GetString(), tool.Parameters)],
        };
        Assert.Equal(c.PromptIds, client.Model.CreatePrompt(c.Messages!, c.Tools).TokenIds);

        var response = await client.GetResponseAsync(ChatClientModelChecks.ToChatMessages(c), options, ct);
        var raw = Assert.IsType<Starlings.Generation.GenerationResult>(response.RawRepresentation);
        bool idsEqual = c.ContentIds.SequenceEqual(raw.TokenIds);
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"{client.Model.Info.Profile} on {client.Model.Info.Provider}: {raw.TokenIds.Length} completion ids, equal to the reference: {idsEqual}");
        if (requireReferenceIds)
        {
            Assert.Equal(c.ContentIds, raw.TokenIds);
        }

        Assert.Equal(c.PromptIds.Length, response.Usage!.InputTokenCount);
        var call = Assert.IsType<FunctionCallContent>(Assert.Single(response.Messages.Single().Contents));
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", Assert.IsType<JsonElement>(call.Arguments!["city"]).GetString());
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);

        var streamed = new List<ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingResponseAsync(ChatClientModelChecks.ToChatMessages(c), options, ct))
        {
            streamed.Add(u);
        }

        var combined = streamed.ToChatResponse();
        Assert.Equal(string.Empty, combined.Text);
        Assert.Equal("Paris", ((JsonElement)combined.Messages.Single().Contents.OfType<FunctionCallContent>().Single().Arguments!["city"]!).GetString());
        Assert.Equal(ChatFinishReason.ToolCalls, combined.FinishReason);
    }

    /// <summary>
    /// The Microsoft.Extensions.AI function-invocation loop: the model calls get_weather, the middleware invokes it and
    /// sends the result back, and the model answers in text.
    /// </summary>
    public static async Task FunctionInvocationLoopCompletes(Qwen35ChatClient client, CancellationToken ct)
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
        // Not disposed: disposing the pipeline disposes the inner client, which the class fixture shares with the other tests.
        IChatClient pipeline = new ChatClientBuilder(client).UseFunctionInvocation().Build();
        var response = await pipeline.GetResponseAsync(
            [new ChatMessage(ChatRole.User, "What is the weather in Paris today?")],
            new ChatOptions { Tools = [getWeather], MaxOutputTokens = 128 },
            ct);

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
}
