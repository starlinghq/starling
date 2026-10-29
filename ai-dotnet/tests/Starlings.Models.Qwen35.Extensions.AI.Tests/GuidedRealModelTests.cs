using System.Text.Json;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>
/// The IChatClient on the fp32x model on CPU in a directory with ONNX Runtime GenAI's guidance files added
/// (<see cref="Qwen35GuidedDirectory"/>). Skips without the model or a verified <c>tokenizer_config.json</c>.
/// </summary>
public sealed class Qwen35GuidedChatClientFixture : IDisposable
{
    private readonly Qwen35GuidedDirectory? _directory;

    public Qwen35GuidedChatClientFixture()
    {
        _directory = Qwen35GuidedDirectory.TryCreate(Qwen35Variant.Fp32x, out string reason);
        Reason = reason;
        if (_directory is not null)
        {
            try
            {
                Client = Qwen35ChatClient.Create(_directory.Files);
            }
            catch
            {
                // A constructor that throws gets no Dispose; remove the temporary directory here.
                _directory.Dispose();
                throw;
            }
        }
    }

    public Qwen35ChatClient? Client { get; }

    public string Reason { get; }

    public void Dispose()
    {
        Client?.Dispose();
        _directory?.Dispose();
    }
}

/// <summary>Runs alone, after the parallel classes have released their models: its fixture loads another 3.1 GB model.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GuidedModelTestGroup
{
    public const string Name = "guided model (not parallel)";
}

/// <summary>Guided IChatClient requests on the real model (greedy; outputs are checked against the constraint).</summary>
[Collection(GuidedModelTestGroup.Name)]
public sealed class GuidedRealModelTests(Qwen35GuidedChatClientFixture fixture) : IClassFixture<Qwen35GuidedChatClientFixture>
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

    [Fact]
    public async Task RequireAnyReturnsAFunctionCall()
    {
        var weather = AIFunctionFactory.Create((string city) => $"Sunny in {city}", "get_weather", "Current weather for a city.");
        var time = AIFunctionFactory.Create(() => "12:00", "get_time", "The current time.");
        var response = await Client.GetResponseAsync(
            "What's the weather in Porto?",
            new ChatOptions { Tools = [weather, time], ToolMode = ChatToolMode.RequireAny, MaxOutputTokens = 64 },
            Ct);
        var call = Assert.IsType<FunctionCallContent>(Assert.Single(response.Messages.Single().Contents));
        Assert.Equal("get_weather", call.Name);
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
    }

    [Fact]
    public async Task AJsonSchemaResponseFormatReturnsMatchingJson()
    {
        var schema = JsonDocument.Parse("""{"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": false}""").RootElement;
        var response = await Client.GetResponseAsync(
            "What is 6 times 7? Reply as JSON.",
            new ChatOptions { ResponseFormat = ChatResponseFormat.ForJsonSchema(schema), MaxOutputTokens = 32 },
            Ct);
        using var json = JsonDocument.Parse(response.Text);
        Assert.Equal(JsonValueKind.Number, json.RootElement.GetProperty("answer").ValueKind);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
    }
}
