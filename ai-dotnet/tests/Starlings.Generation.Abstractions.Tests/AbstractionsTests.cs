using System.Reflection;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Starlings.Generation.Tests;

public sealed class DependencyTests
{
    [Fact]
    public void ReferencesNoRuntimeAssembly()
    {
        // The package exists so that a runtime without ONNX Runtime (for example a CUDA worker) can share these types.
        AssemblyName[] references = typeof(GenerationOptions).Assembly.GetReferencedAssemblies();
        Assert.DoesNotContain(references, r => r.Name!.StartsWith("Microsoft.ML.OnnxRuntime", StringComparison.Ordinal));
        Assert.DoesNotContain(references, r => r.Name!.StartsWith("Starlings.Generation.OnnxGenAI", StringComparison.Ordinal));
        Assert.DoesNotContain(references, r => r.Name!.StartsWith("Starlings.Runtime", StringComparison.Ordinal));
    }
}

public sealed class OptionsTests
{
    [Fact]
    public void ValidateIsPublicAndRejectsAsTheDecodeLoopDoes()
    {
        new GenerationOptions().Validate();
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenerationOptions { MaxNewTokens = 0 }.Validate());
        Assert.Throws<ArgumentException>(() => new GenerationOptions { StopSequences = [""] }.Validate());
        Assert.Throws<ArgumentException>(() => new GenerationOptions { TopLogprobs = 2 }.Validate());
        Assert.Throws<ArgumentException>(() => new GenerationOptions { ReturnLogprobs = true, Guidance = GenerationGuidance.Regex("a") }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenerationOptions { PrefillChunkTokens = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new SamplingOptions { Seed = 1, Temperature = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenerationOptions { Sampling = new SamplingOptions { Seed = 1, TopP = 0 } }.Validate());
    }
}

public sealed class ChatTypeTests
{
    [Fact]
    public void FactoriesSetRoleAndContent()
    {
        Assert.Equal(new ChatTurn(ChatTurnRole.System, "s"), ChatTurn.System("s"));
        Assert.Equal(new ChatTurn(ChatTurnRole.User, "u"), ChatTurn.User("u"));
        Assert.Equal(new ChatTurn(ChatTurnRole.Assistant, "a", "r"), ChatTurn.Assistant("a", "r"));
        Assert.Equal(new ChatTurn(ChatTurnRole.Tool, "t"), ChatTurn.Tool("t"));
        var call = ToolCall.Create("f", [new("x", JsonDocument.Parse("1").RootElement)]);
        var turn = ChatTurn.Assistant("", [call]);
        Assert.Same(call, Assert.Single(turn.ToolCalls!));
    }

    [Fact]
    public void ToolCallKeepsArgumentOrder()
    {
        var call = ToolCall.Create("f", [new("b", JsonDocument.Parse("2").RootElement), new("a", JsonDocument.Parse("\"x\"").RootElement)]);
        Assert.Equal("f", call.Name);
        Assert.Equal("""{"b":2,"a":"x"}""", call.Arguments.GetRawText());
    }

    [Fact]
    public void ToolDefinitionFunctionWritesTheOpenAiShape()
    {
        var schema = JsonDocument.Parse("""{"type": "object"}""").RootElement;
        var tool = ToolDefinition.Function("get_weather", "Weather.", schema);
        Assert.Equal("get_weather", tool.Name);
        Assert.Equal("""{"type":"function","function":{"name":"get_weather","description":"Weather.","parameters":{"type":"object"}}}""", tool.Json.GetRawText());
        Assert.Equal("object", tool.Parameters.GetProperty("type").GetString());
        Assert.Equal(JsonValueKind.Undefined, ToolDefinition.Function("f").Parameters.ValueKind);
    }

    [Fact]
    public void ToolDefinitionFromJsonAcceptsBothForms()
    {
        Assert.Equal("f", ToolDefinition.FromJson(JsonDocument.Parse("""{"name": "f"}""").RootElement).Name);
        Assert.Equal("g", ToolDefinition.FromJson(JsonDocument.Parse("""{"type": "function", "function": {"name": "g"}}""").RootElement).Name);
        Assert.Throws<ArgumentException>(() => ToolDefinition.FromJson(JsonDocument.Parse("[]").RootElement));
        Assert.Throws<ArgumentException>(() => ToolDefinition.FromJson(JsonDocument.Parse("""{"name": ""}""").RootElement));
        Assert.Throws<ArgumentException>(() => ToolDefinition.FromJson(JsonDocument.Parse("""{"name": "\ud800"}""").RootElement));
        Assert.Throws<ArgumentException>(() => ToolDefinition.Function(""));
    }

    [Fact]
    public void ParsedCompletionMergesTextAndDropsEmptyParts()
    {
        var call = new ToolCall("f", default);
        var parser = new ScriptedParser(
            [CompletionPart.FromText("a"), CompletionPart.FromText(""), CompletionPart.FromText("b"), CompletionPart.FromToolCall(call)],
            [CompletionPart.FromText("c")]);
        var parsed = ParsedCompletion.Parse(parser, "ignored");
        Assert.Equal([CompletionPart.FromText("ab"), CompletionPart.FromToolCall(call), CompletionPart.FromText("c")], parsed.Parts);
        Assert.Equal("abc", parsed.Text);
        Assert.Same(call, Assert.Single(parsed.ToolCalls));
        var turn = parsed.ToAssistantTurn();
        Assert.Equal(ChatTurnRole.Assistant, turn.Role);
        Assert.Equal("abc", turn.Content);
        Assert.Null(new ParsedCompletion([CompletionPart.FromText("x")]).ToAssistantTurn().ToolCalls);
    }

    private sealed class ScriptedParser(IReadOnlyList<CompletionPart> append, IReadOnlyList<CompletionPart> complete) : IToolCallParser
    {
        public IReadOnlyList<CompletionPart> Append(string delta) => append;

        public IReadOnlyList<CompletionPart> Complete() => complete;
    }
}

public sealed class PythonJsonTests
{
    [Theory]
    [InlineData("""{"a": 1, "b": [true, null, "x"]}""", """{"a": 1, "b": [true, null, "x"]}""")]
    [InlineData("""{"k":1.0,"e":1e-5,"big":123456789012345678901234567890}""", """{"k": 1.0, "e": 1e-05, "big": 123456789012345678901234567890}""")]
    [InlineData("\"caf\\u00e9 \\n \\u0001\"", "\"café \\n \\u0001\"")]
    [InlineData("1e400", "Infinity")]
    public void DumpsAsPython(string json, string expected) => Assert.Equal(expected, PythonJson.Dumps(JsonDocument.Parse(json).RootElement));

    [Theory]
    [InlineData("\"text\"", "text")]
    [InlineData("true", "True")]
    [InlineData("false", "False")]
    [InlineData("null", "None")]
    [InlineData("2.50", "2.5")]
    [InlineData("1e400", "inf")]
    [InlineData("""{"a": 1}""", """{"a": 1}""")]
    public void StrAsPython(string json, string expected) => Assert.Equal(expected, PythonJson.Str(JsonDocument.Parse(json).RootElement));

    [Theory]
    [InlineData(0.1, "0.1")]
    [InlineData(1e16, "1e+16")]
    [InlineData(1e-5, "1e-05")]
    [InlineData(123.0, "123.0")]
    [InlineData(-0.0, "-0.0")]
    public void FloatRepr(double value, string expected) => Assert.Equal(expected, PythonJson.FloatRepr(value));

    [Fact]
    public void RejectsWhatPythonCannotHold()
    {
        Assert.Throws<PythonJsonException>(() => PythonJson.Dumps(JsonDocument.Parse("""{"a": 1, "a": 2}""").RootElement));
        Assert.Throws<PythonJsonException>(() => PythonJson.Dumps(JsonDocument.Parse("""{"\ud800": 1}""").RootElement));
        Assert.Throws<PythonJsonException>(() => PythonJson.Dumps(JsonDocument.Parse("\"\\ud800\"").RootElement));
        Assert.Throws<PythonJsonException>(() => PythonJson.Dumps(default));
        Assert.Throws<PythonJsonException>(() => PythonJson.Str(default));
        Assert.Throws<PythonJsonException>(() => PythonJson.CheckedString("\ud800"));
        Assert.Equal("\ud83d\ude00", PythonJson.CheckedString("\ud83d\ude00"));
        var sb = new StringBuilder("x");
        PythonJson.Dumps(sb, JsonDocument.Parse("[1]").RootElement);
        Assert.Equal("x[1]", sb.ToString());
    }
}

public sealed class ChatMappingTests
{
    [Fact]
    public void FinishReasonsAndUsage()
    {
        Assert.Equal(ChatFinishReason.Stop, GenerationFinishReason.Stop.ToChatFinishReason());
        Assert.Equal(ChatFinishReason.Stop, GenerationFinishReason.StopSequence.ToChatFinishReason());
        Assert.Equal(ChatFinishReason.ToolCalls, GenerationFinishReason.Stop.ToChatFinishReason(toolCalls: true));
        Assert.Equal(ChatFinishReason.Length, GenerationFinishReason.Length.ToChatFinishReason(toolCalls: true));
        Assert.Throws<ArgumentOutOfRangeException>(() => ((GenerationFinishReason)99).ToChatFinishReason());
        var usage = new GenerationUsage(3, 4).ToUsageDetails();
        Assert.Equal((3L, 4L, 7L), (usage.InputTokenCount, usage.OutputTokenCount, usage.TotalTokenCount));
        Assert.Equal(ChatRole.Tool, ChatTurnRole.Tool.ToChatRole());
        Assert.Equal(ChatRole.System, ChatTurnRole.System.ToChatRole());
    }
}

public sealed class ModelProfileTests
{
    private static ModelProfile Sample() => new()
    {
        Id = "m",
        Repository = "o/m",
        Revision = new string('a', 40),
        TokenizerSha256 = new string('b', 64),
        ChatTemplateSha256 = new string('c', 64),
        EosTokenIds = [1, 2],
        VocabularySize = 10,
        HiddenSize = 8,
        Layers = 2,
        HeadSize = 4,
        ContextLength = 16,
        License = "Apache-2.0",
    };

    [Fact]
    public void EqualityComparesEosIdsByValue()
    {
        Assert.Equal(Sample(), Sample());
        Assert.NotEqual(Sample(), Sample() with { EosTokenIds = [1] });
        Assert.Equal($"o/m@{new string('a', 40)}", Sample().Checkpoint);
    }
}
