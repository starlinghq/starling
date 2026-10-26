using System.Text.Json;
using Starlings.Models.QwenChat;

namespace Starlings.Models.QwenChat.Tests;

/// <summary>The typed tool API and the JSON tool-call parser, without the model.</summary>
public sealed class ToolCallingTests
{
    private const string ParisCall = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    private static readonly QwenChatTool Weather = QwenChatTool.Function("get_weather", "Current weather for a city.",
        JsonDocument.Parse("""{"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}""").RootElement);

    [Fact]
    public void FunctionBuildsTheTransformersForm()
    {
        Assert.Equal("get_weather", Weather.Name);
        Assert.Equal("""{"type":"function","function":{"name":"get_weather","description":"Current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}""",
            Weather.Json.GetRawText());
        Assert.Equal("""{"type":"function","function":{"name":"reset"}}""", QwenChatTool.Function("reset").Json.GetRawText());
    }

    [Theory]
    [InlineData("")]
    [InlineData("a\"b")]
    [InlineData("a\\b")]
    [InlineData("a\nb")]
    public void InvalidNamesAreRejected(string name)
    {
        Assert.Throws<ArgumentException>(() => QwenChatTool.Function(name));
        var call = new QwenChatToolCall(name, JsonDocument.Parse("{}").RootElement);
        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Render(QwenChatProfile.Qwen25,
            [new(QwenChatRole.User, "Q"), new(QwenChatRole.Assistant, "") { ToolCalls = [call] }]));
    }

    [Fact]
    public void ToolCallsOnlyOnAssistantMessages()
    {
        var call = QwenChatToolCall.Create("get_weather", [new("city", JsonDocument.Parse("\"Paris\"").RootElement)]);
        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Render(QwenChatProfile.Qwen3,
            [new(QwenChatRole.User, "Q") { ToolCalls = [call] }]));
        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Render(QwenChatProfile.Qwen3,
            [new(QwenChatRole.Tool, "r", "reason")]));
    }

    [Fact]
    public void DuplicateKeysInToolDataAreRejected()
    {
        var tool = QwenChatTool.FromJson(JsonDocument.Parse("""{"name": "f", "parameters": {"a": 1, "a": 2}}""").RootElement);
        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Render(QwenChatProfile.Qwen25, [new(QwenChatRole.User, "Q")], [tool]));
    }

    [Fact]
    public void LoneSurrogatesInToolDataAreRejected()
    {
        var key = QwenChatTool.FromJson(JsonDocument.Parse("""{"name": "f", "parameters": {"\ud800": 1}}""").RootElement);
        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Render(QwenChatProfile.Qwen25, [new(QwenChatRole.User, "Q")], [key]));
        var value = QwenChatTool.FromJson(JsonDocument.Parse("""{"name": "f", "description": "\ud800"}""").RootElement);
        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Render(QwenChatProfile.Qwen25, [new(QwenChatRole.User, "Q")], [value]));
        Assert.Throws<ArgumentException>(() => QwenChatTool.FromJson(JsonDocument.Parse("""{"name": "f\ud800"}""").RootElement));
    }

    [Fact]
    public void ParsesACallWithTextBefore()
    {
        QwenChatParsedOutput parsed = QwenChatToolCallParser.Parse("Let me check.\n" + ParisCall);
        Assert.Equal("Let me check.", parsed.Text);
        QwenChatToolCall call = Assert.Single(parsed.ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("""{"city": "Paris"}""", call.Arguments.GetRawText());
        Assert.Equal(2, parsed.Parts.Count);
    }

    [Fact]
    public void WhitespaceAroundAndBetweenCallsIsDropped()
    {
        QwenChatParsedOutput parsed = QwenChatToolCallParser.Parse("\n" + ParisCall + "\n\n" + ParisCall + "\n");
        Assert.Equal(2, parsed.ToolCalls.Count);
        Assert.Equal(2, parsed.Parts.Count);
        Assert.Equal(string.Empty, parsed.Text);
    }

    [Fact]
    public void TextAfterACallIsKept()
    {
        QwenChatParsedOutput parsed = QwenChatToolCallParser.Parse(ParisCall + "\nDone.");
        Assert.Equal("\nDone.", parsed.Text);
    }

    [Fact]
    public void NoCallIsPlainText()
    {
        Assert.Equal("Paris is sunny.  ", QwenChatToolCallParser.Parse("Paris is sunny.  ").Text);
        Assert.Empty(QwenChatToolCallParser.Parse("a <tool_ b").ToolCalls);
    }

    [Theory]
    [InlineData("<tool_call>\n{\"name\": \"f\", \"arguments\": {}\n</tool_call>")] // unbalanced
    [InlineData("<tool_call>\n{\"name\": \"f\"}\n</tool_call>")] // no arguments
    [InlineData("<tool_call>\n{\"name\": \"f\", \"arguments\": \"{}\"}\n</tool_call>")] // string arguments
    [InlineData("<tool_call>\n{\"name\": \"f\", \"arguments\": {}, \"id\": 1}\n</tool_call>")] // extra member
    [InlineData("<tool_call>\n{\"name\": \"f\", \"name\": \"g\", \"arguments\": {}}\n</tool_call>")] // repeated member
    [InlineData("<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1, \"a\": 2}}\n</tool_call>")] // repeated argument
    [InlineData("<tool_call>\n{\"name\": \"\", \"arguments\": {}}\n</tool_call>")] // empty name
    [InlineData("<tool_call>\n{\"name\": \"a\\\"b\", \"arguments\": {}}\n</tool_call>")] // name with a quote
    [InlineData("<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": \"\\ud800\"}}\n</tool_call>")] // lone surrogate
    [InlineData("<tool_call>\n{'name': 'f', 'arguments': {}}\n</tool_call>")] // not JSON
    [InlineData("<tool_call>\n{\"name\": \"f\", \"arguments\": {},}\n</tool_call>")] // trailing comma
    [InlineData("<tool_call>\n[\"f\"]\n</tool_call>")] // not an object
    [InlineData("<tool_call>\n{\"name\": \"f\", \"arguments\": {}}")] // unfinished
    public void MalformedBlocksStayTextVerbatim(string completion)
    {
        QwenChatParsedOutput parsed = QwenChatToolCallParser.Parse(completion);
        Assert.Empty(parsed.ToolCalls);
        Assert.Equal(completion, parsed.Text);
    }

    [Fact]
    public void ArgumentsKeepTheirJsonTypes()
    {
        QwenChatToolCall call = Assert.Single(QwenChatToolCallParser.Parse(
            "<tool_call>\n{\"arguments\": {\"n\": 3, \"x\": 0.5, \"b\": true, \"z\": null, \"l\": [1, \"a\"], \"o\": {\"k\": \"東京\"}}, \"name\": \"f\"}\n</tool_call>").ToolCalls);
        Assert.Equal("f", call.Name);
        Assert.Equal(JsonValueKind.Number, call.Arguments.GetProperty("n").ValueKind);
        Assert.Equal(JsonValueKind.True, call.Arguments.GetProperty("b").ValueKind);
        Assert.Equal(JsonValueKind.Null, call.Arguments.GetProperty("z").ValueKind);
        Assert.Equal("東京", call.Arguments.GetProperty("o").GetProperty("k").GetString());
    }

    [Fact]
    public void ClosingTagInsideAStringDoesNotEndTheBlock()
    {
        QwenChatToolCall call = Assert.Single(QwenChatToolCallParser.Parse(
            "<tool_call>\n{\"name\": \"echo\", \"arguments\": {\"text\": \"a \\\" </tool_call> b\"}}\n</tool_call>").ToolCalls);
        Assert.Equal("a \" </tool_call> b", call.Arguments.GetProperty("text").GetString());

        // An unterminated string keeps the block open; the completion is then text.
        string open = "<tool_call>\n{\"name\": \"echo\", \"arguments\": {\"text\": \"a}}\n</tool_call>";
        Assert.Equal(open, QwenChatToolCallParser.Parse(open).Text);
    }

    [Fact]
    public void StreamingHoldsBackOnlyAPossibleTagStart()
    {
        var parser = new QwenChatToolCallParser();
        Assert.Equal("Checking", Assert.Single(parser.Append("Checking\n<tool")).Text);
        Assert.Empty(parser.Append("_call>\n{\"name\": \"get_weather\", "));
        Assert.Empty(parser.Append("\"arguments\": {\"city\": \"Paris\"}}\n</tool_"));
        QwenChatOutputPart call = Assert.Single(parser.Append("call>"));
        Assert.Equal("get_weather", call.ToolCall!.Name);
        Assert.Empty(parser.Complete());
        Assert.Throws<InvalidOperationException>(() => parser.Append("x"));
        Assert.Empty(parser.Complete());
    }

    [Fact]
    public void EverySplitOfACompletionGivesTheWholeParse()
    {
        string completion = "Two calls:\n" + ParisCall + "\n" + ParisCall.Replace("Paris", "Tokyo", StringComparison.Ordinal) +
            "\n<tool_call>\n{bad}\n</tool_call> tail <tool_c";
        string expected = TemplateCorpusTests.Describe(QwenChatToolCallParser.Parse(completion).Parts);
        for (int split = 0; split <= completion.Length; split++)
        {
            var parser = new QwenChatToolCallParser();
            var parts = new List<QwenChatOutputPart>(parser.Append(completion[..split]));
            parts.AddRange(parser.Append(completion[split..]));
            parts.AddRange(parser.Complete());
            Assert.Equal(expected, TemplateCorpusTests.Describe(TemplateCorpusTests.Merge(parts)));
        }
    }

    [Fact]
    public void ParsedOutputBecomesTheNextAssistantMessage()
    {
        QwenChatMessage message = QwenChatToolCallParser.Parse("Checking.\n" + ParisCall).ToAssistantMessage();
        Assert.Equal(QwenChatRole.Assistant, message.Role);
        Assert.Equal("Checking.", message.Content);
        Assert.Single(message.ToolCalls!);
        Assert.Null(QwenChatToolCallParser.Parse("No call.").ToAssistantMessage().ToolCalls);

        string text = QwenChatTemplate.Render(QwenChatProfile.Qwen25,
            [new(QwenChatRole.User, "Weather in Paris?"), message, new(QwenChatRole.Tool, "18 C")], [Weather]);
        Assert.Contains("<|im_start|>assistant\nChecking.\n" + ParisCall + "<|im_end|>\n<|im_start|>user\n<tool_response>\n18 C\n</tool_response><|im_end|>\n",
            text, StringComparison.Ordinal);
    }
}
