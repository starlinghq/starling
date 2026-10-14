using System.Globalization;
using System.Text.Json;
using Starlings.Models.Qwen35;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>
/// The typed tool-calling API without a model: the parser of the template's tool-call format (whole and streamed,
/// malformed input, argument casting), the round trip of a parsed completion through the template, and the template's
/// explicit failures on tool data it cannot render. The renderer itself is checked against upstream renderings in
/// <see cref="TemplateCorpusTests"/>.
/// </summary>
public sealed class ToolCallingTests
{
    private const string WeatherCall = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    private static readonly Qwen35Tool Weather = Qwen35Tool.Function(
        "get_weather",
        "Current weather for a city.",
        Json("""{"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}"""));

    private static readonly Qwen35Tool Typed = Qwen35Tool.Function(
        "typed",
        null,
        Json("""
        {"type": "object", "properties": {
          "count": {"type": "integer"}, "ratio": {"type": "number"}, "whole": {"type": "number"}, "flag": {"type": "boolean"},
          "nothing": {"type": "null"}, "obj": {"type": "object"}, "list": {"type": "array"}, "text": {"type": "string"},
          "either": {"anyOf": [{"type": "integer"}, {"type": "string"}]}, "maybe": {"type": "integer", "nullable": true},
          "bad_int": {"type": "integer"}, "string_first": {"type": ["string", "integer"]}
        }}
        """));

    public static TheoryData<string> Completions() =>
    [
        WeatherCall,
        "Let me check.\n\n" + WeatherCall,
        WeatherCall + "\n" + WeatherCall.Replace("Paris", "Tokyo", StringComparison.Ordinal),
        "No tools needed: the answer is 4.",
        "Text with < and <tool and <tool_call but no block, and </tool_call> alone.",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis",
        "<tool_call>\nnot a function\n</tool_call> then text",
        WeatherCall + " trailing text\n",
        "  leading spaces then " + WeatherCall + "\n\n",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=typed>\n<parameter=count>\n007\n</parameter>\n<parameter=ratio>\n2.5\n</parameter>\n</function>\n</tool_call>",
        "中文 🙂 " + WeatherCall + " 🙂",
    ];

    [Fact]
    public void ParsesOneCall()
    {
        var parsed = Qwen35ToolCallParser.Parse(WeatherCall, [Weather], enableThinking: false);
        var call = Assert.Single(parsed.ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("""{"city":"Paris"}""", call.Arguments.GetRawText());
        Assert.Equal(string.Empty, parsed.Text);
        Assert.Single(parsed.Parts);
    }

    [Fact]
    public void ParsesTextAndSeveralCallsAndDropsSeparatingWhitespace()
    {
        string completion = "Let me check both.\n\n" + WeatherCall + "\n" + WeatherCall.Replace("Paris", "Tokyo", StringComparison.Ordinal) + "\n";
        var parsed = Qwen35ToolCallParser.Parse(completion, [Weather], enableThinking: false);
        Assert.Equal(3, parsed.Parts.Count);
        Assert.Equal("Let me check both.", parsed.Parts[0].Text);
        Assert.Equal(["Paris", "Tokyo"], parsed.ToolCalls.Select(c => c.Arguments.GetProperty("city").GetString()));
    }

    [Fact]
    public void ToleratesWhitespaceVariationsInsideABlock()
    {
        string loose = "<tool_call><function=get_weather>  <parameter=city>Paris</parameter>\n\n</function>  </tool_call>";
        var call = Assert.Single(Qwen35ToolCallParser.Parse(loose, [Weather], false).ToolCalls);
        Assert.Equal("Paris", call.Arguments.GetProperty("city").GetString());
    }

    [Fact]
    public void ValuesKeepInnerAndTrailingWhitespaceExceptTheTemplatesLineFeeds()
    {
        string completion = "<tool_call>\n<function=get_weather>\n<parameter=city>\n  New\nYork  \n\n</parameter>\n</function>\n</tool_call>";
        var call = Assert.Single(Qwen35ToolCallParser.Parse(completion, [Weather], false).ToolCalls);
        Assert.Equal("  New\nYork  \n", call.Arguments.GetProperty("city").GetString());
    }

    [Fact]
    public void ArgumentsAreCastWithTheToolSchemaAsTransformersDoes()
    {
        string Param(string name, string value) => $"<parameter={name}>\n{value}\n</parameter>\n";
        string completion = "<tool_call>\n<function=typed>\n" +
            Param("count", " 007 ") + Param("ratio", "2.5") + Param("whole", "1e3") + Param("flag", "True") + Param("nothing", "None") +
            Param("obj", """{"a": [1, null]}""") + Param("list", "[1, 2]") + Param("text", "42") + Param("either", "x") +
            Param("maybe", "null") + Param("bad_int", "twelve") + Param("string_first", "7") + Param("undeclared", "3") +
            "</function>\n</tool_call>";
        var call = Assert.Single(Qwen35ToolCallParser.Parse(completion, [Typed], false).ToolCalls);
        Assert.Equal(
            """{"count":7,"ratio":2.5,"whole":1000,"flag":true,"nothing":null,"obj":{"a":[1,null]},"list":[1,2],"text":"42","either":"x","maybe":null,"bad_int":"twelve","string_first":7,"undeclared":"3"}""",
            call.Arguments.GetRawText());
    }

    [Fact]
    public void UndeclaredToolsKeepStringArguments()
    {
        string completion = "<tool_call>\n<function=other>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>";
        var call = Assert.Single(Qwen35ToolCallParser.Parse(completion, [Weather], false).ToolCalls);
        Assert.Equal("other", call.Name);
        Assert.Equal("""{"n":"3"}""", call.Arguments.GetRawText());
    }

    [Theory]
    [InlineData("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis")]
    [InlineData("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</tool_call>")]
    [InlineData("<tool_call>\nget_weather(city=\"Paris\")\n</tool_call>")]
    [InlineData("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>")]
    [InlineData("<tool_call>\n<function=>\n</function>\n</tool_call>")]
    [InlineData("<tool_call>\n<function=a>\n</function>\n<function=b>\n</function>\n</tool_call>")]
    [InlineData("<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n<parameter=city>\nB\n</parameter>\n</function>\n</tool_call>")]
    [InlineData("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</function>\n</tool_call>")]
    [InlineData("<tool_call>\n<function=get_weather>\ntext between\n</function>\n</tool_call>")]
    public void MalformedBlocksAreReturnedAsTextWithoutThrowing(string completion)
    {
        var parsed = Qwen35ToolCallParser.Parse(completion, [Weather], false);
        Assert.Empty(parsed.ToolCalls);
        Assert.Equal(completion, parsed.Text);
        Assert.Equal(completion, StreamCharByChar(completion, [Weather], false).Text);
    }

    [Fact]
    public void AMalformedBlockDoesNotHideALaterValidOne()
    {
        string completion = "<tool_call>\nbroken\n</tool_call>\n" + WeatherCall;
        var parsed = Qwen35ToolCallParser.Parse(completion, [Weather], false);
        Assert.Equal("<tool_call>\nbroken\n</tool_call>", parsed.Parts[0].Text);
        Assert.Equal("get_weather", Assert.Single(parsed.ToolCalls).Name);
    }

    [Fact]
    public void WithThinkingNothingBeforeTheThinkCloseIsParsed()
    {
        string reasoning = "Maybe " + WeatherCall + " would help.\n</think>";
        var parsed = Qwen35ToolCallParser.Parse(reasoning + "\n\n" + WeatherCall, [Weather], enableThinking: true);
        Assert.Equal(reasoning, parsed.Parts[0].Text);
        Assert.Equal("get_weather", Assert.Single(parsed.ToolCalls).Name);

        // Without a closing tag (for example at the token limit) everything is reasoning text.
        var open = Qwen35ToolCallParser.Parse("Still thinking " + WeatherCall, [Weather], enableThinking: true);
        Assert.Empty(open.ToolCalls);
    }

    [Theory]
    [MemberData(nameof(Completions))]
    public void StreamedPartsEqualTheWholeParseForEverySplit(string completion)
    {
        foreach (bool thinking in new[] { false, true })
        {
            string text = thinking ? "Reasoning <tool_call> here.\n</think>\n\n" + completion : completion;
            var whole = Describe(Qwen35ToolCallParser.Parse(text, [Weather, Typed], thinking));
            Assert.Equal(whole, Describe(StreamCharByChar(text, [Weather, Typed], thinking)));
            for (int split = 1; split < text.Length; split++)
            {
                var parser = new Qwen35ToolCallParser([Weather, Typed], thinking);
                var parts = parser.Append(text[..split]).Concat(parser.Append(text[split..])).Concat(parser.Complete()).ToList();
                Assert.Equal(whole, Describe(new Qwen35ParsedOutput(MergeText(parts))));
            }
        }
    }

    [Fact]
    public void StreamingNeverReturnsRawToolCallMarkupForAValidBlock()
    {
        string completion = "Checking.\n\n" + WeatherCall;
        var parser = new Qwen35ToolCallParser([Weather], false);
        var texts = new List<string>();
        foreach (char c in completion)
        {
            texts.AddRange(parser.Append(c.ToString()).Where(p => p.Text is not null).Select(p => p.Text!));
        }

        var last = parser.Complete();
        Assert.All(texts, t => Assert.DoesNotContain("<", t, StringComparison.Ordinal));
        Assert.Equal("Checking.", string.Concat(texts));
        Assert.Empty(last);
    }

    [Fact]
    public void ACallIsReturnedAsSoonAsItsBlockCloses()
    {
        var parser = new Qwen35ToolCallParser([Weather], false);
        Assert.Empty(parser.Append(WeatherCall[..^1]));
        var part = Assert.Single(parser.Append(WeatherCall[^1..]));
        Assert.Equal("get_weather", part.ToolCall!.Name);
        Assert.Empty(parser.Complete());
        Assert.Throws<InvalidOperationException>(() => parser.Append("x"));
    }

    [Fact]
    public void AParsedCompletionRendersBackAsTheModelWroteIt()
    {
        var tools = new[] { Weather };
        Qwen35Message[] conversation = [Qwen35Message.User("Weather in Paris?")];
        string prompt = Qwen35ChatTemplate.Render(conversation, tools);
        foreach (string completion in new[] { WeatherCall, "Let me check.\n\n" + WeatherCall, WeatherCall + "\n" + WeatherCall })
        {
            var assistant = Qwen35ToolCallParser.Parse(completion, tools, false).ToAssistantMessage();
            string next = Qwen35ChatTemplate.Render([.. conversation, assistant, Qwen35Message.Tool("18 C")], tools);
            Assert.StartsWith(prompt + completion + "<|im_end|>\n<|im_start|>user\n<tool_response>\n18 C\n</tool_response><|im_end|>\n", next, StringComparison.Ordinal);
        }

        // With thinking, the reasoning returns inside the think block of the turn after the last query.
        string thinkingPrompt = Qwen35ChatTemplate.Render(conversation, tools, enableThinking: true);
        string thought = "I need the weather.\n</think>\n\n" + WeatherCall;
        var withReasoning = Qwen35ToolCallParser.Parse(thought, tools, true).ToAssistantMessage();
        Assert.StartsWith(
            thinkingPrompt + thought + "<|im_end|>\n",
            Qwen35ChatTemplate.Render([.. conversation, withReasoning, Qwen35Message.Tool("18 C")], tools, enableThinking: true),
            StringComparison.Ordinal);
    }

    [Fact]
    public void ToolDefinitionsAreRenderedAsPythonJsonDumps()
    {
        var tool = Qwen35Tool.Function("f", "Say \"hi\"\n\u0001 東京 🙂", Json("""{"type": "object", "properties": {"n": {"type": "number", "default": 1E+16, "minimum": 1e-5, "maximum": 100.0, "x": -0.0, "big": 123456789012345678901234567890}}, "required": []}"""));
        string rendered = Qwen35ChatTemplate.Render([Qwen35Message.User("Hi")], [tool]);
        Assert.Contains(
            """{"type": "function", "function": {"name": "f", "description": "Say \"hi\"\n\u0001 東京 🙂", "parameters": {"type": "object", "properties": {"n": {"type": "number", "default": 1e+16, "minimum": 1e-05, "maximum": 100.0, "x": -0.0, "big": 123456789012345678901234567890}}, "required": []}}}""",
            rendered,
            StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0.1, "0.1")]
    [InlineData(1e16, "1e+16")]
    [InlineData(1e-05, "1e-05")]
    [InlineData(0.0001, "0.0001")]
    [InlineData(123456789.0, "123456789.0")]
    [InlineData(1.5e300, "1.5e+300")]
    [InlineData(5e-324, "5e-324")]
    [InlineData(1.7976931348623157e308, "1.7976931348623157e+308")]
    [InlineData(9999999999999998.0, "9999999999999998.0")]
    [InlineData(1e15, "1000000000000000.0")]
    [InlineData(0.30000000000000004, "0.30000000000000004")]
    [InlineData(-2.5, "-2.5")]
    [InlineData(1152921504606846976.0, "1.152921504606847e+18")]
    [InlineData(1.0 / 3, "0.3333333333333333")]
    [InlineData(100.0, "100.0")]
    [InlineData(-0.0, "-0.0")]
    [InlineData(12345678901234567890.0, "1.2345678901234567e+19")]
    [InlineData(0.00012345, "0.00012345")]
    public void FloatsAreFormattedAsPythonRepr(double value, string expected) =>
        Assert.Equal(expected, Qwen35PythonJson.FloatRepr(value));

    [Fact]
    public void ArgumentScalarsAreRenderedAsPythonStr()
    {
        var call = new Qwen35ToolCall("f", Json("""{"b": true, "n": null, "i": 3, "x": 2.50, "s": " raw \"text\" ", "l": [1, {"k": false}]}"""));
        string rendered = Qwen35ChatTemplate.Render([Qwen35Message.User("Q"), Qwen35Message.Assistant(string.Empty, [call])]);
        Assert.Contains(
            "<tool_call>\n<function=f>\n<parameter=b>\nTrue\n</parameter>\n<parameter=n>\nNone\n</parameter>\n<parameter=i>\n3\n</parameter>\n<parameter=x>\n2.5\n</parameter>\n<parameter=s>\n raw \"text\" \n</parameter>\n<parameter=l>\n[1, {\"k\": false}]\n</parameter>\n</function>\n</tool_call>",
            rendered,
            StringComparison.Ordinal);
    }

    [Fact]
    public void ToolDataTheTemplateCannotRenderFailsExplicitly()
    {
        Qwen35Message[] user = [Qwen35Message.User("Q")];
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render(user, [Qwen35Tool.FromJson(Json("""{"function": {"name": "f", "parameters": {"a": 1, "a": 2}}}"""))]));
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([.. user, Qwen35Message.Assistant(string.Empty, [new Qwen35ToolCall("f", Json("\"not an object\""))])]));
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([.. user, Qwen35Message.Assistant(string.Empty, [new Qwen35ToolCall("f", Json("""{"a": 1, "a": 2}"""))])]));
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([.. user, Qwen35Message.Assistant(string.Empty, [new Qwen35ToolCall("f", Json("{\"s\": \"\\ud800\"}"))])]));
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([new Qwen35Message(Qwen35Role.User, "Q") { ToolCalls = [new Qwen35ToolCall("f", default)] }]));
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([.. user, new Qwen35Message(Qwen35Role.Tool, "r", "reasoning")]));
        Assert.Throws<ArgumentException>(() => Qwen35Tool.Function("bad>name"));
        Assert.Throws<ArgumentException>(() => Qwen35Tool.FromJson(Json("""{"type": "function"}""")));
        Assert.Throws<ArgumentException>(() => Qwen35Tool.FromJson(Json("[]")));
    }

    [Fact]
    public void AnEmptyToolListRendersNoToolsBlock()
    {
        Qwen35Message[] m = [Qwen35Message.System("sys"), Qwen35Message.User("Hi")];
        Assert.Equal(Qwen35ChatTemplate.Render(m), Qwen35ChatTemplate.Render(m, []));
        Assert.Equal(Qwen35ChatTemplate.Render(m), Qwen35ChatTemplate.Render(m, tools: null));
    }

    private static Qwen35ParsedOutput StreamCharByChar(string text, IReadOnlyList<Qwen35Tool> tools, bool thinking)
    {
        var parser = new Qwen35ToolCallParser(tools, thinking);
        var parts = new List<Qwen35OutputPart>();
        foreach (var rune in text.EnumerateRunes())
        {
            parts.AddRange(parser.Append(rune.ToString()));
        }

        parts.AddRange(parser.Complete());
        return new Qwen35ParsedOutput(MergeText(parts));
    }

    private static List<Qwen35OutputPart> MergeText(IEnumerable<Qwen35OutputPart> parts)
    {
        var merged = new List<Qwen35OutputPart>();
        foreach (var p in parts)
        {
            if (p.Text is { } t && merged.Count > 0 && merged[^1].Text is { } prev)
            {
                merged[^1] = Qwen35OutputPart.FromText(prev + t);
            }
            else
            {
                merged.Add(p);
            }
        }

        return merged;
    }

    private static string[] Describe(Qwen35ParsedOutput output) =>
        [.. output.Parts.Select(p => p.Text is { } t ? "text:" + t : string.Create(CultureInfo.InvariantCulture, $"call:{p.ToolCall!.Name}:{p.ToolCall.Arguments.GetRawText()}"))];

    private static JsonElement Json(string json)
    {
        using var document = JsonDocument.Parse(json);
        return document.RootElement.Clone();
    }
}
