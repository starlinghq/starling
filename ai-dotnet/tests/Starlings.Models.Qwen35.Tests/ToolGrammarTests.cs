using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>
/// <see cref="Qwen35ToolGrammar"/> without a model: the grammar text, and what it admits. A test translator turns the
/// grammar's rules into one .NET regular expression (the grammar uses only literals, sequences, optional terminals,
/// alternatives and regex terminals whose syntax .NET shares), so calls the pinned template renders can be checked
/// against it, and calls outside the tools' schemas can be checked to fail it.
/// </summary>
public sealed class ToolGrammarTests
{
    private static readonly Qwen35Tool Weather = Qwen35Tool.Function(
        "get_weather",
        "Current weather for a city.",
        Json("""
        {"type": "object", "properties": {
          "city": {"type": "string"},
          "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }, "required": ["city"]}
        """));

    private static readonly Qwen35Tool Timer = Qwen35Tool.Function(
        "set_timer",
        null,
        Json("""
        {"type": "object", "properties": {
          "minutes": {"type": "integer"},
          "ratio": {"type": "number"},
          "loud": {"type": "boolean"},
          "level": {"enum": [1, 2.5, true, null]}
        }, "required": ["minutes", "loud"]}
        """));

    private static readonly Qwen35Tool NoArguments = Qwen35Tool.Function("now");

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static JsonElement Json(string json) => JsonDocument.Parse(json).RootElement.Clone();

    [Fact]
    public void TheGrammarTextForOneRequiredCall()
    {
        string lark = Qwen35ToolGrammar.BuildLark([Weather, NoArguments]);
        Assert.Equal(
            "start: call\n" +
            "call: \"<tool_call>\\n\" (f0 | f1) \"</tool_call>\"\n" +
            "P0_0: /<parameter=city>\\n[^<]{0,1024}\\n<\\/parameter>\\n/\n" +
            "P0_1: /<parameter=units>\\n(celsius|fahrenheit)\\n<\\/parameter>\\n/\n" +
            "f0: \"<function=get_weather>\\n\" P0_0 P0_1? \"</function>\\n\"\n" +
            "f1: \"<function=now>\\n\" \"</function>\\n\"\n",
            lark);
    }

    [Fact]
    public void ModesMultipleCallsAndValueLength()
    {
        string multiple = Qwen35ToolGrammar.BuildLark([NoArguments], new Qwen35ToolGrammarOptions { AllowMultipleCalls = true });
        Assert.StartsWith("start: call (\"\\n\" call)*\n", multiple, StringComparison.Ordinal);
        string auto = Qwen35ToolGrammar.BuildLark([NoArguments], new Qwen35ToolGrammarOptions { Mode = Qwen35ToolGrammarMode.Auto });
        Assert.StartsWith("start: call | TEXT\nTEXT: /[^<](?s:.)*/\n", auto, StringComparison.Ordinal);
        string shortValues = Qwen35ToolGrammar.BuildLark([Weather], new Qwen35ToolGrammarOptions { MaxValueLength = 8 });
        Assert.Contains("[^<]{0,8}", shortValues, StringComparison.Ordinal);
    }

    [Fact]
    public void ValuePatternsFollowTheSchema()
    {
        string lark = Qwen35ToolGrammar.BuildLark([Timer]);
        Assert.Contains("P0_0: /<parameter=minutes>\\n-?[0-9]{1,20}\\n<\\/parameter>\\n/", lark, StringComparison.Ordinal);
        Assert.Contains("P0_1: /<parameter=ratio>\\n-?[0-9]{1,20}(\\.[0-9]{1,20})?\\n<\\/parameter>\\n/", lark, StringComparison.Ordinal);
        Assert.Contains("P0_2: /<parameter=loud>\\n(true|false|True|False)\\n<\\/parameter>\\n/", lark, StringComparison.Ordinal);
        Assert.Contains("P0_3: /<parameter=level>\\n(1|2\\.5|true|null)\\n<\\/parameter>\\n/", lark, StringComparison.Ordinal);
        Assert.Contains("f0: \"<function=set_timer>\\n\" P0_0 P0_1? P0_2 P0_3? \"</function>\\n\"", lark, StringComparison.Ordinal);
    }

    [Fact]
    public void LiteralsAndRegexTerminalsAreEscaped()
    {
        var odd = Qwen35Tool.Function(
            "say\"hi\\",
            null,
            Json("""{"type": "object", "properties": {"a.b/c": {"enum": ["x|y", "(z)*", "1+1?"]}}}"""));
        string lark = Qwen35ToolGrammar.BuildLark([odd]);
        Assert.Contains("\"<function=say\\\"hi\\\\>\\n\"", lark, StringComparison.Ordinal);
        Assert.Contains("/<parameter=a\\.b\\/c>\\n(x\\|y|\\(z\\)\\*|1\\+1\\?)\\n<\\/parameter>\\n/", lark, StringComparison.Ordinal);
        Assert.Matches(Translate(lark), Call("say\"hi\\", ("a.b/c", "(z)*")));
        Assert.DoesNotMatch(Translate(lark), Call("say\"hi\\", ("a.b/c", "zzz")));
    }

    [Fact]
    public void InvalidInputsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => Qwen35ToolGrammar.BuildLark([]));
        Assert.Throws<ArgumentNullException>(() => Qwen35ToolGrammar.BuildLark(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35ToolGrammar.BuildLark([Weather], new Qwen35ToolGrammarOptions { MaxValueLength = 0 }));
        foreach (string name in new[] { "a>b", "a<b", "a\nb", "" })
        {
            string schema = "{\"type\": \"object\", \"properties\": {" + JsonSerializer.Serialize(name) + ": {\"type\": \"string\"}}}";
            var bad = Qwen35Tool.Function("f", null, Json(schema));
            Assert.Throws<ArgumentException>(() => Qwen35ToolGrammar.BuildLark([bad]));
        }
    }

    [Fact]
    public void BuildReturnsLarkGuidanceWithFastForwardOnlyWhenAskedFor()
    {
        var plain = Qwen35ToolGrammar.Build([Weather]);
        Assert.Equal(GenerationGuidanceKind.Lark, plain.Kind);
        Assert.Equal(Qwen35ToolGrammar.BuildLark([Weather]), plain.Data);
        Assert.False(plain.FastForward);

        var ff = Qwen35ToolGrammar.Build([Weather], new Qwen35ToolGrammarOptions { FastForward = true });
        Assert.True(ff.FastForward);
        Assert.Equal(Encoding.UTF8.GetByteCount(ff.Data) + 1, ff.MaxFastForwardTokens);
    }

    [Fact]
    public void CallsTheTemplateRendersMatchTheGrammar()
    {
        var pattern = Translate(Qwen35ToolGrammar.BuildLark([Weather, Timer, NoArguments]));
        Qwen35ToolCall[] calls =
        [
            Qwen35ToolCall.Create("get_weather", [new("city", Json("\"Porto\""))]),
            Qwen35ToolCall.Create("get_weather", [new("city", Json("\"São Paulo\\nBrazil\"")), new("units", Json("\"fahrenheit\""))]),
            Qwen35ToolCall.Create("set_timer", [new("minutes", Json("25")), new("ratio", Json("-0.5")), new("loud", Json("true")), new("level", Json("2.5"))]),
            Qwen35ToolCall.Create("now", []),
        ];
        foreach (var call in calls)
        {
            string rendered = RenderedCall(call);
            Assert.Matches(pattern, rendered);

            // The parser reads back what the grammar admitted.
            var parsed = Assert.Single(Qwen35ToolCallParser.Parse(rendered, [Weather, Timer, NoArguments], enableThinking: false).ToolCalls);
            Assert.Equal(call.Name, parsed.Name);
        }
    }

    [Theory]
    [InlineData("unknown tool")]
    [InlineData("missing required parameter")]
    [InlineData("undeclared parameter")]
    [InlineData("parameters out of order")]
    [InlineData("value outside the enum")]
    [InlineData("integer with a fraction")]
    [InlineData("boolean as text")]
    [InlineData("value with a tag")]
    [InlineData("value too long")]
    [InlineData("text instead of a call")]
    [InlineData("two calls")]
    public void CallsOutsideTheSchemasDoNotMatch(string name)
    {
        var pattern = Translate(Qwen35ToolGrammar.BuildLark([Weather, Timer], new Qwen35ToolGrammarOptions { MaxValueLength = 16 }));
        string one = Call("get_weather", ("city", "Paris"));
        string text = name switch
        {
            "unknown tool" => Call("send_email", ("city", "Paris")),
            "missing required parameter" => Call("get_weather", ("units", "celsius")),
            "undeclared parameter" => Call("get_weather", ("city", "Paris"), ("country", "FR")),
            "parameters out of order" => Call("get_weather", ("units", "celsius"), ("city", "Paris")),
            "value outside the enum" => Call("get_weather", ("city", "Paris"), ("units", "kelvin")),
            "integer with a fraction" => Call("set_timer", ("minutes", "2.5"), ("loud", "true")),
            "boolean as text" => Call("set_timer", ("minutes", "2"), ("loud", "yes")),
            "value with a tag" => Call("get_weather", ("city", "Paris</parameter>")),
            "value too long" => Call("get_weather", ("city", new string('x', 17))),
            "text instead of a call" => "The weather in Paris is mild.",
            "two calls" => one + "\n" + one,
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
        Assert.DoesNotMatch(pattern, text);
        Assert.Matches(pattern, one);
    }

    [Fact]
    public void MultipleAndAutoModesAdmitWhatTheyDescribe()
    {
        string one = Call("get_weather", ("city", "Paris"));
        var multiple = Translate(Qwen35ToolGrammar.BuildLark([Weather], new Qwen35ToolGrammarOptions { AllowMultipleCalls = true }));
        Assert.Matches(multiple, one);
        Assert.Matches(multiple, one + "\n" + one + "\n" + one);
        Assert.DoesNotMatch(multiple, one + one);

        var auto = Translate(Qwen35ToolGrammar.BuildLark([Weather], new Qwen35ToolGrammarOptions { Mode = Qwen35ToolGrammarMode.Auto }));
        Assert.Matches(auto, one);
        Assert.Matches(auto, "It is sunny.\nNo <tool> needed.");
        Assert.DoesNotMatch(auto, "<b>bold</b>");
    }

    [Fact]
    public void GuidedRequestsNeedTheGuidanceFiles()
    {
        var model = TestModel(out var sequences, ["tokenizer_config.json"]);
        var options = new Qwen35ChatOptions { Guidance = Qwen35ToolGrammar.Build([Weather]), Tools = [Weather] };
        var ex = Assert.Throws<NotSupportedException>(() => model.Generate([Qwen35Message.User("Weather in Paris?")], options, Ct));
        Assert.Contains("tokenizer_config.json is missing", ex.Message, StringComparison.Ordinal);
        Assert.Empty(sequences);

        // Unguided requests do not depend on the files.
        model.Generate([Qwen35Message.User("Hi")], new Qwen35ChatOptions { MaxNewTokens = 4 }, Ct);
        Assert.Null(Assert.Single(sequences).Guidance);
    }

    [Fact]
    public void GuidanceReachesTheSequenceWhenTheFilesArePresent()
    {
        var model = TestModel(out var sequences, []);
        var guidance = Qwen35ToolGrammar.Build([Weather]);
        model.Generate([Qwen35Message.User("Weather in Paris?")], new Qwen35ChatOptions { Guidance = guidance, Tools = [Weather], MaxNewTokens = 4 }, Ct);
        Assert.Same(guidance, Assert.Single(sequences).Guidance);
    }

    private static Qwen35ChatModel TestModel(out List<GenAISequenceOptions> sequences, IReadOnlyList<string> missing)
    {
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        var created = new List<GenAISequenceOptions>();
        sequences = created;
        var generator = new TextGenerator(
            o =>
            {
                created.Add(o);
                return new EndSequence();
            },
            tokenizer!,
            [Qwen35ChatModel.EndOfTextTokenId, Qwen35ChatModel.ImEndTokenId],
            262144,
            Qwen35ChatModel.PadTokenIdOverlay);
        var info = new Qwen35ModelInfo
        {
            Profile = "test",
            Checkpoint = "test",
            Variant = Qwen35Variant.Fp32x,
            Provider = GenAIExecutionProvider.Cpu,
            GenAIVersion = "scripted",
            Tokenizer = "qwen35-bpe-chat",
            ChatTemplate = "owned C# renderer",
            ModelDirectory = "/scripted",
            ConfigOverlays = [],
            SerializesNativeCalls = false,
            ProviderOptions = new Dictionary<string, string>(),
            ContextLength = 8192,
            StopTokenIds = [],
            GuidanceMissingFiles = missing,
        };
        return Qwen35ChatModel.CreateWithoutBackend(generator, info);
    }

    /// <summary>A call as the template writes it (values verbatim).</summary>
    private static string Call(string name, params (string Key, string Value)[] arguments)
    {
        var sb = new StringBuilder("<tool_call>\n<function=").Append(name).Append(">\n");
        foreach (var (key, value) in arguments)
        {
            sb.Append("<parameter=").Append(key).Append(">\n").Append(value).Append("\n</parameter>\n");
        }

        return sb.Append("</function>\n</tool_call>").ToString();
    }

    /// <summary>The tool-call block the pinned template renders for an assistant turn with <paramref name="call"/>.</summary>
    private static string RenderedCall(Qwen35ToolCall call)
    {
        string prompt = Qwen35ChatTemplate.Render(
            [Qwen35Message.User("q"), new Qwen35Message(Qwen35Role.Assistant, string.Empty) { ToolCalls = [call] }, Qwen35Message.User("next")],
            [Weather, Timer, NoArguments],
            enableThinking: false);
        // After the assistant turn's start: the tools block's instructions carry an example call.
        int turn = prompt.IndexOf("<|im_start|>assistant\n", StringComparison.Ordinal);
        int start = prompt.IndexOf("<tool_call>\n<function=", turn, StringComparison.Ordinal);
        const string Close = "</tool_call>";
        int end = prompt.IndexOf(Close, start, StringComparison.Ordinal) + Close.Length;
        return prompt[start..end];
    }

    /// <summary>
    /// The grammar as one anchored .NET regex: string literals are escaped, regex terminals are kept (their syntax is
    /// common to Rust's and .NET's engines), rule references are expanded, and <c>?</c>, <c>|</c>, <c>( )*</c> carry over.
    /// </summary>
    private static Regex Translate(string lark)
    {
        var rules = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string line in lark.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            int colon = line.IndexOf(": ", StringComparison.Ordinal);
            rules[line[..colon]] = line[(colon + 2)..];
        }

        string Expand(string body)
        {
            var sb = new StringBuilder();
            int i = 0;
            while (i < body.Length)
            {
                char c = body[i];
                if (c == '"')
                {
                    int j = i + 1;
                    var literal = new StringBuilder();
                    while (body[j] != '"')
                    {
                        if (body[j] == '\\')
                        {
                            j++;
                            literal.Append(body[j] switch { 'n' => '\n', 'r' => '\r', 't' => '\t', var other => other });
                        }
                        else
                        {
                            literal.Append(body[j]);
                        }

                        j++;
                    }

                    sb.Append(Regex.Escape(literal.ToString()));
                    i = j + 1;
                }
                else if (c == '/')
                {
                    int j = i + 1;
                    while (body[j] != '/' || body[j - 1] == '\\')
                    {
                        j++;
                    }

                    sb.Append("(?:").Append(body[(i + 1)..j]).Append(')');
                    i = j + 1;
                }
                else if (char.IsLetter(c))
                {
                    int j = i;
                    while (j < body.Length && (char.IsLetterOrDigit(body[j]) || body[j] == '_'))
                    {
                        j++;
                    }

                    sb.Append("(?:").Append(Expand(rules[body[i..j]])).Append(')');
                    i = j;
                }
                else if (c == ' ')
                {
                    i++;
                }
                else
                {
                    sb.Append(c);
                    i++;
                }
            }

            return sb.ToString();
        }

        return new Regex("^(?:" + Expand(rules["start"]) + ")$", RegexOptions.CultureInvariant);
    }

    /// <summary>A sequence that ends at once (the end-of-turn id), for request plumbing tests.</summary>
    private sealed class EndSequence : IDecodeSequence
    {
        public int VocabularySize => 248320;

        public bool IsDone { get; private set; }

        public void AppendTokens(ReadOnlySpan<int> tokenIds)
        {
        }

        public int GenerateNextToken()
        {
            IsDone = true;
            return Qwen35ChatModel.ImEndTokenId;
        }

        public void CopyLastLogits(Span<float> destination) => destination.Clear();

        public void Dispose()
        {
        }
    }
}
