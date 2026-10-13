using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>
/// The owned C# chat template renderer against the upstream rendering (transformers 5.18.0 apply_chat_template) on the
/// prompts-v1 reference, and the template's own rules. CI-safe: needs only the fixture and, for ids, tokenizer.json.
/// </summary>
public sealed class TemplateTests
{
    public static TheoryData<string> CaseIds() => [.. Qwen35TestAssets.LoadCases().Select(c => c.Id)];

    private static GenerationCase Case(string id) => Qwen35TestAssets.LoadCases().Single(c => c.Id == id);

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void RenderingEqualsTheReferenceOrIsRejected(string id)
    {
        var c = Case(id);
        if (c.HasTools)
        {
            // Tool definitions are not part of the supported subset (the API cannot express them).
            Assert.Equal("tools", c.Id);
            return;
        }

        Assert.Equal(c.Rendered, Qwen35ChatTemplate.Render(c.Messages!, c.EnableThinking));
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void OwnedTokenizerIdsOfTheRenderingEqualTheReferencePromptIds(string id)
    {
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        var c = Case(id);
        // The reference rendering (including the tools prompt) through the owned tokenizer: the 6.4 boundary.
        Assert.Equal(c.PromptIds, tokenizer!.Encode(c.Rendered, addSpecialTokens: false).Ids);
    }

    [Fact]
    public void SupportedSubsetCoversAllButTheToolsPrompt()
    {
        var cases = Qwen35TestAssets.LoadCases();
        Assert.Equal(22, cases.Count);
        Assert.Equal(["tools"], cases.Where(c => c.HasTools).Select(c => c.Id));
        Assert.Contains(cases, c => c.EnableThinking);
    }

    [Fact]
    public void GenerationPromptWithAndWithoutThinking()
    {
        var m = new[] { Qwen35Message.User("Hi") };
        Assert.Equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", Qwen35ChatTemplate.Render(m));
        Assert.Equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n", Qwen35ChatTemplate.Render(m, enableThinking: true));
    }

    [Fact]
    public void ContentIsStrippedWithPythonWhitespaceSemantics()
    {
        // U+001C..U+001F are whitespace for Python str.strip() but not for char.IsWhiteSpace.
        string fs = ((char)0x1C).ToString();
        string nbsp = ((char)0xA0).ToString();
        string zwsp = ((char)0x200B).ToString();
        string rendered = Qwen35ChatTemplate.Render([Qwen35Message.User(fs + nbsp + " a b \t\n" + fs)]);
        Assert.StartsWith("<|im_start|>user\na b<|im_end|>", rendered, StringComparison.Ordinal);
        // U+200B is not whitespace for Python either.
        Assert.StartsWith("<|im_start|>user\n" + zwsp + "x<|im_end|>", Qwen35ChatTemplate.Render([Qwen35Message.User(zwsp + "x")]), StringComparison.Ordinal);
        Assert.False(char.IsWhiteSpace((char)0x1C));
        Assert.True(Qwen35ChatTemplate.IsPythonSpace((char)0x1C));
    }

    [Fact]
    public void AssistantTurnsBeforeTheLastQueryLoseTheirReasoning()
    {
        var conversation = new[]
        {
            Qwen35Message.System("Be brief."),
            Qwen35Message.User("Q1"),
            Qwen35Message.Assistant("<think>\nr1\n</think>\n\nA1"),
            Qwen35Message.User("Q2"),
        };
        Assert.Equal(
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n" +
            "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            Qwen35ChatTemplate.Render(conversation));
    }

    [Fact]
    public void AssistantTurnsAfterTheLastQueryKeepReasoning()
    {
        var conversation = new[]
        {
            Qwen35Message.User("<tool_response>ignored as a query</tool_response>"),
            Qwen35Message.User("Q"),
            Qwen35Message.Assistant("A", reasoningContent: "  because  "),
            Qwen35Message.User("<tool_response>\n42\n</tool_response>"),
        };
        string rendered = Qwen35ChatTemplate.Render(conversation);
        // The last user message is a tool response, so the last query is "Q" and the assistant turn after it keeps
        // its (trimmed) reasoning.
        Assert.Contains("<|im_start|>assistant\n<think>\nbecause\n</think>\n\nA<|im_end|>\n", rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void ReasoningIsExtractedFromThinkTagsAsUpstreamDoes()
    {
        var conversation = new[]
        {
            Qwen35Message.User("Q"),
            Qwen35Message.Assistant("pre<think>\n\nfirst</think>\n<think>second\n\n</think>\n\nanswer"),
            Qwen35Message.User("<tool_response>t</tool_response>"),
        };
        // split('</think>')[0] = "pre<think>\n\nfirst", rstrip('\n'), split('<think>')[-1] = "\n\nfirst", lstrip('\n')
        // = "first"; content = split('</think>')[-1].lstrip('\n') = "answer".
        Assert.Contains("<|im_start|>assistant\n<think>\nfirst\n</think>\n\nanswer<|im_end|>", Qwen35ChatTemplate.Render(conversation), StringComparison.Ordinal);
    }

    [Fact]
    public void InvalidConversationsFailAsTheTemplateDoes()
    {
        Assert.Equal("No messages provided.", Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([])).Message);
        Assert.Equal("System message must be at the beginning.",
            Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([Qwen35Message.User("a"), Qwen35Message.System("s")])).Message);
        Assert.Equal("No user query found in messages.",
            Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([Qwen35Message.System("s")])).Message);
        Assert.Equal("No user query found in messages.",
            Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([Qwen35Message.User("<tool_response>x</tool_response>")])).Message);
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([new Qwen35Message(Qwen35Role.User, "a", "r")]));
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([new Qwen35Message((Qwen35Role)9, "a")]));
        Assert.Throws<ArgumentException>(() => Qwen35ChatTemplate.Render([new Qwen35Message(Qwen35Role.User, null!)]));
    }

    [Fact]
    public void ContinuedGeneratorStateIsNotAPrefixOfTheNextTurn()
    {
        // Why conversation state reuse is unsupported: the first turn's prompt ends with an empty think block that the
        // template drops from earlier assistant turns, so prompt + reply is not a prefix of the next rendering.
        string first = Qwen35ChatTemplate.Render([Qwen35Message.User("My name is Ana.")]);
        string continued = first + "Nice to meet you, Ana!<|im_end|>\n";
        string next = Qwen35ChatTemplate.Render(
            [Qwen35Message.User("My name is Ana."), Qwen35Message.Assistant("Nice to meet you, Ana!"), Qwen35Message.User("What is my name?")]);
        Assert.False(next.StartsWith(continued, StringComparison.Ordinal));
        Assert.StartsWith("<|im_start|>user\nMy name is Ana.<|im_end|>\n<|im_start|>assistant\nNice to meet you, Ana!<|im_end|>\n", next, StringComparison.Ordinal);
    }
}
