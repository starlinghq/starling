using System.Text.Json;
using Starlings.Models.QwenChat;

namespace Starlings.Models.QwenChat.Tests;

/// <summary>
/// The owned renderer against the upstream renderings of both pinned templates (proposed corpora under
/// model-pipeline/proposed/), and the parser against the call blocks the templates write.
/// </summary>
public sealed class TemplateCorpusTests
{
    public static TheoryData<string, string> RenderCases() => Cases("render");

    public static TheoryData<string, string> UnsupportedCases() => Cases("unsupported");

    [Theory]
    [MemberData(nameof(RenderCases))]
    public void RenderingEqualsUpstream(string model, string id)
    {
        TemplateCorpusCase c = TemplateCorpus.Get(model, id);
        QwenChatProfile profile = TemplateCorpus.Profile(model);
        var (messages, reason) = c.ToTypedMessages(profile);
        Assert.True(messages is not null, reason);
        string rendered = QwenChatTemplate.Render(profile, messages, c.TypedTools);
        Assert.Equal(c.Rendered, rendered);

        var (tokenizer, _) = TemplateCorpus.Tokenizer(model);
        if (tokenizer is not null)
        {
            Assert.Equal(c.PromptIds, tokenizer.Encode(rendered, addSpecialTokens: false).Ids.ToArray());
        }
    }

    [Theory]
    [MemberData(nameof(UnsupportedCases))]
    public void UnsupportedCasesAreRejectedOrNotExpressible(string model, string id)
    {
        TemplateCorpusCase c = TemplateCorpus.Get(model, id);
        QwenChatProfile profile = TemplateCorpus.Profile(model);
        var (messages, reason) = c.ToTypedMessages(profile);
        if (messages is null)
        {
            Assert.NotEmpty(reason);
            return;
        }

        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Render(profile, messages, c.TypedTools));
    }

    [Fact]
    public void CorporaHoldTheExpectedCases()
    {
        Assert.Equal(46, TemplateCorpus.Load(TemplateCorpus.Qwen3).Count(c => c.Owned == "render"));
        Assert.Equal(8, TemplateCorpus.Load(TemplateCorpus.Qwen3).Count(c => c.Owned == "unsupported"));
        Assert.Equal(42, TemplateCorpus.Load(TemplateCorpus.Qwen25).Count(c => c.Owned == "render"));
        Assert.Equal(7, TemplateCorpus.Load(TemplateCorpus.Qwen25).Count(c => c.Owned == "unsupported"));
        Assert.All(TemplateCorpus.Load(TemplateCorpus.Qwen3).Where(c => c.Owned == "render"),
            c => Assert.Equal(JsonValueKind.False, c.Kwargs.GetProperty("enable_thinking").ValueKind));
    }

    [Fact]
    public void PromptIdsAreCheckedWhenTheTokenizerIsPresent()
    {
        foreach (string model in new[] { TemplateCorpus.Qwen3, TemplateCorpus.Qwen25 })
        {
            var (tokenizer, reason) = TemplateCorpus.Tokenizer(model);
            Assert.SkipWhen(tokenizer is null, reason);
        }
    }

    /// <summary>
    /// For every assistant turn with tool calls in a render case, the text the model would have generated for it (the
    /// rendering with that turn, after the prompt without it, up to <c>&lt;|im_end|&gt;</c>) parses back to the same
    /// calls and the turn's text, whole and streamed one character at a time.
    /// </summary>
    [Theory]
    [MemberData(nameof(RenderCases))]
    public void RenderedCallsParseBack(string model, string id)
    {
        TemplateCorpusCase c = TemplateCorpus.Get(model, id);
        QwenChatProfile profile = TemplateCorpus.Profile(model);
        QwenChatMessage[] messages = c.ToTypedMessages(profile).Messages!;
        for (int i = 1; i < messages.Length; i++)
        {
            QwenChatMessage turn = messages[i];
            if (turn.Role != QwenChatRole.Assistant || turn.ToolCalls is not { Count: > 0 } ||
                turn.ReasoningContent is not null || turn.Content.Contains("</think>", StringComparison.Ordinal))
            {
                continue;
            }

            string prompt = QwenChatTemplate.Render(profile, messages[..i], c.TypedTools);
            string withTurn = QwenChatTemplate.Render(profile, messages[..(i + 1)], c.TypedTools);
            Assert.StartsWith(prompt, withTurn, StringComparison.Ordinal);
            // The turn ends with <|im_end|>\n, then the generation prompt follows.
            string tail = "<|im_end|>\n" + GenerationPrompt(profile);
            Assert.EndsWith(tail, withTurn, StringComparison.Ordinal);
            string completion = withTurn[prompt.Length..^tail.Length];

            QwenChatParsedOutput parsed = QwenChatToolCallParser.Parse(completion);
            // Qwen3 writes the last assistant turn's content after its think block, without leading line feeds.
            string expectedText = profile == QwenChatProfile.Qwen3 ? turn.Content.TrimStart('\n').TrimEnd() : turn.Content.TrimEnd();
            Assert.Equal(expectedText, parsed.Text);
            Assert.Equal(turn.ToolCalls.Count, parsed.ToolCalls.Count);
            for (int k = 0; k < turn.ToolCalls.Count; k++)
            {
                Assert.Equal(turn.ToolCalls[k].Name, parsed.ToolCalls[k].Name);
                Assert.True(JsonElement.DeepEquals(turn.ToolCalls[k].Arguments, parsed.ToolCalls[k].Arguments), $"{id} call {k}");
            }

            var parser = new QwenChatToolCallParser();
            var streamed = new List<QwenChatOutputPart>();
            foreach (char ch in completion)
            {
                streamed.AddRange(parser.Append(ch.ToString()));
            }

            streamed.AddRange(parser.Complete());
            Assert.Equal(Describe(parsed.Parts), Describe(QwenChatToolCallParser.Parse(completion).Parts));
            Assert.Equal(Describe(parsed.Parts), Describe(Merge(streamed)));

            // The parsed turn renders back to the same prompt when its text had no surrounding whitespace.
            if (turn.Content == turn.Content.Trim())
            {
                Assert.Equal(withTurn, QwenChatTemplate.Render(profile, [.. messages[..i], parsed.ToAssistantMessage()], c.TypedTools));
            }
        }
    }

    private static string GenerationPrompt(QwenChatProfile profile) =>
        profile == QwenChatProfile.Qwen3 ? "<|im_start|>assistant\n<think>\n\n</think>\n\n" : "<|im_start|>assistant\n";

    internal static string Describe(IEnumerable<QwenChatOutputPart> parts) =>
        string.Join("|", parts.Select(p => p.ToolCall is { } call ? $"call:{call.Name}:{call.Arguments.GetRawText()}" : $"text:{p.Text}"));

    internal static List<QwenChatOutputPart> Merge(IEnumerable<QwenChatOutputPart> parts)
    {
        var merged = new List<QwenChatOutputPart>();
        foreach (var part in parts)
        {
            if (part.Text is { } text && merged.Count > 0 && merged[^1].Text is { } previous)
            {
                merged[^1] = QwenChatOutputPart.FromText(previous + text);
            }
            else
            {
                merged.Add(part);
            }
        }

        return merged;
    }

    private static TheoryData<string, string> Cases(string owned)
    {
        var data = new TheoryData<string, string>();
        foreach (string model in new[] { TemplateCorpus.Qwen3, TemplateCorpus.Qwen25 })
        {
            foreach (TemplateCorpusCase c in TemplateCorpus.Load(model).Where(c => c.Owned == owned))
            {
                data.Add(model, c.Id);
            }
        }

        return data;
    }
}
