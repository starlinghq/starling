using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>
/// P-029: the owned C# renderer against the template corpus, whose expected renderings and errors come from the pinned
/// upstream renderer (transformers 5.18.0 apply_chat_template, jinja2 3.1.6), never from the code under test. CI-safe:
/// rendering needs only the corpus; the id check needs the qwen35-bpe-chat tokenizer.json.
/// </summary>
public sealed class TemplateCorpusTests
{
    public static TheoryData<string> CaseIds() => [.. Qwen35TemplateCorpus.Load().Select(c => c.Id)];

    public static TheoryData<string> RenderCaseIds() => [.. Qwen35TemplateCorpus.Load().Where(c => c.Owned == "render").Select(c => c.Id)];

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void OwnedRendererMatchesUpstreamOrFailsExplicitly(string id)
    {
        var c = Qwen35TemplateCorpus.Get(id);
        var (messages, reason) = c.ToTypedMessages();
        switch (c.Owned)
        {
            case "render":
                Assert.True(messages is not null, $"{id}: in the supported subset but not expressible: {reason}");
                Assert.Equal(c.Rendered, Qwen35ChatTemplate.Render(messages!, c.EnableThinking));
                break;
            case "template_error":
                Assert.True(messages is not null, $"{id}: {reason}");
                var ex = Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render(messages!, c.EnableThinking));
                if (c.ErrorSource == "template")
                {
                    Assert.Equal(c.Error, ex.Message);
                }

                break;
            case "unsupported":
                Assert.NotNull(c.UnsupportedFeature);

                // Either the typed API cannot express the feature at all, or Render rejects it; it never renders.
                if (messages is not null)
                {
                    Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render(messages, c.EnableThinking));
                }

                break;
            default:
                Assert.Fail($"{id}: unknown expectation {c.Owned}");
                break;
        }
    }

    [Theory]
    [MemberData(nameof(RenderCaseIds))]
    public void OwnedTokenizerIdsOfTheOwnedRenderingEqualTheReference(string id)
    {
        var c = Qwen35TemplateCorpus.Get(id);
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        string rendered = Qwen35ChatTemplate.Render(c.ToTypedMessages().Messages!, c.EnableThinking);
        Assert.Equal(c.PromptIds, tokenizer!.Encode(rendered, addSpecialTokens: false).Ids);
    }

    [Fact]
    public void CorpusCoversWhatTheRendererClaims()
    {
        var cases = Qwen35TemplateCorpus.Load();
        Assert.True(cases.Count >= 80, $"{cases.Count} cases");
        foreach (string category in new[] { "multi-turn", "system", "enable_thinking", "history-think", "whitespace", "unicode", "long", "unsupported" })
        {
            Assert.Contains(cases, c => c.Category == category);
        }

        // Thinking on, off (explicit) and undefined; reasoning given and extracted; very long content.
        Assert.Contains(cases, c => c.Owned == "render" && c.EnableThinking);
        Assert.Contains(cases, c => c.Owned == "render" && c.Kwargs.TryGetProperty("enable_thinking", out var e) && e.ValueKind == System.Text.Json.JsonValueKind.False);
        Assert.Contains(cases, c => c.Owned == "render" && !c.Kwargs.TryGetProperty("enable_thinking", out _));
        Assert.Contains(cases, c => c.Owned == "render" && c.PromptIds!.Length > 10000);

        // Every unsupported template feature named in Qwen35ChatTemplate's remarks has a case that must fail.
        string[] features = ["tools", "tool_message", "tool_calls", "image", "video", "add_vision_id", "add_generation_prompt_false", "role"];
        foreach (string feature in features)
        {
            Assert.Contains(cases, c => c.Owned == "unsupported" && c.UnsupportedFeature == feature);
        }

        // Every error the template itself raises in the text subset.
        string[] errors = ["No user query found in messages.", "System message must be at the beginning."];
        foreach (string error in errors)
        {
            Assert.Contains(cases, c => c.Owned == "template_error" && c.ErrorSource == "template" && c.Error == error);
        }
    }

    [Fact]
    public void EmptyConversationFailsWithTheTemplateMessage()
    {
        // transformers rejects an empty conversation before rendering (err-no-messages records its message); the
        // template's own check, which the owned renderer ports, says "No messages provided.".
        var c = Qwen35TemplateCorpus.Get("err-no-messages");
        Assert.Equal("transformers", c.ErrorSource);
        Assert.Equal("No messages provided.", Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([])).Message);
    }
}
