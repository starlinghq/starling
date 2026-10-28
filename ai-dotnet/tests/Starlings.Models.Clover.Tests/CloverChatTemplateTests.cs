using System.Text.Json;
using Starlings.Generation;
using Starlings.Tokenizers;

namespace Starlings.Models.Clover.Tests;

/// <summary>
/// The template port against the checkpoint's <c>chat_template.jinja</c> rendered by transformers 5.19.0
/// (<c>Data/template-reference.jsonl</c>, written by <c>model-pipeline/reference/clover/run_clover_template.py</c> over
/// <c>corpora/five-models/chat.jsonl</c> and <c>corpora/clover/chat.jsonl</c>), and the owned tokenizer's prompt ids
/// against the transformers tokenizer's.
/// </summary>
public sealed class CloverChatTemplateTests
{
    private static readonly Lazy<Dictionary<string, JsonElement>> _cases = new(ReadCases);
    private static readonly Lazy<Tokenizer?> _tokenizer = new(LoadTokenizer);

    public static TheoryData<string> CaseIds => [.. _cases.Value.Keys];

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void RendersLikeTheReference(string id)
    {
        JsonElement row = _cases.Value[id];

        string text = CloverChatTemplate.Render(
            Turns(row.GetProperty("messages")),
            enableThinking: Switch(row, "enable_thinking"),
            preserveThinking: Switch(row, "preserve_thinking"),
            addNonThinkingPrefix: Switch(row, "add_non_thinking_prefix"),
            addGenerationPrompt: Switch(row, "add_generation_prompt"));

        Assert.Equal(row.GetProperty("text").GetString(), text);
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void PromptIdsMatchTheReferenceTokenizer(string id)
    {
        Tokenizer? tokenizer = _tokenizer.Value;
        Assert.SkipWhen(tokenizer is null, $"Run tools/fetch-tokenizer-assets.sh for {CloverCheckpoint.Repository} tokenizer.json.");
        JsonElement row = _cases.Value[id];
        int[] expected = [.. row.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32())];

        Assert.Equal(expected, tokenizer!.Encode(row.GetProperty("text").GetString()!, addSpecialTokens: false).Ids.ToArray());
    }

    [Fact]
    public void CoversEverySwitchAndTheMaskLiteral()
    {
        Assert.True(_cases.Value.Count >= 30);
        foreach (string name in (string[])["enable_thinking", "preserve_thinking", "add_non_thinking_prefix", "add_generation_prompt"])
        {
            Assert.Contains(_cases.Value.Values, r => r.TryGetProperty(name, out var v) && !v.GetBoolean());
        }

        Assert.Contains(_cases.Value.Values, r => r.GetProperty("prompt_ids").EnumerateArray().Any(e => e.GetInt32() == CloverCheckpoint.MaskTokenId));
    }

    [Fact]
    public void RejectsWhatThePortDoesNotCarry()
    {
        Assert.Throws<ArgumentException>(() => CloverChatTemplate.Render([]));
        Assert.Throws<ArgumentException>(() => CloverChatTemplate.Render([ChatTurn.System("s")]));
        Assert.Throws<ArgumentException>(() => CloverChatTemplate.Render([ChatTurn.User("<tool_response>\nx\n</tool_response>")]));
        Assert.Throws<NotSupportedException>(() => CloverChatTemplate.Render([ChatTurn.User("a"), ChatTurn.System("b")]));
        Assert.Throws<NotSupportedException>(() => CloverChatTemplate.Render([ChatTurn.User("a"), ChatTurn.Tool("b")]));
        var call = new ToolCall("get_weather", JsonDocument.Parse("""{"city": "Paris"}""").RootElement.Clone());
        Assert.Throws<NotSupportedException>(() => CloverChatTemplate.Render([ChatTurn.User("a"), ChatTurn.Assistant("", [call])]));
        Assert.Throws<NotSupportedException>(() => CloverChatTemplate.Instance.Render([ChatTurn.User("a")], [ToolDefinition.Function("f")]));
    }

    [Fact]
    public void InterfaceRendersWithItsThinkingSwitch()
    {
        Assert.Equal(CloverChatTemplate.TemplateSha256, CloverChatTemplate.Instance.TemplateSha256);
        Assert.True(CloverChatTemplate.Instance.SupportsThinking);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n", CloverChatTemplate.Instance.Render([ChatTurn.User("a")]), StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n", CloverChatTemplate.Instance.Render([ChatTurn.User("a")], enableThinking: true), StringComparison.Ordinal);
    }

    // Python str.strip() removes the 25 White_Space characters and U+001C to U+001F.
    [Theory]
    [InlineData(" \t\n x y \r\n", "x y")]
    [InlineData("\u001Cx\u001F", "x")]
    [InlineData("\u00A0\u3000x\u2029", "x")]
    [InlineData("\u200Bx\u200B", "\u200Bx\u200B")]
    [InlineData("   ", "")]
    public void TrimFollowsPythonStrip(string input, string expected) =>
        Assert.Equal(expected, CloverChatTemplate.PythonStrip(input));

    // A switch the case does not name keeps the template's default (true for all four).
    private static bool Switch(JsonElement row, string name) => !row.TryGetProperty(name, out var v) || v.GetBoolean();

    private static IReadOnlyList<ChatTurn> Turns(JsonElement messages) =>
    [
        .. messages.EnumerateArray().Select(m =>
        {
            string content = m.GetProperty("content").GetString()!;
            string? reasoning = m.TryGetProperty("reasoning_content", out var r) ? r.GetString() : null;
            return m.GetProperty("role").GetString() switch
            {
                "system" => ChatTurn.System(content),
                "user" => ChatTurn.User(content),
                "assistant" => ChatTurn.Assistant(content, reasoning),
                var role => throw new InvalidDataException($"Unexpected role {role}."),
            };
        }),
    ];

    private static Dictionary<string, JsonElement> ReadCases()
    {
        var cases = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(Path.Combine(AppContext.BaseDirectory, "Data", "template-reference.jsonl")))
        {
            if (line.Length > 0)
            {
                JsonElement row = JsonDocument.Parse(line).RootElement.Clone();
                cases.Add(row.GetProperty("id").GetString()!, row);
            }
        }

        return cases;
    }

    // $STARLING_MODELS/<repo>/<revision>/tokenizer.json, else <repo root>/.models (tools/fetch-tokenizer-assets.sh).
    private static Tokenizer? LoadTokenizer()
    {
        string? root = Environment.GetEnvironmentVariable("STARLING_MODELS");
        if (string.IsNullOrEmpty(root))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "CLAUDE.md")))
            {
                dir = dir.Parent;
            }

            if (dir is null)
            {
                return null;
            }

            root = Path.Combine(dir.FullName, ".models");
        }

        string path = Path.Combine(root, CloverCheckpoint.Repository, CloverCheckpoint.Revision, "tokenizer.json");
        return File.Exists(path) ? Tokenizer.FromFile(path) : null;
    }
}
