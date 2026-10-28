using System.Text.Json;
using Starlings.Generation;

namespace Starlings.Models.Lfm3.Tests;

/// <summary>
/// The template port against the checkpoint's <c>chat_template.jinja</c> rendered by transformers 5.21.0
/// (<c>Data/template-reference.jsonl</c>, written by <c>model-pipeline/reference/lfm3/run_lfm3.py template</c> over
/// <c>corpora/five-models/chat.jsonl</c> and <c>corpora/lfm3/chat.jsonl</c>).
/// </summary>
public sealed class Lfm3ChatTemplateTests
{
    private static readonly Lazy<Dictionary<string, JsonElement>> _cases = new(() => ReadCases());

    public static TheoryData<string> CaseIds => [.. _cases.Value.Keys];

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void RendersLikeTheReference(string id)
    {
        JsonElement row = _cases.Value[id];
        IReadOnlyList<ChatTurn> turns = Turns(row.GetProperty("messages"));
        bool enableThinking = row.TryGetProperty("enable_thinking", out var t) ? t.GetBoolean() : true;
        bool preserveThinking = row.TryGetProperty("preserve_thinking", out var p) && p.GetBoolean();

        string text = Lfm3ChatTemplate.Render(turns, enableThinking, preserveThinking);

        Assert.Equal(row.GetProperty("text").GetString(), text);
    }

    [Fact]
    public void CoversBothThinkingModesAndEarlierReasoning()
    {
        Assert.True(_cases.Value.Count >= 19);
        Assert.Contains(_cases.Value.Values, r => r.GetProperty("enable_thinking").GetBoolean());
        Assert.Contains(_cases.Value.Values, r => r.TryGetProperty("preserve_thinking", out var p) && p.GetBoolean());
        Assert.Contains("assistant-thinking", _cases.Value.Keys);
        Assert.Contains("reasoning-content", _cases.Value.Keys);
    }

    [Fact]
    public void WithoutGenerationPromptEndsAfterTheLastTurn()
    {
        string text = Lfm3ChatTemplate.Render([ChatTurn.User("Hi"), ChatTurn.Assistant("Hello.")], addGenerationPrompt: false);
        Assert.Equal("<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>Hello.<|im_end|>\n", text);
    }

    [Fact]
    public void RejectsWhatThePortDoesNotCarry()
    {
        Assert.Throws<ArgumentException>(() => Lfm3ChatTemplate.Render([]));
        Assert.Throws<NotSupportedException>(() => Lfm3ChatTemplate.Render([ChatTurn.User("a"), ChatTurn.System("b")]));
        Assert.Throws<NotSupportedException>(() => Lfm3ChatTemplate.Render([ChatTurn.User("a"), ChatTurn.Tool("b")]));
        var call = new ToolCall("get_weather", JsonDocument.Parse("""{"city": "Paris"}""").RootElement.Clone());
        Assert.Throws<NotSupportedException>(() => Lfm3ChatTemplate.Render([ChatTurn.User("a"), ChatTurn.Assistant("", [call])]));
        Assert.Throws<NotSupportedException>(() => Lfm3ChatTemplate.Instance.Render([ChatTurn.User("a")], [ToolDefinition.Function("f")]));
    }

    [Fact]
    public void InterfaceRendersWithItsThinkingSwitch()
    {
        Assert.Equal(Lfm3ChatTemplate.TemplateSha256, Lfm3ChatTemplate.Instance.TemplateSha256);
        Assert.True(Lfm3ChatTemplate.Instance.SupportsThinking);
        Assert.EndsWith("<|im_start|>assistant\n<think></think>", Lfm3ChatTemplate.Instance.Render([ChatTurn.User("a")]), StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>assistant\n<think>", Lfm3ChatTemplate.Instance.Render([ChatTurn.User("a")], enableThinking: true), StringComparison.Ordinal);
    }

    // Python str.strip() removes the 25 White_Space characters and U+001C to U+001F.
    [Theory]
    [InlineData(" \t\n x y \r\n", "x y")]
    [InlineData("\u001Cx\u001F", "x")]
    [InlineData("\u00A0\u3000x\u2029", "x")]
    [InlineData("\u200Bx\u200B", "\u200Bx\u200B")]
    [InlineData("   ", "")]
    public void TrimFollowsPythonStrip(string input, string expected) =>
        Assert.Equal(expected, Lfm3ChatTemplate.PythonStrip(input));

    internal static IReadOnlyList<ChatTurn> Turns(JsonElement messages) =>
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

    internal static Dictionary<string, JsonElement> ReadCases()
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
}
