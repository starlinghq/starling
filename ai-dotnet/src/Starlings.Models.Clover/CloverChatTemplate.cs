// Ported from the chat_template.jinja of Diffuspace/Clover-35B-A3B-Preview at 4b0402b65e523d0183a14c431955438714cb16ff
// (derived by DiffuSpace from the Qwen/Qwen3.6-35B-A3B template), Copyright 2026 Alibaba Cloud, Apache-2.0. Modified:
// rewritten in C#, restricted to text system, user and assistant turns.

using System.Text;
using Starlings.Generation;

namespace Starlings.Models.Clover;

/// <summary>
/// The Clover chat template (<c>chat_template.jinja</c> of <c>Diffuspace/Clover-35B-A3B-Preview</c> at <c>4b0402b6</c>)
/// for text turns: system, user and assistant, with the assistant's reasoning and the template's
/// <c>enable_thinking</c>, <c>preserve_thinking</c> and <c>add_non_thinking_prefix</c> switches.
/// </summary>
/// <remarks>
/// <para>What the template renders here, in order: the first turn, if it is a system turn, as
/// <c>&lt;|im_start|&gt;system\n…&lt;|im_end|&gt;\n</c> (also when its content is empty); each other turn as
/// <c>&lt;|im_start|&gt;role\n…&lt;|im_end|&gt;\n</c>; then <c>&lt;|im_start|&gt;assistant\n&lt;think&gt;\n</c>
/// (<c>&lt;think&gt;\n\n&lt;/think&gt;\n\n</c> with thinking disabled). Every turn's content is trimmed as Python
/// <c>str.strip()</c> does.</para>
/// <para>An assistant turn's reasoning is its <see cref="ChatTurn.ReasoningContent"/> when set (an empty string
/// included); otherwise, when the content holds <c>&lt;/think&gt;</c>, the text before the first <c>&lt;/think&gt;</c>
/// after its last <c>&lt;think&gt;</c>, and the content becomes the text after the last <c>&lt;/think&gt;</c>. The turn
/// is prefixed with <c>&lt;think&gt;\nreasoning\n&lt;/think&gt;\n\n</c> (<c>&lt;think&gt;\n\n&lt;/think&gt;\n\n</c>
/// without reasoning, when <c>add_non_thinking_prefix</c> is on) if <c>preserve_thinking</c> is on or the turn comes
/// after the last user query. A user turn whose trimmed content is a whole <c>&lt;tool_response&gt;…&lt;/tool_response&gt;</c>
/// is not a query.</para>
/// <para>Not supported (rejected with <see cref="NotSupportedException"/>): tools, tool calls and tool turns, and a
/// system turn after the first position. Text is inserted as given, as the template does: a special token's text in a
/// turn (for example <c>&lt;|im_end|&gt;</c> or <c>&lt;|mask|&gt;</c>) becomes that special token when tokenized.</para>
/// </remarks>
public static class CloverChatTemplate
{
    /// <summary>SHA-256 of the upstream <c>chat_template.jinja</c> this renderer follows (7,912 bytes).</summary>
    public const string TemplateSha256 = "01094df7cd8905d1220ea8b31e4f1086dc9bedf0305d3c2fa6578b5fa33b6815";

    private const string ThinkOpen = "<think>";
    private const string ThinkClose = "</think>";

    /// <summary>The renderer as the runtime-independent <see cref="IChatTemplate"/>.</summary>
    public static IChatTemplate Instance { get; } = new Renderer();

    /// <summary>
    /// Renders <paramref name="turns"/> as the template does with <c>add_generation_prompt=True</c> (or false).
    /// </summary>
    /// <param name="turns">The conversation (at least one turn, and a user query).</param>
    /// <param name="enableThinking">The template's <c>enable_thinking</c> (its default is true).</param>
    /// <param name="preserveThinking">The template's <c>preserve_thinking</c> (its default is true).</param>
    /// <param name="addNonThinkingPrefix">The template's <c>add_non_thinking_prefix</c> (its default is true).</param>
    /// <param name="addGenerationPrompt">Append the assistant generation prompt (default true).</param>
    /// <exception cref="ArgumentException">The conversation is empty or has no user query.</exception>
    /// <exception cref="NotSupportedException">A tool turn, tool calls, or a system turn that is not first.</exception>
    public static string Render(
        IReadOnlyList<ChatTurn> turns,
        bool enableThinking = true,
        bool preserveThinking = true,
        bool addNonThinkingPrefix = true,
        bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(turns);
        if (turns.Count == 0)
        {
            throw new ArgumentException("The conversation needs at least one turn.", nameof(turns));
        }

        for (int i = 0; i < turns.Count; i++)
        {
            ChatTurn turn = turns[i];
            ArgumentNullException.ThrowIfNull(turn);
            switch (turn.Role)
            {
                case ChatTurnRole.System when i > 0:
                    throw new NotSupportedException("A system turn is supported only as the first turn.");
                case ChatTurnRole.Assistant when turn.ToolCalls is { Count: > 0 }:
                    throw new NotSupportedException("Tool calls are not supported by this Clover template port.");
                case ChatTurnRole.System or ChatTurnRole.User or ChatTurnRole.Assistant:
                    break;
                default:
                    throw new NotSupportedException($"Turn role {turn.Role} is not supported by this Clover template port.");
            }
        }

        int lastQuery = -1;
        for (int i = turns.Count - 1; i >= 0; i--)
        {
            if (turns[i].Role == ChatTurnRole.User && !IsToolResponse(PythonStrip(turns[i].Content)))
            {
                lastQuery = i;
                break;
            }
        }

        if (lastQuery < 0)
        {
            throw new ArgumentException("No user query found in the conversation.", nameof(turns));
        }

        var sb = new StringBuilder();
        if (turns[0].Role == ChatTurnRole.System)
        {
            sb.Append("<|im_start|>system\n").Append(PythonStrip(turns[0].Content)).Append("<|im_end|>\n");
        }

        for (int i = 0; i < turns.Count; i++)
        {
            ChatTurn turn = turns[i];
            string content = PythonStrip(turn.Content);
            if (turn.Role == ChatTurnRole.User)
            {
                sb.Append("<|im_start|>user\n").Append(content).Append("<|im_end|>\n");
                continue;
            }

            if (turn.Role != ChatTurnRole.Assistant)
            {
                continue;
            }

            string reasoning = string.Empty;
            if (turn.ReasoningContent is { } given)
            {
                reasoning = given;
            }
            else
            {
                int firstClose = content.IndexOf(ThinkClose, StringComparison.Ordinal);
                if (firstClose >= 0)
                {
                    // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                    string before = content[..firstClose].TrimEnd('\n');
                    int lastOpen = before.LastIndexOf(ThinkOpen, StringComparison.Ordinal);
                    reasoning = (lastOpen >= 0 ? before[(lastOpen + ThinkOpen.Length)..] : before).TrimStart('\n');

                    // content.split('</think>')[-1].lstrip('\n')
                    int lastClose = content.LastIndexOf(ThinkClose, StringComparison.Ordinal);
                    content = content[(lastClose + ThinkClose.Length)..].TrimStart('\n');
                }
            }

            reasoning = PythonStrip(reasoning);
            bool hasReasoning = reasoning.Length > 0;
            bool wrapThinking = (hasReasoning || addNonThinkingPrefix) && (preserveThinking || i > lastQuery);

            sb.Append("<|im_start|>assistant\n");
            if (wrapThinking)
            {
                sb.Append(hasReasoning ? $"<think>\n{reasoning}\n</think>\n\n" : "<think>\n\n</think>\n\n");
            }

            sb.Append(content.TrimStart('\n')).Append("<|im_end|>\n");
        }

        if (addGenerationPrompt)
        {
            sb.Append("<|im_start|>assistant\n").Append(enableThinking ? "<think>\n" : "<think>\n\n</think>\n\n");
        }

        return sb.ToString();
    }

    private static bool IsToolResponse(string content) =>
        content.StartsWith("<tool_response>", StringComparison.Ordinal) &&
        content.EndsWith("</tool_response>", StringComparison.Ordinal);

    // Jinja's trim filter is Python str.strip(): it removes the characters for which str.isspace() is true.
    internal static string PythonStrip(string s)
    {
        int begin = 0;
        int end = s.Length;
        while (begin < end && IsPythonSpace(s, begin, out int width))
        {
            begin += width;
        }

        while (end > begin && IsPythonSpaceEndingAt(s, end, out int width))
        {
            end -= width;
        }

        return s[begin..end];
    }

    private static bool IsPythonSpace(string s, int index, out int width)
    {
        Rune.DecodeFromUtf16(s.AsSpan(index), out Rune rune, out width);
        return IsPythonSpace(rune);
    }

    private static bool IsPythonSpaceEndingAt(string s, int end, out int width)
    {
        Rune.DecodeLastFromUtf16(s.AsSpan(0, end), out Rune rune, out width);
        return IsPythonSpace(rune);
    }

    // Python's str.isspace(): bidirectional class WS, B or S, or general category Zs, which is the 25 White_Space
    // characters plus U+001C to U+001F.
    private static bool IsPythonSpace(Rune r) =>
        r.Value is >= 0x1C and <= 0x1F || Rune.IsWhiteSpace(r);

    private sealed class Renderer : IChatTemplate
    {
        public string TemplateSha256 => CloverChatTemplate.TemplateSha256;

        public bool SupportsThinking => true;

        public string Render(IReadOnlyList<ChatTurn> turns, IReadOnlyList<ToolDefinition>? tools = null, bool enableThinking = false)
        {
            if (tools is { Count: > 0 })
            {
                throw new NotSupportedException("Tools are not supported by this Clover template port.");
            }

            return CloverChatTemplate.Render(turns, enableThinking);
        }
    }
}
