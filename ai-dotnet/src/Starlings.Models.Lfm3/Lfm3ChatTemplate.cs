// Ported from the chat_template.jinja of LiquidAI/LFM3-1.7B at 40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd, Copyright
// Liquid AI, Inc., LFM Open License v1.0. Modified: rewritten in C#, restricted to text system, user and assistant turns.

using System.Text;
using Starlings.Generation;

namespace Starlings.Models.Lfm3;

/// <summary>
/// The LFM3 chat template (<c>chat_template.jinja</c> of <c>LiquidAI/LFM3-1.7B</c> at <c>40ebe0aa</c>, identical in
/// <c>LiquidAI/LFM3-1.7B-ONNX</c> at <c>c680759a</c>) for text turns: system, user and assistant, with the assistant's
/// reasoning and the template's <c>enable_thinking</c> and <c>preserve_thinking</c> switches.
/// </summary>
/// <remarks>
/// <para>What the template renders here, in order: <c>&lt;|startoftext|&gt;</c>; the first turn, if it is a system
/// turn with non-empty content, as <c>&lt;|im_start|&gt;system\n…&lt;|im_end|&gt;\n</c>; each other turn as
/// <c>&lt;|im_start|&gt;role\n…&lt;|im_end|&gt;\n</c>; then <c>&lt;|im_start|&gt;assistant\n&lt;think&gt;</c>
/// (<c>&lt;think&gt;&lt;/think&gt;</c> with thinking disabled).</para>
/// <para>Assistant turns after the last user turn keep their reasoning as <c>&lt;think&gt;…&lt;/think&gt;</c> (an
/// empty <c>&lt;think&gt;&lt;/think&gt;</c> when none is given and the content has no <c>&lt;/think&gt;</c>). Earlier
/// assistant turns drop it: their content is cut after its last <c>&lt;/think&gt;</c>, trimmed, and prefixed with an
/// empty <c>&lt;think&gt;&lt;/think&gt;</c>. <c>preserve_thinking</c> keeps every turn's reasoning.</para>
/// <para>Not supported (rejected with <see cref="NotSupportedException"/>): tools, tool calls and tool turns, image
/// and video parts, and a system turn after the first position. Text is inserted as given, as the template does: a
/// special token's text in a turn (for example <c>&lt;|im_end|&gt;</c>) becomes that special token when tokenized.</para>
/// </remarks>
public static class Lfm3ChatTemplate
{
    /// <summary>SHA-256 of the upstream <c>chat_template.jinja</c> this renderer follows (12,129 bytes).</summary>
    public const string TemplateSha256 = "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818";

    /// <summary>The template's <c>bos_token</c>.</summary>
    public const string BeginOfText = "<|startoftext|>";

    private const string ThinkClose = "</think>";

    /// <summary>The renderer as the runtime-independent <see cref="IChatTemplate"/>.</summary>
    public static IChatTemplate Instance { get; } = new Renderer();

    /// <summary>
    /// Renders <paramref name="turns"/> as the template does with <c>add_generation_prompt=True</c> (or false).
    /// </summary>
    /// <param name="turns">The conversation (at least one turn).</param>
    /// <param name="enableThinking">The template's <c>enable_thinking</c> (its default is true).</param>
    /// <param name="preserveThinking">The template's <c>preserve_thinking</c> (default false).</param>
    /// <param name="addGenerationPrompt">Append the assistant generation prompt (default true).</param>
    /// <exception cref="ArgumentException">The conversation is empty.</exception>
    /// <exception cref="NotSupportedException">A tool turn, tool calls, or a system turn that is not first.</exception>
    public static string Render(
        IReadOnlyList<ChatTurn> turns, bool enableThinking = true, bool preserveThinking = false, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(turns);
        if (turns.Count == 0)
        {
            throw new ArgumentException("The conversation needs at least one turn.", nameof(turns));
        }

        int start = 0;
        string systemPrompt = string.Empty;
        if (turns[0].Role == ChatTurnRole.System)
        {
            systemPrompt = turns[0].Content;
            start = 1;
        }

        int lastUser = -1;
        for (int i = start; i < turns.Count; i++)
        {
            ChatTurn turn = turns[i];
            ArgumentNullException.ThrowIfNull(turn);
            switch (turn.Role)
            {
                case ChatTurnRole.User:
                    lastUser = i;
                    break;
                case ChatTurnRole.Assistant when turn.ToolCalls is { Count: > 0 }:
                    throw new NotSupportedException("Tool calls are not supported by this LFM3 template port.");
                case ChatTurnRole.Assistant:
                    break;
                case ChatTurnRole.System:
                    throw new NotSupportedException("A system turn is supported only as the first turn.");
                default:
                    throw new NotSupportedException($"Turn role {turn.Role} is not supported by this LFM3 template port.");
            }
        }

        var sb = new StringBuilder(BeginOfText);
        if (systemPrompt.Length > 0)
        {
            sb.Append("<|im_start|>system\n").Append(systemPrompt).Append("<|im_end|>\n");
        }

        for (int i = start; i < turns.Count; i++)
        {
            ChatTurn turn = turns[i];
            if (turn.Role == ChatTurnRole.User)
            {
                sb.Append("<|im_start|>user\n").Append(turn.Content).Append("<|im_end|>\n");
                continue;
            }

            sb.Append("<|im_start|>assistant\n");
            bool keepThinking = preserveThinking || i > lastUser;
            string content = turn.Content;
            if (turn.ReasoningContent is { } reasoning && keepThinking)
            {
                sb.Append("<think>").Append(reasoning).Append(ThinkClose);
            }
            else if (keepThinking && !content.Contains(ThinkClose, StringComparison.Ordinal))
            {
                sb.Append("<think></think>");
            }

            if (!keepThinking)
            {
                int close = content.LastIndexOf(ThinkClose, StringComparison.Ordinal);
                if (close >= 0)
                {
                    content = PythonStrip(content[(close + ThinkClose.Length)..]);
                }

                content = "<think></think>" + content;
            }

            sb.Append(content).Append("<|im_end|>\n");
        }

        if (addGenerationPrompt)
        {
            sb.Append("<|im_start|>assistant\n").Append(enableThinking ? "<think>" : "<think></think>");
        }

        return sb.ToString();
    }

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
        public string TemplateSha256 => Lfm3ChatTemplate.TemplateSha256;

        public bool SupportsThinking => true;

        public string Render(IReadOnlyList<ChatTurn> turns, IReadOnlyList<ToolDefinition>? tools = null, bool enableThinking = false)
        {
            if (tools is { Count: > 0 })
            {
                throw new NotSupportedException("Tools are not supported by this LFM3 template port.");
            }

            return Lfm3ChatTemplate.Render(turns, enableThinking);
        }
    }
}
