// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Rendering logic ported from chat_template.jinja of Qwen/Qwen3.5-0.8B at commit
// 2fc06364715b967f1860aea9cf38778875588b17 (SHA-256 273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80),
// Copyright 2026 Alibaba Cloud, Apache License 2.0. Modified by StarlingHq: C# rendering of the text and tool-calling
// subset; vision content is rejected.

using System.Text;
using Starlings.Generation;

namespace Starlings.Models.Qwen35;

/// <summary>
/// The pinned chat template of <c>Qwen/Qwen3.5-0.8B</c> (<c>chat_template.jinja</c>, SHA-256
/// <see cref="TemplateSha256"/>) rendered in C# for text and tool-calling conversations, with the template's own
/// semantics (Jinja <c>trim</c> as Python <c>str.strip()</c>, reasoning extraction from <c>&lt;think&gt;</c> tags, the
/// generation prompt with or without thinking, transformers' <c>tojson</c> as Python <c>json.dumps</c>). Template
/// features outside that subset are rejected, never approximated.
/// </summary>
/// <remarks>
/// <para>
/// Supported: a non-empty list of system (first message only), user, assistant and tool messages with string content;
/// assistant reasoning content; assistant tool calls (<see cref="Qwen35Message.ToolCalls"/>) rendered as the template's
/// <c>&lt;tool_call&gt;&lt;function=...&gt;&lt;parameter=...&gt;</c> blocks; tool messages rendered as
/// <c>&lt;tool_response&gt;</c> blocks, consecutive ones in one user turn; tool definitions (<see cref="Qwen35Tool"/>)
/// rendered into the tools system block; <c>add_generation_prompt=True</c> and <c>enable_thinking</c> true or false
/// (undefined in the template behaves as false).
/// </para>
/// <para>
/// Unsupported (the API has no way to express them, or they fail with <see cref="Qwen35TemplateException"/>): image and
/// video content, <c>add_vision_id</c>, <c>add_generation_prompt=False</c>, tool data with duplicate JSON keys or lone
/// surrogates, and tool-call arguments that are not a JSON object. A conversation whose only user messages are wrapped in
/// <c>&lt;tool_response&gt;</c> (tool messages do not count) raises "No user query found in messages." exactly as the
/// template does.
/// </para>
/// <para>
/// This is the port that backlog P-029 qualifies on a template corpus. Until then it is experimental: it equals the
/// upstream rendering (transformers 5.18.0 <c>apply_chat_template</c>) on the 21 text prompts of
/// <c>prompts-v1.jsonl</c>, including <c>enable_thinking=true</c>, which ONNX Runtime GenAI's template engine cannot
/// render. It also equals the upstream rendering on the 105 supported cases of the proposed P-029 corpus
/// (<c>model-pipeline/proposed/qwen3.5-0.8b/template/</c>), 40 of them with tools, tool calls or tool results, and it
/// rejects that corpus's unsupported and invalid cases.
/// </para>
/// </remarks>
public static class Qwen35ChatTemplate
{
    /// <summary>SHA-256 of the pinned <c>chat_template.jinja</c> this renderer was ported from.</summary>
    public const string TemplateSha256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80";

    /// <summary>
    /// This renderer behind the backend-neutral <see cref="IChatTemplate"/> seam: the turns and tool definitions are
    /// converted (<see cref="Qwen35Message.FromChatTurn"/>, <see cref="Qwen35Tool.FromDefinition"/>) and rendered by
    /// <see cref="Render(IReadOnlyList{Qwen35Message}, IReadOnlyList{Qwen35Tool}?, bool)"/>, so the text is the same.
    /// An empty conversation and tool names the call format cannot carry throw <see cref="ArgumentException"/>; other
    /// invalid conversations throw <see cref="Qwen35TemplateException"/>.
    /// </summary>
    public static IChatTemplate Instance { get; } = new Renderer();

    private const string ImStart = "<|im_start|>";
    private const string ImEnd = "<|im_end|>";

    // The template's tool instructions, verbatim.
    private const string ToolsHeader = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    private const string ToolsInstructions =
        "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

    /// <summary>Renders a conversation followed by the assistant generation prompt.</summary>
    /// <param name="messages">The conversation.</param>
    /// <param name="enableThinking">
    /// The template's <c>enable_thinking</c>: true ends the prompt with <c>&lt;think&gt;\n</c> so the model reasons first;
    /// false (the template default) ends it with an empty think block.
    /// </param>
    /// <exception cref="Qwen35TemplateException">The conversation is invalid for the template.</exception>
    public static string Render(IReadOnlyList<Qwen35Message> messages, bool enableThinking = false) => Render(messages, tools: null, enableThinking);

    /// <summary>Renders a conversation with tool definitions, followed by the assistant generation prompt.</summary>
    /// <param name="messages">The conversation.</param>
    /// <param name="tools">
    /// The template's <c>tools</c>: rendered into the system block with the template's tool-call instructions. Null or
    /// empty renders no tools block (an empty list is falsy in the template).
    /// </param>
    /// <param name="enableThinking">The template's <c>enable_thinking</c>, as in <see cref="Render(IReadOnlyList{Qwen35Message}, bool)"/>.</param>
    /// <exception cref="Qwen35TemplateException">The conversation or tool data is invalid for the template.</exception>
    public static string Render(IReadOnlyList<Qwen35Message> messages, IReadOnlyList<Qwen35Tool>? tools, bool enableThinking = false)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (messages.Count == 0)
        {
            throw new Qwen35TemplateException("No messages provided.");
        }

        for (int i = 0; i < messages.Count; i++)
        {
            var m = messages[i] ?? throw new ArgumentException($"Message {i} is null.", nameof(messages));
            if (m.Content is null)
            {
                throw new ArgumentException($"Message {i} has null content.", nameof(messages));
            }

            if (!Enum.IsDefined(m.Role))
            {
                throw new Qwen35TemplateException("Unexpected message role.");
            }

            if (m.ReasoningContent is not null && m.Role != Qwen35Role.Assistant)
            {
                throw new Qwen35TemplateException($"Message {i}: reasoning content is only rendered for assistant messages.");
            }

            if (m.ToolCalls is { Count: > 0 } && m.Role != Qwen35Role.Assistant)
            {
                throw new Qwen35TemplateException($"Message {i}: tool calls are only rendered for assistant messages.");
            }
        }

        var sb = new StringBuilder();
        if (tools is { Count: > 0 })
        {
            sb.Append(ImStart).Append("system\n").Append(ToolsHeader);
            foreach (var tool in tools)
            {
                ArgumentNullException.ThrowIfNull(tool, nameof(tools));
                sb.Append('\n');
                Qwen35PythonJson.Dumps(sb, tool.Json);
            }

            sb.Append("\n</tools>").Append(ToolsInstructions);
            if (messages[0].Role == Qwen35Role.System)
            {
                string system = PyStrip(messages[0].Content);
                if (system.Length > 0)
                {
                    sb.Append("\n\n").Append(system);
                }
            }

            sb.Append(ImEnd).Append('\n');
        }
        else if (messages[0].Role == Qwen35Role.System)
        {
            sb.Append(ImStart).Append("system\n").Append(PyStrip(messages[0].Content)).Append(ImEnd).Append('\n');
        }

        // {%- for message in messages[::-1] %}: the last user message that is not a tool response.
        int lastQueryIndex = -1;
        for (int index = messages.Count - 1; index >= 0; index--)
        {
            if (messages[index].Role == Qwen35Role.User)
            {
                string content = PyStrip(messages[index].Content);
                if (!(content.StartsWith("<tool_response>", StringComparison.Ordinal) && content.EndsWith("</tool_response>", StringComparison.Ordinal)))
                {
                    lastQueryIndex = index;
                    break;
                }
            }
        }

        if (lastQueryIndex < 0)
        {
            throw new Qwen35TemplateException("No user query found in messages.");
        }

        for (int i = 0; i < messages.Count; i++)
        {
            var message = messages[i];
            string content = PyStrip(message.Content);
            switch (message.Role)
            {
                case Qwen35Role.System:
                    if (i != 0)
                    {
                        throw new Qwen35TemplateException("System message must be at the beginning.");
                    }

                    break;
                case Qwen35Role.User:
                    sb.Append(ImStart).Append("user\n").Append(content).Append(ImEnd).Append('\n');
                    break;
                case Qwen35Role.Assistant:
                    {
                        string reasoning = string.Empty;
                        if (message.ReasoningContent is not null)
                        {
                            reasoning = message.ReasoningContent;
                        }
                        else if (content.Contains("</think>", StringComparison.Ordinal))
                        {
                            // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                            string beforeClose = content[..content.IndexOf("</think>", StringComparison.Ordinal)].TrimEnd('\n');
                            int open = beforeClose.LastIndexOf("<think>", StringComparison.Ordinal);
                            reasoning = (open >= 0 ? beforeClose[(open + "<think>".Length)..] : beforeClose).TrimStart('\n');

                            // content.split('</think>')[-1].lstrip('\n')
                            content = content[(content.LastIndexOf("</think>", StringComparison.Ordinal) + "</think>".Length)..].TrimStart('\n');
                        }

                        reasoning = PyStrip(reasoning);
                        sb.Append(ImStart).Append("assistant\n");
                        if (i > lastQueryIndex)
                        {
                            sb.Append("<think>\n").Append(reasoning).Append("\n</think>\n\n");
                        }

                        sb.Append(content);
                        if (message.ToolCalls is { Count: > 0 } calls)
                        {
                            AppendToolCalls(sb, calls, contentIsBlank: PyStrip(content).Length == 0);
                        }

                        sb.Append(ImEnd).Append('\n');
                        break;
                    }

                case Qwen35Role.Tool:
                    // loop.previtem is undefined (falsy) for the first message.
                    if (i > 0 && messages[i - 1].Role != Qwen35Role.Tool)
                    {
                        sb.Append(ImStart).Append("user");
                    }

                    sb.Append("\n<tool_response>\n").Append(content).Append("\n</tool_response>");
                    if (i == messages.Count - 1 || messages[i + 1].Role != Qwen35Role.Tool)
                    {
                        sb.Append(ImEnd).Append('\n');
                    }

                    break;
            }
        }

        sb.Append(ImStart).Append("assistant\n").Append(enableThinking ? "<think>\n" : "<think>\n\n</think>\n\n");
        return sb.ToString();
    }

    private static void AppendToolCalls(StringBuilder sb, IReadOnlyList<Qwen35ToolCall> calls, bool contentIsBlank)
    {
        for (int j = 0; j < calls.Count; j++)
        {
            var call = calls[j] ?? throw new ArgumentException($"Tool call {j} is null.", nameof(calls));
            if (call.Name is null)
            {
                throw new Qwen35TemplateException($"Tool call {j} has no name.");
            }

            sb.Append(j == 0 ? (contentIsBlank ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=") : "\n<tool_call>\n<function=")
                .Append(call.Name).Append(">\n");
            switch (call.Arguments.ValueKind)
            {
                case System.Text.Json.JsonValueKind.Undefined:
                    break;
                case System.Text.Json.JsonValueKind.Object:
                    {
                        HashSet<string> seen = new(StringComparer.Ordinal);
                        foreach (var argument in call.Arguments.EnumerateObject())
                        {
                            if (!seen.Add(argument.Name))
                            {
                                throw new Qwen35TemplateException($"Tool call '{call.Name}' has the argument '{argument.Name}' twice.");
                            }

                            sb.Append("<parameter=").Append(argument.Name).Append(">\n")
                                .Append(Qwen35PythonJson.ArgumentText(argument.Value))
                                .Append("\n</parameter>\n");
                        }

                        break;
                    }

                default:
                    throw new Qwen35TemplateException($"Tool call '{call.Name}': arguments must be a JSON object, not {call.Arguments.ValueKind}.");
            }

            sb.Append("</function>\n</tool_call>");
        }
    }

    /// <summary>Python <c>str.strip()</c> (what Jinja's <c>trim</c> filter calls): strips <see cref="IsPythonSpace"/> characters.</summary>
    internal static string PyStrip(string s)
    {
        int start = 0;
        int end = s.Length;
        while (start < end && IsPythonSpace(s[start]))
        {
            start++;
        }

        while (end > start && IsPythonSpace(s[end - 1]))
        {
            end--;
        }

        return start == 0 && end == s.Length ? s : s[start..end];
    }

    /// <summary>
    /// Python's <c>str.isspace()</c> for one code point: bidirectional class WS, B or S, or general category Zs. Every
    /// such code point is in the BMP. Differs from <see cref="char.IsWhiteSpace(char)"/> on U+001C to U+001F (Python
    /// strips them, .NET does not).
    /// </summary>
    internal static bool IsPythonSpace(char c) => c switch
    {
        >= '\u0009' and <= '\u000D' => true,
        >= '\u001C' and <= '\u0020' => true,
        '\u0085' or '\u00A0' or '\u1680' => true,
        >= '\u2000' and <= '\u200A' => true,
        '\u2028' or '\u2029' or '\u202F' or '\u205F' or '\u3000' => true,
        _ => false,
    };
}

/// <summary><see cref="Qwen35ChatTemplate.Instance"/>.</summary>
file sealed class Renderer : IChatTemplate
{
    public string TemplateSha256 => Qwen35ChatTemplate.TemplateSha256;

    public bool SupportsThinking => true;

    public string Render(IReadOnlyList<ChatTurn> turns, IReadOnlyList<ToolDefinition>? tools = null, bool enableThinking = false)
    {
        ArgumentNullException.ThrowIfNull(turns);
        if (turns.Count == 0)
        {
            // IChatTemplate's contract for an empty conversation (the typed renderer throws Qwen35TemplateException).
            throw new ArgumentException("At least one turn is required.", nameof(turns));
        }

        return Qwen35ChatTemplate.Render(
            [.. turns.Select(Qwen35Message.FromChatTurn)],
            tools is null ? null : [.. tools.Select(Qwen35Tool.FromDefinition)],
            enableThinking);
    }
}

/// <summary>A conversation the pinned chat template rejects, or a template feature this package does not render.</summary>
public sealed class Qwen35TemplateException : Exception
{
    /// <summary>Creates the exception.</summary>
    public Qwen35TemplateException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public Qwen35TemplateException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public Qwen35TemplateException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
