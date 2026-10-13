// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Rendering logic ported from chat_template.jinja of Qwen/Qwen3.5-0.8B at commit
// 2fc06364715b967f1860aea9cf38778875588b17 (SHA-256 273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80),
// Copyright 2026 Alibaba Cloud, Apache License 2.0. Modified by StarlingHq: C# rendering of the text
// subset; tools, tool calls, tool messages and vision content are rejected.

using System.Text;

namespace Starlings.Models.Qwen35;

/// <summary>
/// The pinned chat template of <c>Qwen/Qwen3.5-0.8B</c> (<c>chat_template.jinja</c>, SHA-256
/// <see cref="TemplateSha256"/>) rendered in C# for text conversations, with the template's own semantics (Jinja
/// <c>trim</c> as Python <c>str.strip()</c>, reasoning extraction from <c>&lt;think&gt;</c> tags, the generation prompt
/// with or without thinking). Template features outside that subset are rejected, never approximated.
/// </summary>
/// <remarks>
/// <para>
/// Supported: a non-empty list of system (first message only), user and assistant messages with string content,
/// assistant reasoning content, <c>add_generation_prompt=True</c> and <c>enable_thinking</c> true or false (undefined in
/// the template behaves as false).
/// </para>
/// <para>
/// Unsupported (the API has no way to express them, or they fail with <see cref="Qwen35TemplateException"/>): tool
/// definitions and the tools system block, assistant tool calls, tool-role messages, image and video content,
/// <c>add_vision_id</c>, <c>add_generation_prompt=False</c>. A conversation whose only user messages are wrapped in
/// <c>&lt;tool_response&gt;</c> raises "No user query found in messages." exactly as the template does.
/// </para>
/// <para>
/// This is the port that backlog P-029 qualifies on a template corpus. Until then it is experimental: it equals the
/// upstream rendering (transformers 5.18.0 <c>apply_chat_template</c>) on the 21 text prompts of
/// <c>prompts-v1.jsonl</c>, including <c>enable_thinking=true</c>, which ONNX Runtime GenAI's template engine cannot
/// render.
/// </para>
/// </remarks>
public static class Qwen35ChatTemplate
{
    /// <summary>SHA-256 of the pinned <c>chat_template.jinja</c> this renderer was ported from.</summary>
    public const string TemplateSha256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80";

    private const string ImStart = "<|im_start|>";
    private const string ImEnd = "<|im_end|>";

    /// <summary>Renders a conversation followed by the assistant generation prompt.</summary>
    /// <param name="messages">The conversation.</param>
    /// <param name="enableThinking">
    /// The template's <c>enable_thinking</c>: true ends the prompt with <c>&lt;think&gt;\n</c> so the model reasons first;
    /// false (the template default) ends it with an empty think block.
    /// </param>
    /// <exception cref="Qwen35TemplateException">The conversation is invalid for the template.</exception>
    public static string Render(IReadOnlyList<Qwen35Message> messages, bool enableThinking = false)
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
        }

        var sb = new StringBuilder();
        if (messages[0].Role == Qwen35Role.System)
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

                        sb.Append(content).Append(ImEnd).Append('\n');
                        break;
                    }
            }
        }

        sb.Append(ImStart).Append("assistant\n").Append(enableThinking ? "<think>\n" : "<think>\n\n</think>\n\n");
        return sb.ToString();
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
