// Text and tool rendering adapted from the pinned Qwen tokenizer_config.json chat_template strings.
// Copyright Alibaba Cloud. Apache-2.0. See THIRD-PARTY-NOTICES.md.
using System.Text;
using System.Text.Json;
using Starlings.Generation;

namespace Starlings.Models.QwenChat;

/// <summary>Owned rendering of the admitted upstream templates for text and tool-calling conversations.</summary>
/// <remarks>
/// <para>
/// Supported: a non-empty list of system, user, assistant and tool messages with string content; assistant reasoning
/// content (Qwen3); assistant tool calls (<see cref="QwenChatMessage.ToolCalls"/>) rendered as the templates'
/// <c>&lt;tool_call&gt;</c> JSON blocks; tool messages rendered as <c>&lt;tool_response&gt;</c> blocks, consecutive
/// ones in one user turn; tool definitions (<see cref="QwenChatTool"/>) rendered into the tools system block with the
/// templates' instructions, verbatim. The generation prompt is always appended; Qwen3 uses <c>enable_thinking=false</c>.
/// </para>
/// <para>
/// Rejected with <see cref="NotSupportedException"/>: reasoning content or tool calls on a message that is not an
/// assistant message, tool-call arguments that are not a JSON object, tool-call names the templates cannot write as
/// JSON, and tool data with duplicate JSON keys or lone surrogates.
/// </para>
/// <para>
/// Experimental: equal to transformers 5.18.0 <c>apply_chat_template</c> on the render cases of the proposed template
/// corpora (<c>model-pipeline/proposed/qwen3-0.6b/template/</c>, <c>model-pipeline/proposed/qwen2.5-0.5b-instruct/template/</c>).
/// </para>
/// </remarks>
public static class QwenChatTemplate
{
    private const string ImStart = "<|im_start|>";
    private const string ImEnd = "<|im_end|>";
    private const string Qwen25DefaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    // The templates' tool instructions, verbatim (identical in both).
    private const string ToolsHeader =
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    private const string ToolsInstructions =
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

    private static readonly IChatTemplate Qwen3Renderer = new Renderer(QwenChatCheckpoint.Qwen3);
    private static readonly IChatTemplate Qwen25Renderer = new Renderer(QwenChatCheckpoint.Qwen25);

    /// <summary>
    /// The renderer of <paramref name="checkpoint"/>'s template behind the backend-neutral <see cref="IChatTemplate"/>
    /// seam: turns and tool definitions are converted (<see cref="QwenChatMessage.FromChatTurn"/>,
    /// <see cref="QwenChatTool.FromDefinition"/>) and rendered by
    /// <see cref="Render(QwenChatCheckpoint, IReadOnlyList{QwenChatMessage}, IReadOnlyList{QwenChatTool}?)"/>, so the text
    /// is the same. <see cref="IChatTemplate.SupportsThinking"/> is false: Qwen3 renders with
    /// <c>enable_thinking=false</c>, and <c>enableThinking: true</c> throws <see cref="NotSupportedException"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">Not an admitted checkpoint.</exception>
    public static IChatTemplate For(QwenChatCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        return ReferenceEquals(checkpoint, QwenChatCheckpoint.Qwen3) ? Qwen3Renderer
            : ReferenceEquals(checkpoint, QwenChatCheckpoint.Qwen25) ? Qwen25Renderer
            : throw new NotSupportedException($"Unsupported Qwen chat checkpoint: {checkpoint.Id}");
    }

    /// <summary>Render a conversation followed by the assistant generation prompt.</summary>
    /// <param name="checkpoint">The admitted checkpoint whose template is rendered (a <c>QwenChatProfile</c> converts implicitly).</param>
    /// <param name="messages">The conversation.</param>
    public static string Render(QwenChatCheckpoint checkpoint, IReadOnlyList<QwenChatMessage> messages) => Render(checkpoint, messages, tools: null);

    /// <summary>Render a conversation with tool definitions, followed by the assistant generation prompt.</summary>
    /// <param name="checkpoint">The admitted checkpoint whose template is rendered (a <c>QwenChatProfile</c> converts implicitly).</param>
    /// <param name="messages">The conversation.</param>
    /// <param name="tools">
    /// The template's <c>tools</c>: rendered into the system block with the template's tool-call instructions. Null or
    /// empty renders no tools block (an empty list is falsy in the template). Experimental.
    /// </param>
    /// <exception cref="NotSupportedException">The conversation or tool data is outside the supported subset.</exception>
    public static string Render(QwenChatCheckpoint checkpoint, IReadOnlyList<QwenChatMessage> messages, IReadOnlyList<QwenChatTool>? tools)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException("At least one text message is required.", nameof(messages));
        }

        for (int index = 0; index < messages.Count; index++)
        {
            QwenChatMessage message = messages[index] ?? throw new ArgumentException("Null message.", nameof(messages));
            ArgumentNullException.ThrowIfNull(message.Content);
            if (!Enum.IsDefined(message.Role))
            {
                throw new NotSupportedException("Only system, user, assistant and tool text messages are supported.");
            }

            if (message.ReasoningContent is not null && message.Role != QwenChatRole.Assistant)
            {
                throw new NotSupportedException("Reasoning content can only be attached to assistant history.");
            }

            if (message.ToolCalls is { Count: > 0 } && message.Role != QwenChatRole.Assistant)
            {
                throw new NotSupportedException("Tool calls can only be attached to assistant history.");
            }
        }

        bool qwen3 = ReferenceEquals(checkpoint, QwenChatCheckpoint.Qwen3);
        bool firstIsSystem = messages[0].Role == QwenChatRole.System;
        var result = new StringBuilder();
        if (tools is { Count: > 0 })
        {
            result.Append(ImStart).Append("system\n");
            if (qwen3)
            {
                if (firstIsSystem)
                {
                    result.Append(messages[0].Content).Append("\n\n");
                }
            }
            else
            {
                result.Append(firstIsSystem ? messages[0].Content : Qwen25DefaultSystem).Append("\n\n");
            }

            result.Append(ToolsHeader);
            foreach (QwenChatTool tool in tools)
            {
                ArgumentNullException.ThrowIfNull(tool, nameof(tools));
                result.Append('\n');
                QwenChatPythonJson.Dumps(result, tool.Json);
            }

            result.Append(ToolsInstructions);
        }
        else if (firstIsSystem)
        {
            result.Append(ImStart).Append("system\n").Append(messages[0].Content).Append(ImEnd).Append('\n');
        }
        else if (!qwen3)
        {
            result.Append(ImStart).Append("system\n").Append(Qwen25DefaultSystem).Append(ImEnd).Append('\n');
        }

        // Qwen3: the last user message that is not a wrapped tool response (tool messages do not count).
        int lastQuery = messages.Count - 1;
        if (qwen3)
        {
            for (int index = messages.Count - 1; index >= 0; index--)
            {
                string content = messages[index].Content;
                if (messages[index].Role == QwenChatRole.User &&
                    !(content.StartsWith("<tool_response>", StringComparison.Ordinal) && content.EndsWith("</tool_response>", StringComparison.Ordinal)))
                {
                    lastQuery = index;
                    break;
                }
            }
        }

        for (int index = 0; index < messages.Count; index++)
        {
            QwenChatMessage message = messages[index];
            switch (message.Role)
            {
                case QwenChatRole.System when index == 0:
                    // Rendered in the header above.
                    break;
                case QwenChatRole.System or QwenChatRole.User:
                    AppendTurn(result, message.Role, message.Content);
                    break;
                case QwenChatRole.Assistant when qwen3:
                    AppendQwen3Assistant(result, message, index, lastQuery, index == messages.Count - 1);
                    break;
                case QwenChatRole.Assistant when message.ToolCalls is not { Count: > 0 }:
                    AppendTurn(result, message.Role, message.Content);
                    break;
                case QwenChatRole.Assistant:
                    result.Append(ImStart).Append("assistant");
                    if (message.Content.Length != 0)
                    {
                        result.Append('\n').Append(message.Content);
                    }

                    foreach (QwenChatToolCall call in message.ToolCalls!)
                    {
                        result.Append('\n');
                        AppendToolCall(result, call);
                    }

                    result.Append(ImEnd).Append('\n');
                    break;
                default:
                    // Tool: consecutive results share one user turn.
                    if (index == 0 || messages[index - 1].Role != QwenChatRole.Tool)
                    {
                        result.Append(ImStart).Append("user");
                    }

                    result.Append("\n<tool_response>\n").Append(message.Content).Append("\n</tool_response>");
                    if (index == messages.Count - 1 || messages[index + 1].Role != QwenChatRole.Tool)
                    {
                        result.Append(ImEnd).Append('\n');
                    }

                    break;
            }
        }

        result.Append(ImStart).Append("assistant\n");
        if (qwen3)
        {
            result.Append("<think>\n\n</think>\n\n");
        }

        return result.ToString();
    }

    private static void AppendTurn(StringBuilder result, QwenChatRole role, string content) =>
        result.Append(ImStart).Append(role == QwenChatRole.System ? "system" : role == QwenChatRole.User ? "user" : "assistant").Append('\n')
            .Append(content).Append(ImEnd).Append('\n');

    private static void AppendQwen3Assistant(StringBuilder result, QwenChatMessage message, int index, int lastQuery, bool last)
    {
        string content = message.Content;
        string reasoning = message.ReasoningContent ?? string.Empty;
        if (message.ReasoningContent is null && content.Contains("</think>", StringComparison.Ordinal))
        {
            int firstClose = content.IndexOf("</think>", StringComparison.Ordinal);
            string before = content[..firstClose].TrimEnd('\n');
            int lastOpen = before.LastIndexOf("<think>", StringComparison.Ordinal);
            reasoning = (lastOpen < 0 ? before : before[(lastOpen + 7)..]).TrimStart('\n');
            content = content[(content.LastIndexOf("</think>", StringComparison.Ordinal) + 8)..].TrimStart('\n');
        }

        result.Append(ImStart).Append("assistant\n");
        if (index > lastQuery && (last || reasoning.Length != 0))
        {
            result.Append("<think>\n").Append(reasoning.Trim('\n')).Append("\n</think>\n\n").Append(content.TrimStart('\n'));
        }
        else
        {
            result.Append(content);
        }

        if (message.ToolCalls is { Count: > 0 } calls)
        {
            for (int i = 0; i < calls.Count; i++)
            {
                // The template tests its content variable (before lstrip), not the rendered text.
                if (i > 0 || content.Length != 0)
                {
                    result.Append('\n');
                }

                AppendToolCall(result, calls[i]);
            }
        }

        result.Append(ImEnd).Append('\n');
    }

    private static void AppendToolCall(StringBuilder result, QwenChatToolCall call)
    {
        ArgumentNullException.ThrowIfNull(call);
        if (!QwenChatToolCall.IsValidName(call.Name))
        {
            throw new NotSupportedException(
                $"Tool-call name '{call.Name}' is empty or contains '\"', '\\' or a control character, which the template writes unescaped into the call JSON.");
        }

        if (call.Arguments.ValueKind != JsonValueKind.Object)
        {
            throw new NotSupportedException(
                "Tool-call arguments must be a JSON object (upstream raises for missing arguments, and the Qwen3 and Qwen2.5 templates render a JSON string differently).");
        }

        result.Append("<tool_call>\n{\"name\": \"").Append(QwenChatPythonJson.CheckedString(call.Name)).Append("\", \"arguments\": ");
        QwenChatPythonJson.Dumps(result, call.Arguments);
        result.Append("}\n</tool_call>");
    }
}

/// <summary><see cref="QwenChatTemplate.For"/>.</summary>
file sealed class Renderer(QwenChatCheckpoint checkpoint) : IChatTemplate
{
    public string TemplateSha256 => checkpoint.TemplateSha256;

    public bool SupportsThinking => false;

    public string Render(IReadOnlyList<ChatTurn> turns, IReadOnlyList<ToolDefinition>? tools = null, bool enableThinking = false)
    {
        ArgumentNullException.ThrowIfNull(turns);
        if (enableThinking)
        {
            throw new NotSupportedException("These profiles render with enable_thinking=false; thinking mode is unsupported.");
        }

        return QwenChatTemplate.Render(
            checkpoint,
            [.. turns.Select(QwenChatMessage.FromChatTurn)],
            tools is null ? null : [.. tools.Select(QwenChatTool.FromDefinition)]);
    }
}
