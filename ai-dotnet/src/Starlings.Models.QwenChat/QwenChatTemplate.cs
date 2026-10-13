// Text rendering adapted from the pinned Qwen tokenizer_config.json chat_template strings.
// Copyright Alibaba Cloud. Apache-2.0. See THIRD-PARTY-NOTICES.md.
using System.Text;

namespace Starlings.Models.QwenChat;

/// <summary>Owned rendering of the admitted upstream text-only templates.</summary>
public static class QwenChatTemplate
{
    /// <summary>Render a conversation followed by the assistant generation prompt.</summary>
    public static string Render(QwenChatProfile profile, IReadOnlyList<QwenChatMessage> messages)
    {
        ArgumentNullException.ThrowIfNull(profile);
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
                throw new NotSupportedException("Only system, user and assistant text messages are supported.");
            }

            if (message.ReasoningContent is not null && message.Role != QwenChatRole.Assistant)
            {
                throw new NotSupportedException("Reasoning content can only be attached to assistant history.");
            }
        }

        var result = new StringBuilder();
        if (profile == QwenChatProfile.Qwen25 && messages[0].Role != QwenChatRole.System)
        {
            result.Append("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n");
        }

        int lastQuery = messages.Count - 1;
        if (profile == QwenChatProfile.Qwen3)
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
            string content = message.Content;
            if (profile == QwenChatProfile.Qwen3 && message.Role == QwenChatRole.Assistant)
            {
                string reasoning = message.ReasoningContent ?? string.Empty;
                if (message.ReasoningContent is null && content.Contains("</think>", StringComparison.Ordinal))
                {
                    int firstClose = content.IndexOf("</think>", StringComparison.Ordinal);
                    string before = content[..firstClose].TrimEnd('\n');
                    int lastOpen = before.LastIndexOf("<think>", StringComparison.Ordinal);
                    reasoning = (lastOpen < 0 ? before : before[(lastOpen + 7)..]).TrimStart('\n');
                    content = content[(content.LastIndexOf("</think>", StringComparison.Ordinal) + 8)..].TrimStart('\n');
                }

                if (index > lastQuery && (index == messages.Count - 1 || reasoning.Length != 0))
                {
                    content = "<think>\n" + reasoning.Trim('\n') + "\n</think>\n\n" + content.TrimStart('\n');
                }
            }

            result.Append("<|im_start|>").Append(message.Role.ToString().ToLowerInvariant()).Append('\n')
                .Append(content).Append("<|im_end|>\n");
        }

        result.Append("<|im_start|>assistant\n");
        if (profile == QwenChatProfile.Qwen3)
        {
            result.Append("<think>\n\n</think>\n\n");
        }

        return result.ToString();
    }
}
