using Microsoft.Extensions.AI;

namespace Starlings.Generation;

/// <summary>
/// The Microsoft.Extensions.AI view of a generation result, as Starling's <c>IChatClient</c> adapters report it: finish
/// reasons and usage.
/// </summary>
public static class GenerationChatMapping
{
    /// <summary>
    /// The chat finish reason of a generation that ended with <paramref name="reason"/>:
    /// <see cref="ChatFinishReason.ToolCalls"/> when <paramref name="toolCalls"/> is true and generation stopped (a stop
    /// token or a stop sequence), else <see cref="ChatFinishReason.Stop"/> for a stop and
    /// <see cref="ChatFinishReason.Length"/> for the token limit.
    /// </summary>
    /// <param name="reason">Why generation ended.</param>
    /// <param name="toolCalls">True when the completion held at least one parsed tool call.</param>
    /// <exception cref="ArgumentOutOfRangeException">An undefined <paramref name="reason"/>.</exception>
    public static ChatFinishReason ToChatFinishReason(this GenerationFinishReason reason, bool toolCalls = false) => reason switch
    {
        GenerationFinishReason.Stop or GenerationFinishReason.StopSequence => toolCalls ? ChatFinishReason.ToolCalls : ChatFinishReason.Stop,
        GenerationFinishReason.Length => ChatFinishReason.Length,
        _ => throw new ArgumentOutOfRangeException(nameof(reason), reason, "Unknown finish reason."),
    };

    /// <summary>The usage details of <paramref name="usage"/>: input, output and total token counts.</summary>
    public static UsageDetails ToUsageDetails(this GenerationUsage usage) => new()
    {
        InputTokenCount = usage.PromptTokens,
        OutputTokenCount = usage.CompletionTokens,
        TotalTokenCount = usage.TotalTokens,
    };

    /// <summary>The chat role of a turn role.</summary>
    /// <exception cref="ArgumentOutOfRangeException">An undefined <paramref name="role"/>.</exception>
    public static ChatRole ToChatRole(this ChatTurnRole role) => role switch
    {
        ChatTurnRole.System => ChatRole.System,
        ChatTurnRole.User => ChatRole.User,
        ChatTurnRole.Assistant => ChatRole.Assistant,
        ChatTurnRole.Tool => ChatRole.Tool,
        _ => throw new ArgumentOutOfRangeException(nameof(role), role, "Unknown role."),
    };
}
