using System.Collections.Immutable;
using Starlings.Generation;

namespace Starlings.Models.QwenChat;

/// <summary>The roles supported by the experimental text profiles.</summary>
public enum QwenChatRole
{
    /// <summary>A system instruction.</summary>
    System,
    /// <summary>A user text message.</summary>
    User,
    /// <summary>An assistant text message, optionally with tool calls (<see cref="QwenChatMessage.ToolCalls"/>).</summary>
    Assistant,
    /// <summary>
    /// The result of one tool call (experimental). Consecutive tool messages are rendered into one user turn of
    /// <c>&lt;tool_response&gt;</c> blocks, as the templates do.
    /// </summary>
    Tool,
}

/// <summary>A text message; multimodal input has no representation in this profile.</summary>
/// <param name="Role">The role.</param>
/// <param name="Content">Plain text content (may be empty), rendered as given. For a tool message: the tool's result as text.</param>
/// <param name="ReasoningContent">Assistant only: the reasoning of that turn (Qwen3 renders it after the last user query; Qwen2.5 ignores it).</param>
public sealed record QwenChatMessage(QwenChatRole Role, string Content, string? ReasoningContent = null)
{
    /// <summary>
    /// Assistant only: the tool calls of that turn, rendered after <see cref="Content"/> as the templates'
    /// <c>&lt;tool_call&gt;</c> JSON blocks. Null or empty: no tool calls. Experimental.
    /// </summary>
    public IReadOnlyList<QwenChatToolCall>? ToolCalls { get; init; }

    /// <summary>The message of a backend-neutral <see cref="ChatTurn"/> (role, content, reasoning and tool calls).</summary>
    public static QwenChatMessage FromChatTurn(ChatTurn turn)
    {
        ArgumentNullException.ThrowIfNull(turn);
        QwenChatRole role = turn.Role switch
        {
            ChatTurnRole.System => QwenChatRole.System,
            ChatTurnRole.User => QwenChatRole.User,
            ChatTurnRole.Assistant => QwenChatRole.Assistant,
            ChatTurnRole.Tool => QwenChatRole.Tool,
            _ => throw new ArgumentOutOfRangeException(nameof(turn), turn.Role, "Unknown role."),
        };
        return new QwenChatMessage(role, turn.Content, turn.ReasoningContent)
        {
            ToolCalls = turn.ToolCalls is null ? null : [.. turn.ToolCalls.Select(QwenChatToolCall.FromToolCall)],
        };
    }

    /// <summary>This message as a backend-neutral <see cref="ChatTurn"/>.</summary>
    public ChatTurn ToChatTurn()
    {
        ChatTurnRole role = Role switch
        {
            QwenChatRole.System => ChatTurnRole.System,
            QwenChatRole.User => ChatTurnRole.User,
            QwenChatRole.Assistant => ChatTurnRole.Assistant,
            QwenChatRole.Tool => ChatTurnRole.Tool,
            _ => throw new InvalidOperationException($"Unknown role {Role}."),
        };
        return new ChatTurn(role, Content, ReasoningContent)
        {
            ToolCalls = ToolCalls is null ? null : [.. ToolCalls.Select(c => c.ToToolCall())],
        };
    }
}

/// <summary>A rendered prompt with owned tokenizer ids.</summary>
public sealed record QwenChatPrompt(string Text, ImmutableArray<int> TokenIds);

/// <summary>Per-request settings of <c>QwenChatModel</c>. Qwen3 explicitly uses enable_thinking=false.</summary>
public sealed record QwenChatOptions
{
    /// <summary>Maximum completion tokens, excluding an EOS or stop id.</summary>
    public int MaxNewTokens { get; init; } = 16;

    /// <summary>Additional token ids that terminate generation and are excluded from output.</summary>
    public IReadOnlyCollection<int> AdditionalStopTokenIds { get; init; } = [];

    /// <summary>Strings that terminate generation and are excluded from output text.</summary>
    public IReadOnlyList<string> StopSequences { get; init; } = [];

    /// <summary>Optional bounded prefill chunks; null runs a single prefill.</summary>
    public int? PrefillChunkTokens { get; init; }

    /// <summary>
    /// Tool definitions rendered into the template's tools block by <c>QwenChatModel.Generate</c> (Starlings.Models.QwenChat.OnnxGenAI)
    /// (default none). The completion may then contain tool calls; split them out with
    /// <see cref="QwenChatToolCallParser"/>. Experimental.
    /// </summary>
    public IReadOnlyList<QwenChatTool> Tools { get; init; } = [];

    /// <summary>The decode-loop options of this request (greedy).</summary>
    public GenerationOptions ToGenerationOptions() => new()
    {
        MaxNewTokens = MaxNewTokens,
        AdditionalStopTokenIds = AdditionalStopTokenIds,
        StopSequences = StopSequences,
        PrefillChunkTokens = PrefillChunkTokens,
    };
}
