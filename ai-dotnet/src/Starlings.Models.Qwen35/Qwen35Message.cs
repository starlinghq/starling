using Starlings.Generation;

namespace Starlings.Models.Qwen35;

/// <summary>Role of a chat message.</summary>
public enum Qwen35Role
{
    /// <summary>System instructions; only allowed as the first message.</summary>
    System,

    /// <summary>A user turn.</summary>
    User,

    /// <summary>An earlier assistant turn, optionally with tool calls (<see cref="Qwen35Message.ToolCalls"/>).</summary>
    Assistant,

    /// <summary>
    /// The result of one tool call. Consecutive tool messages are rendered into one user turn of
    /// <c>&lt;tool_response&gt;</c> blocks, as the template does.
    /// </summary>
    Tool,
}

/// <summary>One message of a conversation.</summary>
/// <param name="Role">The role.</param>
/// <param name="Content">
/// Plain text content (may be empty). The template trims surrounding whitespace, as upstream. For a
/// <see cref="Qwen35Role.Tool"/> message: the tool's result as text.
/// </param>
/// <param name="ReasoningContent">
/// Assistant only: the reasoning of that turn, rendered inside <c>&lt;think&gt;</c> for assistant turns after the last
/// user query (the template drops it for earlier turns). Null means not given: the template then extracts reasoning
/// from <c>&lt;think&gt;...&lt;/think&gt;</c> in <paramref name="Content"/>, as upstream.
/// </param>
public sealed record Qwen35Message(Qwen35Role Role, string Content, string? ReasoningContent = null)
{
    /// <summary>
    /// Assistant only: the tool calls of that turn, rendered after <see cref="Content"/> as the template's
    /// <c>&lt;tool_call&gt;</c> blocks. Null or empty: no tool calls.
    /// </summary>
    public IReadOnlyList<Qwen35ToolCall>? ToolCalls { get; init; }

    /// <summary>A system message.</summary>
    public static Qwen35Message System(string content) => new(Qwen35Role.System, content);

    /// <summary>A user message.</summary>
    public static Qwen35Message User(string content) => new(Qwen35Role.User, content);

    /// <summary>An assistant message.</summary>
    public static Qwen35Message Assistant(string content, string? reasoningContent = null) => new(Qwen35Role.Assistant, content, reasoningContent);

    /// <summary>An assistant message with tool calls (and optional text before them).</summary>
    public static Qwen35Message Assistant(string content, IReadOnlyList<Qwen35ToolCall> toolCalls, string? reasoningContent = null) =>
        new(Qwen35Role.Assistant, content, reasoningContent) { ToolCalls = toolCalls };

    /// <summary>A tool result message.</summary>
    public static Qwen35Message Tool(string content) => new(Qwen35Role.Tool, content);

    /// <summary>The message of a backend-neutral <see cref="ChatTurn"/> (role, content, reasoning and tool calls).</summary>
    public static Qwen35Message FromChatTurn(ChatTurn turn)
    {
        ArgumentNullException.ThrowIfNull(turn);
        Qwen35Role role = turn.Role switch
        {
            ChatTurnRole.System => Qwen35Role.System,
            ChatTurnRole.User => Qwen35Role.User,
            ChatTurnRole.Assistant => Qwen35Role.Assistant,
            ChatTurnRole.Tool => Qwen35Role.Tool,
            _ => throw new ArgumentOutOfRangeException(nameof(turn), turn.Role, "Unknown role."),
        };
        return new Qwen35Message(role, turn.Content, turn.ReasoningContent)
        {
            ToolCalls = turn.ToolCalls is null ? null : [.. turn.ToolCalls.Select(Qwen35ToolCall.FromToolCall)],
        };
    }

    /// <summary>This message as a backend-neutral <see cref="ChatTurn"/>.</summary>
    public ChatTurn ToChatTurn()
    {
        ChatTurnRole role = Role switch
        {
            Qwen35Role.System => ChatTurnRole.System,
            Qwen35Role.User => ChatTurnRole.User,
            Qwen35Role.Assistant => ChatTurnRole.Assistant,
            Qwen35Role.Tool => ChatTurnRole.Tool,
            _ => throw new InvalidOperationException($"Unknown role {Role}."),
        };
        return new ChatTurn(role, Content, ReasoningContent)
        {
            ToolCalls = ToolCalls is null ? null : [.. ToolCalls.Select(c => c.ToToolCall())],
        };
    }
}
