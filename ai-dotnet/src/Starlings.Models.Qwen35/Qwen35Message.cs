namespace Starlings.Models.Qwen35;

/// <summary>Role of a chat message. Tool messages are not supported by this package version.</summary>
public enum Qwen35Role
{
    /// <summary>System instructions; only allowed as the first message.</summary>
    System,

    /// <summary>A user turn.</summary>
    User,

    /// <summary>An earlier assistant turn.</summary>
    Assistant,
}

/// <summary>One text message of a conversation.</summary>
/// <param name="Role">The role.</param>
/// <param name="Content">Plain text content (may be empty). The template trims surrounding whitespace, as upstream.</param>
/// <param name="ReasoningContent">
/// Assistant only: the reasoning of that turn, rendered inside <c>&lt;think&gt;</c> for assistant turns after the last
/// user query (the template drops it for earlier turns). Null means not given: the template then extracts reasoning
/// from <c>&lt;think&gt;...&lt;/think&gt;</c> in <paramref name="Content"/>, as upstream.
/// </param>
public sealed record Qwen35Message(Qwen35Role Role, string Content, string? ReasoningContent = null)
{
    /// <summary>A system message.</summary>
    public static Qwen35Message System(string content) => new(Qwen35Role.System, content);

    /// <summary>A user message.</summary>
    public static Qwen35Message User(string content) => new(Qwen35Role.User, content);

    /// <summary>An assistant message.</summary>
    public static Qwen35Message Assistant(string content, string? reasoningContent = null) => new(Qwen35Role.Assistant, content, reasoningContent);
}
