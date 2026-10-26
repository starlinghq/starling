namespace Starlings.Generation;

/// <summary>
/// A model's chat template rendered in C#: a conversation, tool definitions and the thinking switch in, the prompt text
/// (ending with the assistant generation prompt) out. Tokenize the text with the model's owned tokenizer without added
/// special tokens. Implementations are stateless and thread-safe.
/// </summary>
public interface IChatTemplate
{
    /// <summary>Lowercase hexadecimal SHA-256 of the upstream template text this renderer was ported from and is tested against.</summary>
    string TemplateSha256 { get; }

    /// <summary>True when the template has a thinking switch (<c>enable_thinking</c>) that <see cref="Render"/> honors.</summary>
    bool SupportsThinking { get; }

    /// <summary>Renders <paramref name="turns"/> and <paramref name="tools"/> followed by the assistant generation prompt.</summary>
    /// <param name="turns">The conversation (at least one turn).</param>
    /// <param name="tools">Tool definitions for the template's tools block, or null or empty for none.</param>
    /// <param name="enableThinking">The template's <c>enable_thinking</c>; true is rejected when <see cref="SupportsThinking"/> is false.</param>
    /// <exception cref="ArgumentException">The conversation is empty or a tool name cannot appear in the template's call format.</exception>
    /// <exception cref="NotSupportedException">
    /// The conversation or tool data is outside what the template supports. A model package may throw its own exception
    /// type instead, as its typed renderer does.
    /// </exception>
    string Render(IReadOnlyList<ChatTurn> turns, IReadOnlyList<ToolDefinition>? tools = null, bool enableThinking = false);
}
