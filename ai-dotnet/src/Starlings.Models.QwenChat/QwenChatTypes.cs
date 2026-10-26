using System.Collections.Immutable;
using Starlings.Generation.OnnxGenAI;

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
}

/// <summary>A rendered prompt with owned tokenizer ids.</summary>
public sealed record QwenChatPrompt(string Text, ImmutableArray<int> TokenIds);

/// <summary>Experimental CPU load settings; no provider fallback.</summary>
public sealed record QwenChatModelOptions
{
    /// <summary>Bounded native thread count, from 1 through 2.</summary>
    public int IntraOpThreads { get; init; } = 2;
}

/// <summary>Per-request settings. Qwen3 explicitly uses enable_thinking=false.</summary>
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
    /// Tool definitions rendered into the template's tools block by <see cref="QwenChatModel.Generate(IReadOnlyList{QwenChatMessage}, QwenChatOptions?, CancellationToken)"/>
    /// (default none). The completion may then contain tool calls; split them out with
    /// <see cref="QwenChatToolCallParser"/>. Experimental.
    /// </summary>
    public IReadOnlyList<QwenChatTool> Tools { get; init; } = [];

    internal GenerationOptions ToGenerationOptions() => new()
    {
        MaxNewTokens = MaxNewTokens,
        AdditionalStopTokenIds = AdditionalStopTokenIds,
        StopSequences = StopSequences,
        PrefillChunkTokens = PrefillChunkTokens,
    };
}

/// <summary>Loaded profile identity and explicitly scoped capabilities.</summary>
public sealed record QwenChatModelInfo(
    string Profile, string Repository, string Revision, string TokenizerSha256, string TemplateSha256,
    string GenAIVersion, string? NativePackage, int ContextLength)
{
    /// <summary>The profile is experimental pending independent protected fixture adoption.</summary>
    public string TextGeneration { get; } = "experimental";
    /// <summary>
    /// Tool calling is experimental: the tools block, assistant tool calls and tool messages of the owned template
    /// equal the upstream rendering on the proposed template corpus, and <see cref="QwenChatToolCallParser"/> reads the
    /// templates' JSON call form. The model's choice of tools is not qualified.
    /// </summary>
    public string Tools { get; } = "experimental";
    /// <summary>Multimodal input is unsupported.</summary>
    public string Multimodal { get; } = "unsupported";
    /// <summary>Each request gets a fresh generator; weights are shared, conversation cache state is never reused.</summary>
    public string Cache { get; } = "fresh-request";
    /// <summary>Qwen3 thinking is explicitly disabled for this profile.</summary>
    public bool EnableThinking { get; }
    /// <summary>Unused embedding row used for padding so literal endoftext prompt positions remain attended.</summary>
    public int PadTokenIdOverlay { get; } = 151935;
}
