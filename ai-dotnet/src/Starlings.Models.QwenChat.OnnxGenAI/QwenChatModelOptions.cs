namespace Starlings.Models.QwenChat;

/// <summary>Experimental CPU load settings; no provider fallback.</summary>
public sealed record QwenChatModelOptions
{
    /// <summary>Bounded native thread count, from 1 through 2.</summary>
    public int IntraOpThreads { get; init; } = 2;
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
