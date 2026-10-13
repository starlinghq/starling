using System.Collections.Immutable;
using Starlings.Generation.OnnxGenAI;

namespace Starlings.Models.QwenChat;

/// <summary>The roles supported by the experimental text-only profiles.</summary>
public enum QwenChatRole
{
    /// <summary>A system instruction.</summary>
    System,
    /// <summary>A user text message.</summary>
    User,
    /// <summary>An assistant text message.</summary>
    Assistant,
}

/// <summary>A text message; tools and multimodal input have no representation in this profile.</summary>
public sealed record QwenChatMessage(QwenChatRole Role, string Content, string? ReasoningContent = null);

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
    /// <summary>Tool invocation is unsupported.</summary>
    public string Tools { get; } = "unsupported";
    /// <summary>Multimodal input is unsupported.</summary>
    public string Multimodal { get; } = "unsupported";
    /// <summary>Each request gets a fresh generator; weights are shared, conversation cache state is never reused.</summary>
    public string Cache { get; } = "fresh-request";
    /// <summary>Qwen3 thinking is explicitly disabled for this profile.</summary>
    public bool EnableThinking { get; }
    /// <summary>Unused embedding row used for padding so literal endoftext prompt positions remain attended.</summary>
    public int PadTokenIdOverlay { get; } = 151935;
}
