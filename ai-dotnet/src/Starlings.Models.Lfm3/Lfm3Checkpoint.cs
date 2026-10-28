using Starlings.Generation;

namespace Starlings.Models.Lfm3;

/// <summary>
/// The pinned checkpoint <c>LiquidAI/LFM3-1.7B</c>, independent of a runtime: its repository and commit, the special
/// token ids the template and the owned tokenizer use, the layer layout of its step graph, and its
/// <see cref="ModelProfile"/>.
/// </summary>
public static class Lfm3Checkpoint
{
    /// <summary>Hub repository of the checkpoint.</summary>
    public const string Repository = "LiquidAI/LFM3-1.7B";

    /// <summary>Pinned commit of <see cref="Repository"/>.</summary>
    public const string Revision = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd";

    /// <summary>Owned tokenizer profile (Starlings.Tokenizers).</summary>
    public const string TokenizerProfile = "lfm3-bpe-chat";

    /// <summary>SHA-256 of the checkpoint's <c>tokenizer.json</c> (the <c>lfm3-bpe-chat</c> profile asset).</summary>
    public const string TokenizerSha256 = "7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701";

    /// <summary><c>&lt;|pad|&gt;</c>.</summary>
    public const int PadTokenId = 124893;

    /// <summary><c>&lt;|startoftext|&gt;</c>: the template's <c>bos_token</c>.</summary>
    public const int StartOfTextTokenId = 124894;

    /// <summary><c>&lt;|endoftext|&gt;</c>.</summary>
    public const int EndOfTextTokenId = 124895;

    /// <summary><c>&lt;|im_start|&gt;</c>: starts a turn.</summary>
    public const int ImStartTokenId = 124899;

    /// <summary><c>&lt;|im_end|&gt;</c>: ends an assistant turn; the <c>eos_token_id</c> of <c>config.json</c> and <c>generation_config.json</c>.</summary>
    public const int ImEndTokenId = 124900;

    /// <summary>Rows of the embedding and the logits; the tokenizer's ids end at 125,032.</summary>
    public const int VocabularySize = 128000;

    /// <summary>Hidden size.</summary>
    public const int HiddenSize = 2048;

    /// <summary>Decoder layers.</summary>
    public const int Layers = 24;

    /// <summary>Key/value heads of the attention layers.</summary>
    public const int KeyValueHeads = 4;

    /// <summary>Attention head width.</summary>
    public const int HeadSize = 128;

    /// <summary>Width of a convolution layer's cache (<c>conv_L_cache</c>).</summary>
    public const int ConvCacheLength = 7;

    /// <summary>The checkpoint's positions (<c>max_position_embeddings</c>).</summary>
    public const int ContextLength = 262144;

    /// <summary>
    /// <c>layer_types</c> of <c>config.json</c>: 13 gated short convolutions, 5 sliding-window attention (window 256) and
    /// 6 global attention layers. The step graph names a convolution layer's cache <c>past_conv.N</c> and an attention
    /// layer's caches <c>past_key_values.N.key</c> and <c>.value</c>.
    /// </summary>
    public static IReadOnlyList<string> LayerTypes { get; } =
    [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    ];

    /// <summary>
    /// The checkpoint's identity and decoder shape. The end-of-sequence id is <see cref="ImEndTokenId"/>, the only
    /// <c>eos_token_id</c> of the checkpoint's generation config.
    /// </summary>
    public static ModelProfile Profile { get; } = new()
    {
        Id = "lfm3-1.7b",
        Repository = Repository,
        Revision = Revision,
        TokenizerProfile = TokenizerProfile,
        TokenizerSha256 = TokenizerSha256,
        ChatTemplateSha256 = Lfm3ChatTemplate.TemplateSha256,
        EosTokenIds = [ImEndTokenId],
        VocabularySize = VocabularySize,
        HiddenSize = HiddenSize,
        Layers = Layers,
        HeadSize = HeadSize,
        ContextLength = ContextLength,
        License = "LicenseRef-LFM-Open-License-1.0",
    };
}
