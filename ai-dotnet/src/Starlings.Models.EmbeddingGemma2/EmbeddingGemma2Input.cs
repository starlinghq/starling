namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// One already-tokenized sequence: the ids the pinned <c>tokenizer.json</c> produces for <c>prompt + text</c>,
/// including <c>&lt;bos&gt;</c> (2) at the start and <c>&lt;eos&gt;</c> (1) at the end, untruncated. This is the seam
/// between tokenization and the encoder: a tokenizer produces <see cref="EmbeddingGemma2Input"/>, the encoder never
/// tokenizes.
/// </summary>
/// <remarks>
/// A prompt (<see cref="EmbeddingGemma2Prompt"/>) is part of the ids: tokenize <c>prompt.Text + text</c> as one string.
/// Every attended position, prompt tokens included, takes part in mean pooling (<c>1_Pooling/config.json</c>
/// <c>include_prompt: true</c>). The encoder only reads the memory during the call that receives it.
/// </remarks>
public readonly struct EmbeddingGemma2Input : IEquatable<EmbeddingGemma2Input>
{
    /// <summary>A sequence whose tokens all take part in attention and pooling (attention mask all ones).</summary>
    public EmbeddingGemma2Input(ReadOnlyMemory<long> inputIds)
    {
        InputIds = inputIds;
        AttentionMask = default;
    }

    /// <summary>
    /// A sequence with an explicit attention mask of the same length, values 0 or 1. Positions with mask 0 are excluded
    /// from attention and from mean pooling.
    /// </summary>
    public EmbeddingGemma2Input(ReadOnlyMemory<long> inputIds, ReadOnlyMemory<long> attentionMask)
    {
        InputIds = inputIds;
        AttentionMask = attentionMask;
    }

    /// <summary>Token ids, including special tokens and prompt tokens.</summary>
    public ReadOnlyMemory<long> InputIds { get; }

    /// <summary>Attention mask, or empty to mean all ones.</summary>
    public ReadOnlyMemory<long> AttentionMask { get; }

    /// <summary>Number of token positions.</summary>
    public int Length => InputIds.Length;

    /// <summary>Equality by memory identity, not by content.</summary>
    public static bool operator ==(EmbeddingGemma2Input left, EmbeddingGemma2Input right) => left.Equals(right);

    /// <summary>Inequality by memory identity, not by content.</summary>
    public static bool operator !=(EmbeddingGemma2Input left, EmbeddingGemma2Input right) => !left.Equals(right);

    /// <inheritdoc/>
    public bool Equals(EmbeddingGemma2Input other) => InputIds.Equals(other.InputIds) && AttentionMask.Equals(other.AttentionMask);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is EmbeddingGemma2Input other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(InputIds, AttentionMask);
}

/// <summary>
/// Turns text into encoder input. The owned Gemma BPE profile of <c>Starlings.Tokenizers</c> plugs in here
/// through <see cref="EmbeddingGemma2Tokenizer"/>.
/// </summary>
/// <remarks>
/// An implementation must reproduce the pinned <c>tokenizer.json</c> (<see cref="EmbeddingGemma2ModelAssets.TokenizerSha256"/>)
/// exactly: ids and mask (G1), <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c> added, no truncation (the encoder rejects
/// inputs above 8,192 tokens instead), no padding, one sequence. It must not add a prompt; that is the caller's
/// decision (<see cref="EmbeddingGemma2TextEmbedder"/> prepends <see cref="EmbeddingGemma2Prompt.Text"/> to the text
/// before calling <see cref="Tokenize"/>). Implementations must be safe for concurrent calls if the embedder is shared
/// across threads.
/// </remarks>
public interface IEmbeddingGemma2Tokenizer
{
    /// <summary>Tokenizes one text exactly as given. The returned memory must stay valid until the caller is done with it.</summary>
    EmbeddingGemma2Input Tokenize(string text);
}
