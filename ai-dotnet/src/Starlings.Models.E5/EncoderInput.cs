namespace Starlings.Models.E5;

/// <summary>
/// One already-tokenized sequence: the token ids produced by the model's tokenizer, including the special tokens
/// (<c>&lt;s&gt;</c> = 0 at the start, <c>&lt;/s&gt;</c> = 2 at the end for e5), already truncated to the model
/// maximum (512). This is the seam between tokenization and the encoder: a tokenizer produces
/// <see cref="EncoderInput"/>, the encoder never tokenizes.
/// </summary>
/// <remarks>
/// The encoder only reads the memory during the call that receives it; it does not keep a reference.
/// </remarks>
public readonly struct EncoderInput : IEquatable<EncoderInput>
{
    /// <summary>A sequence whose tokens all take part in pooling (attention mask all ones).</summary>
    public EncoderInput(ReadOnlyMemory<long> inputIds)
    {
        InputIds = inputIds;
        AttentionMask = default;
    }

    /// <summary>
    /// A sequence with an explicit attention mask of the same length, values 0 or 1. Positions with mask 0 (for
    /// example padding a tokenizer already added) are excluded from attention and from mean pooling.
    /// </summary>
    public EncoderInput(ReadOnlyMemory<long> inputIds, ReadOnlyMemory<long> attentionMask)
    {
        InputIds = inputIds;
        AttentionMask = attentionMask;
    }

    /// <summary>Token ids, including special tokens.</summary>
    public ReadOnlyMemory<long> InputIds { get; }

    /// <summary>Attention mask, or empty to mean all ones.</summary>
    public ReadOnlyMemory<long> AttentionMask { get; }

    /// <summary>Number of token positions.</summary>
    public int Length => InputIds.Length;

    /// <summary>Equality by memory identity, not by content.</summary>
    public static bool operator ==(EncoderInput left, EncoderInput right) => left.Equals(right);

    /// <summary>Inequality by memory identity, not by content.</summary>
    public static bool operator !=(EncoderInput left, EncoderInput right) => !left.Equals(right);

    /// <inheritdoc/>
    public bool Equals(EncoderInput other) => InputIds.Equals(other.InputIds) && AttentionMask.Equals(other.AttentionMask);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is EncoderInput other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(InputIds, AttentionMask);
}
