using System.Collections.Immutable;

namespace Starlings.Generation;

/// <summary>
/// The identity and shape of a pinned generation checkpoint, independent of a runtime: where it comes from, which owned
/// tokenizer and template render its prompts, which ids end a turn, and the decoder dimensions a backend checks a
/// build against. A model package publishes one per admitted checkpoint.
/// </summary>
public sealed record ModelProfile
{
    /// <summary>Short model id as the registry names it, for example <c>qwen3.5-0.8b</c>.</summary>
    public required string Id { get; init; }

    /// <summary>Hub repository of the checkpoint, <c>owner/name</c>.</summary>
    public required string Repository { get; init; }

    /// <summary>Pinned full commit SHA of <see cref="Repository"/>.</summary>
    public required string Revision { get; init; }

    /// <summary>
    /// The <c>Starlings.Tokenizers</c> profile of the owned tokenizer, or null when the checkpoint's own
    /// <c>tokenizer.json</c> is loaded without a named profile.
    /// </summary>
    public string? TokenizerProfile { get; init; }

    /// <summary>Lowercase hexadecimal SHA-256 of the <c>tokenizer.json</c> the owned tokenizer loads.</summary>
    public required string TokenizerSha256 { get; init; }

    /// <summary>Lowercase hexadecimal SHA-256 of the upstream chat template the C# renderer follows.</summary>
    public required string ChatTemplateSha256 { get; init; }

    /// <summary>End-of-sequence token ids that end an assistant turn, ascending.</summary>
    public required ImmutableArray<int> EosTokenIds { get; init; }

    /// <summary>Rows of the embedding and the logits (can exceed the tokenizer's ids).</summary>
    public required int VocabularySize { get; init; }

    /// <summary>Hidden-state width of the decoder.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Decoder layers.</summary>
    public required int Layers { get; init; }

    /// <summary>Attention head width of the full-attention layers.</summary>
    public required int HeadSize { get; init; }

    /// <summary>The checkpoint's own context length (prompt plus completion tokens); a runtime may accept less.</summary>
    public required int ContextLength { get; init; }

    /// <summary>SPDX licence expression of the weights.</summary>
    public required string License { get; init; }

    /// <summary><c>Repository@Revision</c>.</summary>
    public string Checkpoint => $"{Repository}@{Revision}";

    /// <inheritdoc />
    public bool Equals(ModelProfile? other) =>
        other is not null &&
        Id == other.Id && Repository == other.Repository && Revision == other.Revision &&
        TokenizerProfile == other.TokenizerProfile && TokenizerSha256 == other.TokenizerSha256 &&
        ChatTemplateSha256 == other.ChatTemplateSha256 && EosTokenIds.AsSpan().SequenceEqual(other.EosTokenIds.AsSpan()) &&
        VocabularySize == other.VocabularySize && HiddenSize == other.HiddenSize && Layers == other.Layers &&
        HeadSize == other.HeadSize && ContextLength == other.ContextLength && License == other.License;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Id, Repository, Revision, TokenizerSha256, ChatTemplateSha256);
}
