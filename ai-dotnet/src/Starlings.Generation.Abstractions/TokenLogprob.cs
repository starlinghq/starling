using System.Collections.Immutable;

namespace Starlings.Generation;

/// <summary>
/// The log probability of one generated token (<see cref="GenerationOptions.ReturnLogprobs"/>), with the most likely
/// tokens of the same step. The Microsoft.Extensions.AI adapters put lists of it under their <c>logprobs</c> additional
/// property.
/// </summary>
/// <param name="TokenId">The generated token id.</param>
/// <param name="Text">
/// The token decoded on its own with the owned tokenizer (special tokens kept). A byte-level token that carries part
/// of a multi-byte UTF-8 character decodes to U+FFFD here; the streamed text never splits a character.
/// </param>
/// <param name="Logprob">Natural log of the token's probability: the log-softmax of the step's raw logits at the token.</param>
/// <param name="TopLogprobs">
/// The <see cref="GenerationOptions.TopLogprobs"/> most likely tokens of the step, most likely first (ties by lower id);
/// empty when <see cref="GenerationOptions.TopLogprobs"/> is 0. The generated token appears in it only when it is
/// among them.
/// </param>
/// <remarks>Equality compares <see cref="TopLogprobs"/> element by element.</remarks>
public sealed record TokenLogprob(int TokenId, string Text, double Logprob, ImmutableArray<TopTokenLogprob> TopLogprobs)
{
    /// <inheritdoc />
    public bool Equals(TokenLogprob? other) =>
        other is not null &&
        TokenId == other.TokenId &&
        string.Equals(Text, other.Text, StringComparison.Ordinal) &&
        Logprob.Equals(other.Logprob) &&
        TopLogprobs.AsSpan().SequenceEqual(other.TopLogprobs.AsSpan());

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(TokenId, Text, Logprob, TopLogprobs.IsDefault ? 0 : TopLogprobs.Length);
}

/// <summary>One of the most likely tokens of a step (<see cref="TokenLogprob.TopLogprobs"/>).</summary>
/// <param name="TokenId">The token id.</param>
/// <param name="Text">The token decoded on its own (see <see cref="TokenLogprob.Text"/>).</param>
/// <param name="Logprob">Natural log of its probability at that step.</param>
public sealed record TopTokenLogprob(int TokenId, string Text, double Logprob);
