namespace Starlings.Models;

/// <summary>
/// The result of checking one input against a model's token limit without embedding it (<see cref="ITextEmbedder.Validate"/>).
/// Nothing is thrown for an input above the limit; the caller decides.
/// </summary>
/// <param name="TokenCount">
/// Tokens the input needs, special tokens and any prompt or prefix included, before any truncation. A lower bound
/// when <paramref name="TokenCountIsLowerBound"/> is true.
/// </param>
/// <param name="MaxTokens">The limit in effect.</param>
/// <param name="TokenCountIsLowerBound">
/// True when the input was not tokenized because its length alone proves it is above the limit; then
/// <paramref name="TokenCount"/> is a lower bound and <see cref="ExceedsLimit"/> is true.
/// </param>
public readonly record struct TokenLimitCheck(int TokenCount, int MaxTokens, bool TokenCountIsLowerBound = false)
{
    /// <summary>
    /// True when the input has more tokens than <see cref="MaxTokens"/>: a profile that rejects long input
    /// (<see cref="TruncationPolicy.Rejects"/>) throws <see cref="InputTooLongException"/> for it, and one that truncates
    /// embeds only its first <see cref="MaxTokens"/> tokens.
    /// </summary>
    public bool ExceedsLimit => TokenCount > MaxTokens;
}
