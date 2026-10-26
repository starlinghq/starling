namespace Starlings.Models;

/// <summary>
/// Text in, one embedding per text out, in input order: the shape every Starlings text embedder offers, so code that
/// only needs vectors can take any of them.
/// </summary>
/// <remarks>
/// <para>
/// Implementations embed text exactly as given. A model that expects a prefix or a prompt (e5's <c>"query: "</c>,
/// EmbeddingGemma 2's named prompts) is used through this interface with the caller's text unchanged; the model's own
/// API states how to add one.
/// </para>
/// <para>
/// What happens to text longer than <see cref="MaxSequenceLength"/> tokens is the profile's truncation policy
/// (<see cref="ModelProfileCapabilities.Truncation"/>): some profiles truncate, others throw
/// <see cref="InputTooLongException"/>.
/// </para>
/// </remarks>
public interface ITextEmbedder
{
    /// <summary>Floats per embedding.</summary>
    int Dimension { get; }

    /// <summary>Tokens per text at most, special tokens (and any prompt) included.</summary>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Embeds <paramref name="texts"/> in order and writes each text's token count (special tokens and any prompt
    /// included, after truncation where the profile truncates) to <paramref name="tokenCounts"/>, which must hold
    /// exactly <c>texts.Count</c> values.
    /// </summary>
    /// <param name="texts">The texts; none may be null.</param>
    /// <param name="tokenCounts">Receives one token count per text.</param>
    /// <param name="cancellationToken">Cancels the call; nothing is returned for a cancelled call.</param>
    /// <returns>One new vector of <see cref="Dimension"/> floats per text.</returns>
    float[][] EmbedBatch(IReadOnlyList<string> texts, Span<int> tokenCounts, CancellationToken cancellationToken = default);

    /// <summary>
    /// Tokens <paramref name="text"/> needs, special tokens included, without embedding it. The count is taken before any
    /// truncation, so on a truncating profile it can exceed <see cref="MaxSequenceLength"/> (an embedder whose tokenizer
    /// can only return truncated sequences says so on its own member). Tokenizes the whole text.
    /// </summary>
    /// <param name="text">The text, exactly as it would be embedded.</param>
    int CountTokens(string text);

    /// <summary>The token count of each text, in order, as <see cref="CountTokens(string)"/> returns it.</summary>
    /// <param name="texts">The texts; none may be null.</param>
    int[] CountTokens(IReadOnlyList<string> texts);

    /// <summary>
    /// Checks <paramref name="text"/> against <see cref="MaxSequenceLength"/> without embedding it and without throwing
    /// for an input above the limit. An implementation may skip tokenization when the text's length alone proves it is
    /// above the limit (<see cref="TokenLimitCheck.TokenCountIsLowerBound"/>).
    /// </summary>
    /// <param name="text">The text, exactly as it would be embedded.</param>
    TokenLimitCheck Validate(string text);
}
