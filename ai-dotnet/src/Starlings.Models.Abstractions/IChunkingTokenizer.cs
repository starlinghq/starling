namespace Starlings.Models;

/// <summary>
/// One token of a text as a span of the text: UTF-16 positions <see cref="Start"/> (inclusive) to <see cref="End"/>
/// (exclusive), so <c>text[Start..End]</c> is the text the token came from.
/// </summary>
/// <param name="Start">First UTF-16 position the token covers.</param>
/// <param name="End">Position after the last one it covers.</param>
public readonly record struct TokenSpan(int Start, int End)
{
    /// <summary>UTF-16 code units covered.</summary>
    public int Length => End - Start;
}

/// <summary>
/// What <see cref="TextChunker"/> needs from a model's tokenizer: untruncated token counts and the span of each token
/// in the text. Implemented by the owned tokenizers of the text embedding packages (<c>E5Tokenizer</c>,
/// <c>EmbeddingGemma2Tokenizer</c>) and by <c>BertTextEmbedder.ChunkingTokenizer</c>.
/// </summary>
/// <remarks>Implementations must be safe for concurrent calls.</remarks>
public interface IChunkingTokenizer
{
    /// <summary>Tokens the tokenizer adds around every sequence (for example <c>&lt;s&gt;</c> and <c>&lt;/s&gt;</c>: 2).</summary>
    int SpecialTokenCount { get; }

    /// <summary>
    /// Tokens of <paramref name="text"/> as the model would see it, special tokens included, never truncated: a text
    /// above a truncating profile's limit reports its full count.
    /// </summary>
    /// <param name="text">The text, exactly as it would be embedded (prefix or prompt included).</param>
    int CountTokens(string text);

    /// <summary>
    /// The span of every token that comes from <paramref name="text"/>, in order, without special tokens and without
    /// truncation. Several tokens may share a span (byte-fallback tokens of one character).
    /// </summary>
    /// <param name="text">The text.</param>
    IReadOnlyList<TokenSpan> GetTokenSpans(string text);
}
