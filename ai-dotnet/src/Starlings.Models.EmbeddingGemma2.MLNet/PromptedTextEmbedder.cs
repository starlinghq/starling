namespace Starlings.Models.EmbeddingGemma2.MLNet;

/// <summary>
/// An <see cref="EmbeddingGemma2TextEmbedder"/> with a fixed prompt, as the <see cref="ITextEmbedder"/> the shared
/// transform code embeds through: <see cref="EmbeddingGemma2TextEmbedder.EmbedBatch(IReadOnlyList{string}, EmbeddingGemma2Prompt, Span{int}, CancellationToken)"/>.
/// </summary>
internal sealed class PromptedTextEmbedder(EmbeddingGemma2TextEmbedder inner, EmbeddingGemma2Prompt prompt) : ITextEmbedder
{
    public int Dimension => inner.Dimension;

    public int MaxSequenceLength => inner.MaxSequenceLength;

    public float[][] EmbedBatch(IReadOnlyList<string> texts, Span<int> tokenCounts, CancellationToken cancellationToken = default) =>
        inner.EmbedBatch(texts, prompt, tokenCounts, cancellationToken);

    public int CountTokens(string text) => inner.CountTokens(text, prompt);

    public int[] CountTokens(IReadOnlyList<string> texts) => inner.CountTokens(texts, prompt);

    public TokenLimitCheck Validate(string text) => inner.Validate(text, prompt);
}
