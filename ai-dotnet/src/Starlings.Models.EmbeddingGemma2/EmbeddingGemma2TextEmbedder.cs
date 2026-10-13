namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// Text in, embeddings out: an explicit prompt, an <see cref="IEmbeddingGemma2Tokenizer"/> and an
/// <see cref="EmbeddingGemma2Encoder"/>. The text of a prompt other than <see cref="EmbeddingGemma2Prompt.None"/> is
/// prepended to every text before tokenization, as sentence-transformers does with <c>prompt_name</c>.
/// </summary>
/// <remarks>
/// No prompt is ever inserted implicitly: every method takes the prompt as an argument, and the overloads without one
/// use <see cref="EmbeddingGemma2Prompt.None"/>. With <see cref="EmbeddingGemma2Tokenizer"/> this is experimental on
/// Linux x64: fixture texts under each documented prompt reproduce the protected ids exactly and the protected
/// embeddings within the binding tolerance.
/// </remarks>
public sealed class EmbeddingGemma2TextEmbedder
{
    private readonly IEmbeddingGemma2Tokenizer _tokenizer;
    private readonly EmbeddingGemma2Encoder _encoder;

    /// <summary>Combines a tokenizer and an encoder. Neither is owned or disposed by this instance.</summary>
    public EmbeddingGemma2TextEmbedder(IEmbeddingGemma2Tokenizer tokenizer, EmbeddingGemma2Encoder encoder)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(encoder);
        _tokenizer = tokenizer;
        _encoder = encoder;
    }

    /// <summary>The encoder (dimension, options, session diagnostics). Not owned by this instance.</summary>
    public EmbeddingGemma2Encoder Encoder => _encoder;

    /// <summary>The tokenizer. Not owned by this instance.</summary>
    public IEmbeddingGemma2Tokenizer Tokenizer => _tokenizer;

    /// <summary>Embeds one text with no prompt.</summary>
    public float[] Embed(string text, CancellationToken cancellationToken = default) =>
        Embed(text, EmbeddingGemma2Prompt.None, cancellationToken);

    /// <summary>Embeds one text with <paramref name="prompt"/> prepended.</summary>
    public float[] Embed(string text, EmbeddingGemma2Prompt prompt, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(prompt);
        return _encoder.Embed(_tokenizer.Tokenize(prompt.Apply(text)), cancellationToken);
    }

    /// <summary>Embeds texts in order with no prompt.</summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) =>
        EmbedBatch(texts, EmbeddingGemma2Prompt.None, cancellationToken);

    /// <summary>Embeds texts in order, each with <paramref name="prompt"/> prepended.</summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, EmbeddingGemma2Prompt prompt, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentNullException.ThrowIfNull(prompt);
        return EmbedBatchCore(texts, prompt, default, cancellationToken);
    }

    /// <summary>
    /// Embeds texts in order, each with <paramref name="prompt"/> prepended, and writes each text's token count (special
    /// tokens and prompt included) to <paramref name="tokenCounts"/>, which must hold exactly <c>texts.Count</c> values.
    /// </summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, EmbeddingGemma2Prompt prompt, Span<int> tokenCounts, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentNullException.ThrowIfNull(prompt);
        if (tokenCounts.Length != texts.Count)
        {
            throw new ArgumentException($"tokenCounts holds {tokenCounts.Length} values; {texts.Count} texts need {texts.Count}.", nameof(tokenCounts));
        }

        return EmbedBatchCore(texts, prompt, tokenCounts, cancellationToken);
    }

    private float[][] EmbedBatchCore(IReadOnlyList<string> texts, EmbeddingGemma2Prompt prompt, Span<int> tokenCounts, CancellationToken cancellationToken)
    {
        var inputs = new EmbeddingGemma2Input[texts.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            inputs[i] = _tokenizer.Tokenize(prompt.Apply(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts))));
            if (!tokenCounts.IsEmpty)
            {
                tokenCounts[i] = inputs[i].Length;
            }
        }

        return _encoder.EmbedBatch(inputs, cancellationToken);
    }
}
