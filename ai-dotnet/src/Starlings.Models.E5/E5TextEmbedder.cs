namespace Starlings.Models.E5;

/// <summary>
/// Text in, embeddings out: an <see cref="IEncoderTokenizer"/> followed by an <see cref="E5Encoder"/>. With
/// <see cref="E5Tokenizer"/> this is experimental on Linux x64 CPU: fixture texts reproduce the protected ids exactly
/// and the protected embeddings within the binding tolerance.
/// </summary>
/// <remarks>
/// The text is tokenized exactly as given. e5 was trained with <c>"query: "</c> and <c>"passage: "</c> prefixes; the
/// caller adds the right one (for symmetric tasks such as clustering, the model card uses <c>"query: "</c> on both
/// sides). Neither this class nor the encoder inserts or detects prefixes.
/// </remarks>
public sealed class E5TextEmbedder
{
    private readonly IEncoderTokenizer _tokenizer;
    private readonly E5Encoder _encoder;

    /// <summary>Combines a tokenizer and an encoder. Neither is owned or disposed by this instance.</summary>
    public E5TextEmbedder(IEncoderTokenizer tokenizer, E5Encoder encoder)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(encoder);
        _tokenizer = tokenizer;
        _encoder = encoder;
    }

    /// <summary>The prefix e5 expects on search queries.</summary>
    public const string QueryPrefix = "query: ";

    /// <summary>The prefix e5 expects on documents being searched.</summary>
    public const string PassagePrefix = "passage: ";

    /// <summary>Embeds one text, tokenized as given.</summary>
    public float[] Embed(string text, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        return _encoder.Embed(_tokenizer.Tokenize(text), cancellationToken);
    }

    /// <summary>The encoder (dimension, options, session diagnostics). Not owned by this instance.</summary>
    public E5Encoder Encoder => _encoder;

    /// <summary>Embeds texts in order, tokenized as given.</summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        return EmbedBatchCore(texts, default, cancellationToken);
    }

    /// <summary>
    /// Embeds texts in order, tokenized as given, and writes each text's token count (special tokens included, after
    /// truncation) to <paramref name="tokenCounts"/>, which must hold exactly <c>texts.Count</c> values. The embeddings
    /// are those of <see cref="EmbedBatch(IReadOnlyList{string}, CancellationToken)"/>.
    /// </summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, Span<int> tokenCounts, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        if (tokenCounts.Length != texts.Count)
        {
            throw new ArgumentException($"tokenCounts holds {tokenCounts.Length} values; {texts.Count} texts need {texts.Count}.", nameof(tokenCounts));
        }

        return EmbedBatchCore(texts, tokenCounts, cancellationToken);
    }

    private float[][] EmbedBatchCore(IReadOnlyList<string> texts, Span<int> tokenCounts, CancellationToken cancellationToken)
    {
        var inputs = new EncoderInput[texts.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            inputs[i] = _tokenizer.Tokenize(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts)));
            if (!tokenCounts.IsEmpty)
            {
                tokenCounts[i] = inputs[i].Length;
            }
        }

        return _encoder.EmbedBatch(inputs, cancellationToken);
    }
}
