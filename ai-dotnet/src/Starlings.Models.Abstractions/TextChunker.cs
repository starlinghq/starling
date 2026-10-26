namespace Starlings.Models;

/// <summary>Immutable configuration of a <see cref="TextChunker"/>.</summary>
public sealed record TextChunkerOptions
{
    /// <summary>
    /// Tokens per chunk at most, special tokens and <see cref="Prefix"/> included: the embedder's
    /// <see cref="ITextEmbedder.MaxSequenceLength"/> (or less). Required; at least the special and prefix tokens plus one.
    /// </summary>
    public required int MaxTokens { get; init; }

    /// <summary>
    /// Tokens each chunk repeats from the end of the previous one. Default 0. Must leave room for at least one new token
    /// per chunk (less than the text tokens a chunk holds).
    /// </summary>
    public int OverlapTokens { get; init; }

    /// <summary>
    /// Text the embedder will see before every chunk (an e5 prefix such as <c>"passage: "</c>, or an EmbeddingGemma 2
    /// prompt's <c>Text</c>), counted against <see cref="MaxTokens"/> and prepended to the chunk texts
    /// <see cref="TextChunker.EmbedPooled"/> embeds. Not part of <see cref="TextChunk.Text"/>. Default empty.
    /// </summary>
    public string Prefix { get; init; } = string.Empty;
}

/// <summary>One chunk of a text: <c>text.Substring(Start, Length)</c>.</summary>
/// <param name="Start">UTF-16 position of the chunk's first character in the source text.</param>
/// <param name="Length">UTF-16 code units in the chunk.</param>
/// <param name="Text">The chunk's text, without <see cref="TextChunkerOptions.Prefix"/>.</param>
/// <param name="TokenCount">Tokens of <see cref="TextChunkerOptions.Prefix"/> plus <see cref="Text"/>, special tokens included, as the tokenizer counts them.</param>
public readonly record struct TextChunk(int Start, int Length, string Text, int TokenCount);

/// <summary>
/// Splits long text into overlapping windows of tokens that each fit a model's limit, using the model's own tokenizer,
/// and optionally pools the chunks' embeddings into one vector. Experimental.
/// </summary>
/// <remarks>
/// <para>
/// <b>Chunks.</b> The text is tokenized once (<see cref="IChunkingTokenizer.GetTokenSpans"/>). Each chunk is the text
/// from its first token's start to its last token's end, so text between two tokens at a chunk boundary (usually a
/// space) belongs to neither chunk. Every chunk is tokenized again with <see cref="TextChunkerOptions.Prefix"/> and
/// shrunk until it fits <see cref="TextChunkerOptions.MaxTokens"/>, because a substring can tokenize to more tokens
/// than the same span did inside the whole text. A text that fits whole is one chunk covering all of it.
/// </para>
/// <para>
/// <b>No parity claim for pooled vectors.</b> <see cref="EmbedPooled"/> returns the mean of the chunks' L2-normalized
/// embeddings, L2-normalized again. That is a convention, not the reference model's output for the whole text: the
/// reference truncates or rejects such text, and no fidelity gate covers pooled vectors. Each chunk's own embedding is
/// the model's output for that chunk text, with the profile's usual fidelity.
/// </para>
/// <para>Instances are immutable and safe for concurrent use when the tokenizer is.</para>
/// </remarks>
public sealed class TextChunker
{
    private readonly IChunkingTokenizer _tokenizer;
    private readonly TextChunkerOptions _options;
    private readonly int _textBudget;

    /// <summary>Creates a chunker over <paramref name="tokenizer"/>.</summary>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <param name="options">Limits, overlap and prefix.</param>
    /// <exception cref="ArgumentException">The options leave no room for a text token, or the overlap leaves no room for a new token.</exception>
    public TextChunker(IChunkingTokenizer tokenizer, TextChunkerOptions options)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(options.Prefix, nameof(options));
        ArgumentOutOfRangeException.ThrowIfNegative(options.OverlapTokens, nameof(options));
        int reserved = options.Prefix.Length == 0
            ? tokenizer.SpecialTokenCount
            : Math.Max(tokenizer.SpecialTokenCount, tokenizer.CountTokens(options.Prefix));
        _textBudget = options.MaxTokens - reserved;
        if (_textBudget < 1)
        {
            throw new ArgumentException($"MaxTokens = {options.MaxTokens} leaves no room for text: the special tokens and prefix take {reserved}.", nameof(options));
        }

        if (options.OverlapTokens >= _textBudget)
        {
            throw new ArgumentException($"OverlapTokens = {options.OverlapTokens} must be less than the {_textBudget} text tokens a chunk holds.", nameof(options));
        }

        _tokenizer = tokenizer;
        _options = options;
    }

    /// <summary>Options in effect.</summary>
    public TextChunkerOptions Options => _options;

    /// <summary>
    /// Splits <paramref name="text"/> into chunks of at most <see cref="TextChunkerOptions.MaxTokens"/> tokens, prefix and
    /// special tokens included, in text order. An empty text, or one that fits whole, is one chunk. If the span of a
    /// single token needs more tokens than fit when tokenized alone with the prefix, that one-token chunk is returned
    /// as it is, with its larger <see cref="TextChunk.TokenCount"/>; check it before embedding with a profile that
    /// rejects long input.
    /// </summary>
    /// <param name="text">The text.</param>
    /// <param name="cancellationToken">Checked before each chunk.</param>
    public IReadOnlyList<TextChunk> Chunk(string text, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        string prefix = _options.Prefix;
        int maxTokens = _options.MaxTokens;
        int whole = _tokenizer.CountTokens(prefix + text);
        if (whole <= maxTokens)
        {
            return [new TextChunk(0, text.Length, text, whole)];
        }

        var spans = _tokenizer.GetTokenSpans(text);
        int n = spans.Count;
        if (n == 0)
        {
            return [new TextChunk(0, text.Length, text, whole)];
        }

        var chunks = new List<TextChunk>();
        int first = 0;
        while (first < n)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int end = Math.Min(n, first + _textBudget);
            TextChunk chunk;
            while (true)
            {
                int start = spans[first].Start;
                int length = spans[end - 1].End - start;
                string chunkText = text.Substring(start, length);
                int count = _tokenizer.CountTokens(prefix + chunkText);
                chunk = new TextChunk(start, length, chunkText, count);
                if (count <= maxTokens || end == first + 1)
                {
                    break;
                }

                // A substring tokenized alone can need more tokens than its span did in the whole text; drop the excess.
                end = Math.Max(first + 1, end - Math.Max(1, count - maxTokens));
            }

            chunks.Add(chunk);
            if (end == n)
            {
                break;
            }

            first = Math.Max(first + 1, end - _options.OverlapTokens);
        }

        return chunks;
    }

    /// <summary>
    /// Chunks <paramref name="text"/>, embeds every chunk (with <see cref="TextChunkerOptions.Prefix"/> prepended) in one
    /// <see cref="ITextEmbedder.EmbedBatch"/> call, and returns the mean of the L2-normalized chunk vectors, L2-normalized
    /// again. Experimental; the pooled vector is not the reference model's output and carries no parity claim (see the
    /// class remarks).
    /// </summary>
    /// <param name="embedder">The embedder whose tokenizer this chunker uses.</param>
    /// <param name="text">The text.</param>
    /// <param name="cancellationToken">Cancels chunking and the embedding call.</param>
    public float[] EmbedPooled(ITextEmbedder embedder, string text, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        var chunks = Chunk(text, cancellationToken);
        var texts = new string[chunks.Count];
        for (int i = 0; i < texts.Length; i++)
        {
            texts[i] = _options.Prefix + chunks[i].Text;
        }

        var counts = new int[texts.Length];
        return Pool(embedder.EmbedBatch(texts, counts, cancellationToken));
    }

    /// <summary>
    /// The mean of <paramref name="vectors"/> after L2-normalizing each, L2-normalized again: the pooling
    /// <see cref="EmbedPooled"/> uses. A zero vector stays zero. No parity claim.
    /// </summary>
    /// <param name="vectors">At least one vector; all of the same length.</param>
    public static float[] Pool(IReadOnlyList<float[]> vectors)
    {
        ArgumentNullException.ThrowIfNull(vectors);
        if (vectors.Count == 0)
        {
            throw new ArgumentException("At least one vector is needed.", nameof(vectors));
        }

        int dimension = vectors[0]?.Length ?? throw new ArgumentException("Vector 0 is null.", nameof(vectors));
        var sum = new double[dimension];
        for (int v = 0; v < vectors.Count; v++)
        {
            float[] vector = vectors[v] ?? throw new ArgumentException($"Vector {v} is null.", nameof(vectors));
            if (vector.Length != dimension)
            {
                throw new ArgumentException($"Vector {v} has {vector.Length} values; vector 0 has {dimension}.", nameof(vectors));
            }

            double norm = Norm(vector);
            if (norm == 0)
            {
                continue;
            }

            for (int i = 0; i < dimension; i++)
            {
                sum[i] += vector[i] / norm;
            }
        }

        double total = 0;
        for (int i = 0; i < dimension; i++)
        {
            total += sum[i] * sum[i];
        }

        double scale = total == 0 ? 0 : 1 / Math.Sqrt(total);
        var pooled = new float[dimension];
        for (int i = 0; i < dimension; i++)
        {
            pooled[i] = (float)(sum[i] * scale);
        }

        return pooled;
    }

    private static double Norm(float[] vector)
    {
        double total = 0;
        foreach (float value in vector)
        {
            total += (double)value * value;
        }

        return Math.Sqrt(total);
    }
}
