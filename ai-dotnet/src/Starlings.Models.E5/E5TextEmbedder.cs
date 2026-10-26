using Starlings.Runtime.Onnx;

namespace Starlings.Models.E5;

/// <summary>
/// Text in, embeddings out: an <see cref="IEncoderTokenizer"/> followed by an <see cref="E5Encoder"/>. With
/// <see cref="E5Tokenizer"/> this is experimental on Linux x64 CPU: fixture texts reproduce the protected ids exactly
/// and the protected embeddings within the binding tolerance.
/// </summary>
/// <remarks>
/// <para>
/// The text is tokenized exactly as given. e5 was trained with <c>"query: "</c> and <c>"passage: "</c> prefixes; the
/// caller adds the right one (for symmetric tasks such as clustering, the model card uses <c>"query: "</c> on both
/// sides). Neither this class nor the encoder inserts or detects prefixes.
/// </para>
/// <para>
/// <b>Length.</b> With <see cref="E5Tokenizer"/>, text longer than <see cref="E5EncoderOptions.MaxSequenceLength"/>
/// tokens (512 by default) is truncated on the right, as the reference does, and embedded; the encoder itself rejects
/// an id sequence longer than its limit, so another <see cref="IEncoderTokenizer"/> must truncate to the same length.
/// </para>
/// <para>
/// <b>Lifetime.</b> An instance from <see cref="Create"/> owns its encoder and session and disposes them; an instance
/// from the constructor owns nothing and its <see cref="Dispose"/> does nothing. Both are idempotent.
/// </para>
/// </remarks>
public sealed class E5TextEmbedder : ITextEmbedder, IDisposable
{
    private readonly IEncoderTokenizer _tokenizer;
    private readonly E5Encoder _encoder;
    private readonly bool _ownsEncoder;
    private int _disposed;

    /// <summary>Combines a tokenizer and an encoder. Neither is owned or disposed by this instance.</summary>
    public E5TextEmbedder(IEncoderTokenizer tokenizer, E5Encoder encoder)
        : this(tokenizer, encoder, ownsEncoder: false)
    {
    }

    private E5TextEmbedder(IEncoderTokenizer tokenizer, E5Encoder encoder, bool ownsEncoder)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(encoder);
        _tokenizer = tokenizer;
        _encoder = encoder;
        _ownsEncoder = ownsEncoder;
    }

    /// <summary>
    /// Loads the pinned tokenizer and model from verified local files (<see cref="E5ModelAssets.OpenLocal"/> or
    /// <see cref="E5ModelAssets.OpenDirectory"/>; never downloads) and owns the resulting encoder and session. The
    /// tokenizer's SHA-256 (<see cref="E5ModelAssets.TokenizerSha256"/>) is checked while loading, and the model's
    /// (<see cref="E5ModelAssets.ModelSha256"/>) is set as <see cref="OnnxSessionOptions.ExpectedModelSha256"/> on
    /// <paramref name="options"/>.<see cref="E5EncoderOptions.Session"/>. The tokenizer truncates to
    /// <see cref="E5EncoderOptions.MaxSequenceLength"/>.
    /// </summary>
    /// <param name="files">Verified files.</param>
    /// <param name="options">Encoder options; defaults to the e5 profile.</param>
    public static E5TextEmbedder Create(E5ModelFiles files, E5EncoderOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new E5EncoderOptions();
        options.Validate();
        var tokenizer = E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256, options.MaxSequenceLength);
        var encoderOptions = options with
        {
            Session = options.Session with
            {
                ExpectedModelSha256 = E5ModelAssets.ModelSha256,
                KnownModelSha256 = options.Session.KnownModelSha256 ?? files.ModelSha256,
            },
        };
        var encoder = E5Encoder.Create(files.ModelPath, encoderOptions);
        try
        {
            return new E5TextEmbedder(tokenizer, encoder, ownsEncoder: true);
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    /// <summary>Disposes the encoder and session if this instance owns them (<see cref="Create"/>). Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsEncoder)
        {
            _encoder.Dispose();
        }
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

    /// <summary>Floats per embedding (<see cref="E5Encoder.Dimension"/>, 384).</summary>
    public int Dimension => _encoder.Dimension;

    /// <summary>
    /// Tokens per text at most (<see cref="E5EncoderOptions.MaxSequenceLength"/>, 512 by default). With
    /// <see cref="E5Tokenizer"/>, longer text is truncated on the right to this length.
    /// </summary>
    public int MaxSequenceLength => _encoder.Options.MaxSequenceLength;

    /// <summary>The encoder (dimension, options, session diagnostics). Owned by this instance only when it came from <see cref="Create"/>.</summary>
    public E5Encoder Encoder => _encoder;

    /// <summary>The tokenizer (<see cref="E5Tokenizer"/> when this instance came from <see cref="Create"/>).</summary>
    public IEncoderTokenizer Tokenizer => _tokenizer;

    /// <summary>
    /// The tokenizer as an <see cref="IChunkingTokenizer"/> for <see cref="TextChunker"/> (Experimental), or null when it
    /// is not one (an <see cref="IEncoderTokenizer"/> other than <see cref="E5Tokenizer"/>).
    /// </summary>
    public IChunkingTokenizer? ChunkingTokenizer => _tokenizer as IChunkingTokenizer;

    /// <summary>
    /// Tokens <paramref name="text"/> needs, special tokens included, without embedding it. With
    /// <see cref="E5Tokenizer"/> the count is before truncation, so text that is embedded truncated reports more than
    /// <see cref="MaxSequenceLength"/>; with another <see cref="IEncoderTokenizer"/> that does not implement
    /// <see cref="IChunkingTokenizer"/>, it is the length of <see cref="IEncoderTokenizer.Tokenize"/>'s output (after
    /// that tokenizer's own truncation). Experimental (capability <c>token_counting</c>).
    /// </summary>
    public int CountTokens(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return _tokenizer is IChunkingTokenizer counter ? counter.CountTokens(text) : _tokenizer.Tokenize(text).Length;
    }

    /// <summary>The token count of each text, in order, as <see cref="CountTokens(string)"/> returns it.</summary>
    public int[] CountTokens(IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var counts = new int[texts.Count];
        for (int i = 0; i < counts.Length; i++)
        {
            counts[i] = CountTokens(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts)));
        }

        return counts;
    }

    /// <summary>
    /// <see cref="CountTokens(string)"/> against <see cref="MaxSequenceLength"/>. e5 truncates rather than rejects: an
    /// input with <see cref="TokenLimitCheck.ExceedsLimit"/> is embedded from its first <see cref="MaxSequenceLength"/>
    /// tokens. Experimental (capability <c>token_counting</c>).
    /// </summary>
    public TokenLimitCheck Validate(string text) => new(CountTokens(text), MaxSequenceLength);

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
