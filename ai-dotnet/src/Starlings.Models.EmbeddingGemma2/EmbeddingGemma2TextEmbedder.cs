using Starlings.Runtime.Onnx;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// Text in, embeddings out: an explicit prompt, an <see cref="IEmbeddingGemma2Tokenizer"/> and an
/// <see cref="EmbeddingGemma2Encoder"/>. The text of a prompt other than <see cref="EmbeddingGemma2Prompt.None"/> is
/// prepended to every text before tokenization, as sentence-transformers does with <c>prompt_name</c>.
/// </summary>
/// <remarks>
/// <para>
/// No prompt is ever inserted implicitly: every method takes the prompt as an argument, and the overloads without one
/// use <see cref="EmbeddingGemma2Prompt.None"/>. With <see cref="EmbeddingGemma2Tokenizer"/> this is experimental on
/// Linux x64: fixture texts under each documented prompt reproduce the protected ids exactly and the protected
/// embeddings within the binding tolerance.
/// </para>
/// <para>
/// <b>Lifetime.</b> An instance from <see cref="Create"/> owns its encoder and session and disposes them; an instance
/// from the constructor owns nothing and its <see cref="Dispose"/> does nothing. Both are idempotent.
/// </para>
/// </remarks>
public sealed class EmbeddingGemma2TextEmbedder : ITextEmbedder, IDisposable
{
    private readonly IEmbeddingGemma2Tokenizer _tokenizer;
    private readonly EmbeddingGemma2Encoder _encoder;
    private readonly bool _ownsEncoder;
    private int _disposed;

    /// <summary>Combines a tokenizer and an encoder. Neither is owned or disposed by this instance.</summary>
    public EmbeddingGemma2TextEmbedder(IEmbeddingGemma2Tokenizer tokenizer, EmbeddingGemma2Encoder encoder)
        : this(tokenizer, encoder, ownsEncoder: false)
    {
    }

    private EmbeddingGemma2TextEmbedder(IEmbeddingGemma2Tokenizer tokenizer, EmbeddingGemma2Encoder encoder, bool ownsEncoder)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(encoder);
        _tokenizer = tokenizer;
        _encoder = encoder;
        _ownsEncoder = ownsEncoder;
    }

    /// <summary>Floats per embedding (<see cref="EmbeddingGemma2EncoderOptions.Dimension"/>: 768, 512, 256 or 128).</summary>
    public int Dimension => _encoder.Dimension;

    /// <summary>
    /// Tokens per text at most, prompt included (<see cref="EmbeddingGemma2EncoderOptions.MaxSequenceLength"/>, 8,192
    /// by default). Longer text throws <see cref="EmbeddingGemma2InputTooLongException"/>.
    /// </summary>
    public int MaxSequenceLength => _encoder.Options.MaxSequenceLength;

    /// <summary>
    /// Loads the pinned graph and the owned tokenizer (<see cref="EmbeddingGemma2Tokenizer"/>) from verified local files
    /// (<see cref="EmbeddingGemma2ModelAssets.OpenLocal"/> or <see cref="EmbeddingGemma2ModelAssets.OpenDirectory"/>;
    /// never downloads) and owns the resulting encoder and session. The tokenizer's SHA-256
    /// (<see cref="EmbeddingGemma2ModelAssets.TokenizerSha256"/>) is checked while loading, and the graph's
    /// (<see cref="EmbeddingGemma2ModelAssets.ModelSha256"/>) is set as <see cref="OnnxSessionOptions.ExpectedModelSha256"/>
    /// on <paramref name="options"/>.<see cref="EmbeddingGemma2EncoderOptions.Session"/>. The other options, the
    /// dimension included, are taken as given.
    /// </summary>
    /// <param name="files">Verified files.</param>
    /// <param name="options">Encoder options; defaults to the profile.</param>
    public static EmbeddingGemma2TextEmbedder Create(EmbeddingGemma2ModelFiles files, EmbeddingGemma2EncoderOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new EmbeddingGemma2EncoderOptions();
        options.Validate();
        var tokenizer = EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath, EmbeddingGemma2ModelAssets.TokenizerSha256);
        var encoderOptions = options with
        {
            Session = options.Session with
            {
                ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256,
                KnownModelSha256 = options.Session.KnownModelSha256 ?? files.ModelSha256,
            },
        };
        var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, encoderOptions);
        try
        {
            return new EmbeddingGemma2TextEmbedder(tokenizer, encoder, ownsEncoder: true);
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

    /// <summary>The encoder (dimension, options, session diagnostics). Owned by this instance only when it came from <see cref="Create"/>.</summary>
    public EmbeddingGemma2Encoder Encoder => _encoder;

    /// <summary>The tokenizer (<see cref="EmbeddingGemma2Tokenizer"/> when this instance came from <see cref="Create"/>).</summary>
    public IEmbeddingGemma2Tokenizer Tokenizer => _tokenizer;

    /// <summary>Embeds one text with no prompt.</summary>
    public float[] Embed(string text, CancellationToken cancellationToken = default) =>
        Embed(text, EmbeddingGemma2Prompt.None, cancellationToken);

    /// <summary>Embeds one text with <paramref name="prompt"/> prepended.</summary>
    public float[] Embed(string text, EmbeddingGemma2Prompt prompt, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(prompt);
        return _encoder.Embed(Tokenize(text, prompt), cancellationToken);
    }

    /// <summary>
    /// The tokenizer as an <see cref="IChunkingTokenizer"/> for <see cref="TextChunker"/> (Experimental), or null when it
    /// is not one (an <see cref="IEmbeddingGemma2Tokenizer"/> other than <see cref="EmbeddingGemma2Tokenizer"/>). Pass
    /// the prompt's <see cref="EmbeddingGemma2Prompt.Text"/> as <see cref="TextChunkerOptions.Prefix"/>.
    /// </summary>
    public IChunkingTokenizer? ChunkingTokenizer => _tokenizer as IChunkingTokenizer;

    /// <summary>
    /// Tokenizes <paramref name="text"/> with <paramref name="prompt"/> prepended and checks it against
    /// <see cref="MaxSequenceLength"/>, as every embedding method does before running the encoder. With
    /// <see cref="EmbeddingGemma2Tokenizer"/>, text whose length alone proves it too long is rejected without being
    /// tokenized (<see cref="InputTooLongException.TokenCountIsLowerBound"/>).
    /// </summary>
    /// <param name="text">The text.</param>
    /// <param name="prompt">The prompt to prepend.</param>
    /// <param name="inputIndex">Position reported in the exception.</param>
    /// <exception cref="EmbeddingGemma2InputTooLongException">The input has more than <see cref="MaxSequenceLength"/> tokens.</exception>
    public EmbeddingGemma2Input Tokenize(string text, EmbeddingGemma2Prompt prompt, int inputIndex = 0)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(prompt);
        string prompted = prompt.Apply(text);
        int maxTokens = MaxSequenceLength;
        if (_tokenizer is EmbeddingGemma2Tokenizer owned)
        {
            return owned.Tokenize(prompted, maxTokens, inputIndex);
        }

        var input = _tokenizer.Tokenize(prompted);
        if (input.Length > maxTokens)
        {
            throw new EmbeddingGemma2InputTooLongException(inputIndex, input.Length, maxTokens);
        }

        return input;
    }

    /// <summary>
    /// Tokens <paramref name="text"/> needs with no prompt, <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c> included, without
    /// embedding it. Tokenizes the whole text. Experimental (capability <c>token_counting</c>).
    /// </summary>
    public int CountTokens(string text) => CountTokens(text, EmbeddingGemma2Prompt.None);

    /// <summary>
    /// Tokens <paramref name="text"/> needs with <paramref name="prompt"/> prepended, special tokens and prompt included,
    /// without embedding it. Tokenizes the whole text. Experimental (capability <c>token_counting</c>).
    /// </summary>
    public int CountTokens(string text, EmbeddingGemma2Prompt prompt)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(prompt);
        string prompted = prompt.Apply(text);
        return _tokenizer is IChunkingTokenizer counter ? counter.CountTokens(prompted) : _tokenizer.Tokenize(prompted).Length;
    }

    /// <summary>The token count of each text with no prompt, in order, as <see cref="CountTokens(string)"/> returns it.</summary>
    public int[] CountTokens(IReadOnlyList<string> texts) => CountTokens(texts, EmbeddingGemma2Prompt.None);

    /// <summary>The token count of each text with <paramref name="prompt"/> prepended, in order.</summary>
    public int[] CountTokens(IReadOnlyList<string> texts, EmbeddingGemma2Prompt prompt)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var counts = new int[texts.Count];
        for (int i = 0; i < counts.Length; i++)
        {
            counts[i] = CountTokens(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts)), prompt);
        }

        return counts;
    }

    /// <summary>
    /// Checks <paramref name="text"/> with no prompt against <see cref="MaxSequenceLength"/> without embedding it and
    /// without throwing. With <see cref="EmbeddingGemma2Tokenizer"/>, text whose length alone proves it too long is not
    /// tokenized (<see cref="TokenLimitCheck.TokenCountIsLowerBound"/>). An input with
    /// <see cref="TokenLimitCheck.ExceedsLimit"/> would throw <see cref="EmbeddingGemma2InputTooLongException"/>.
    /// Experimental (capability <c>token_counting</c>).
    /// </summary>
    public TokenLimitCheck Validate(string text) => Validate(text, EmbeddingGemma2Prompt.None);

    /// <summary>As <see cref="Validate(string)"/>, with <paramref name="prompt"/> prepended.</summary>
    public TokenLimitCheck Validate(string text, EmbeddingGemma2Prompt prompt)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(prompt);
        return _tokenizer is EmbeddingGemma2Tokenizer owned
            ? owned.Validate(prompt.Apply(text), MaxSequenceLength)
            : new TokenLimitCheck(CountTokens(text, prompt), MaxSequenceLength);
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
    /// Embeds texts in order with no prompt (<see cref="EmbeddingGemma2Prompt.None"/>) and writes each text's token
    /// count (special tokens included) to <paramref name="tokenCounts"/>, which must hold exactly <c>texts.Count</c>
    /// values. The <see cref="ITextEmbedder"/> form; pass a prompt with the overload that takes one.
    /// </summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, Span<int> tokenCounts, CancellationToken cancellationToken = default) =>
        EmbedBatch(texts, EmbeddingGemma2Prompt.None, tokenCounts, cancellationToken);

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
            inputs[i] = Tokenize(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts)), prompt, i);
            if (!tokenCounts.IsEmpty)
            {
                tokenCounts[i] = inputs[i].Length;
            }
        }

        return _encoder.EmbedBatch(inputs, cancellationToken);
    }
}
