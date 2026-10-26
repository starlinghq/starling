using Starlings.Runtime.Onnx;

namespace Starlings.Models.Bert;

/// <summary>A reusable owned-tokenizer plus ONNX text embedding session for an explicit BERT profile.</summary>
public sealed class BertTextEmbedder : ITextEmbedder, IDisposable
{
    private readonly BertSession _model;
    private BertTextEmbedder(BertSession model) => _model = model;

    /// <summary>Profile and immutable source/runtime asset identities.</summary>
    public BertModelProfile Profile => _model.Profile;
    /// <summary>Embedding dimension.</summary>
    public int Dimension => Profile.HiddenSize;
    /// <summary>Tokens per text at most (<see cref="BertModelProfile.MaxTokens"/>); longer text is truncated by the tokenizer.</summary>
    public int MaxSequenceLength => Profile.MaxTokens;
    /// <summary>Loaded model and native runtime diagnostics.</summary>
    public OnnxSessionDiagnostics Diagnostics => _model.Session.Diagnostics;
    /// <summary>Advanced backend API for intermediate diagnostics. Owned by this embedder.</summary>
    public OnnxModelSession Session => _model.Session;

    /// <summary>Verifies local assets and creates a CPU session; never downloads.</summary>
    /// <param name="profile">The candidate profile.</param>
    /// <param name="snapshot">The local snapshot directory holding the locked files.</param>
    /// <param name="threads">Intra-op threads (1 to 64).</param>
    /// <param name="maxBatchSize">Rows per ONNX Runtime call at most (1 to 128). Default 32.</param>
    /// <param name="maxBatchTokens">
    /// Padded cells (rows x longest row) per call at most, or null (default) for no token budget. A row longer than the
    /// budget runs alone. Experimental (<see cref="BertCapabilities.BatchTokenBudget"/>).
    /// </param>
    /// <param name="sortByLength">
    /// Chunk rows in ascending token length order (ties in input order) and return results in input order. Outputs
    /// then depend on which rows share a call (within padded-batch tolerance, not bit for bit). Default false: chunks
    /// are consecutive rows in input order. Experimental (<see cref="BertCapabilities.BatchSortByLength"/>).
    /// </param>
    public static BertTextEmbedder Create(BertModelProfile profile, string snapshot, int threads = 2, int maxBatchSize = 32, int? maxBatchTokens = null, bool sortByLength = false)
    {
        ArgumentNullException.ThrowIfNull(profile);
        if (profile.Task != BertTask.Embeddings)
        {
            throw new ArgumentException("Use BertReranker for paired-input scoring profiles.", nameof(profile));
        }

        return new BertTextEmbedder(new BertSession(profile, snapshot, threads, maxBatchSize, maxBatchTokens, sortByLength));
    }

    /// <summary>Tokenizes an explicitly named prompt followed by text as one sequence.</summary>
    public BertInput Tokenize(string text, string prompt = "none")
    {
        ArgumentNullException.ThrowIfNull(text);
        if (!Profile.Prompts.TryGetValue(prompt, out string? prefix))
        {
            throw new ArgumentException($"Prompt '{prompt}' is not declared by {Profile.Id}.", nameof(prompt));
        }

        return _model.Tokenize(prefix + text);
    }

    /// <summary>Embeds one text with an explicit optional prompt; passages use none.</summary>
    public float[] Embed(string text, string prompt = "none", CancellationToken cancellationToken = default) =>
        EmbedTokens([Tokenize(text, prompt)], cancellationToken)[0];

    /// <summary>Embeds a batch in input order, right-padding bounded chunks.</summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, string prompt = "none", CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        cancellationToken.ThrowIfCancellationRequested();
        return EmbedTokens(texts.Select(text => Tokenize(text, prompt)).ToArray(), cancellationToken);
    }

    /// <summary>
    /// Embeds texts in order with the prompt <c>none</c> and writes each text's token count (special tokens included,
    /// after truncation) to <paramref name="tokenCounts"/>, which must hold exactly <c>texts.Count</c> values. The
    /// <see cref="ITextEmbedder"/> form.
    /// </summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, Span<int> tokenCounts, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        if (tokenCounts.Length != texts.Count)
        {
            throw new ArgumentException($"tokenCounts holds {tokenCounts.Length} values; {texts.Count} texts need {texts.Count}.", nameof(tokenCounts));
        }

        cancellationToken.ThrowIfCancellationRequested();
        var inputs = new BertInput[texts.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = Tokenize(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts)));
            tokenCounts[i] = inputs[i].Length;
        }

        return EmbedTokens(inputs, cancellationToken);
    }

    /// <summary>
    /// The profile's tokenizer without truncation, as an <see cref="IChunkingTokenizer"/> for <see cref="TextChunker"/>
    /// (Experimental). Pass a prompt's text as <see cref="TextChunkerOptions.Prefix"/>.
    /// </summary>
    public IChunkingTokenizer ChunkingTokenizer => _model;

    /// <summary>
    /// Tokens <paramref name="text"/> needs with the prompt <c>none</c>, special tokens included, before truncation (so
    /// above <see cref="MaxSequenceLength"/> for text that is embedded truncated). Tokenizes the whole text.
    /// </summary>
    public int CountTokens(string text) => _model.CountTokens(text);

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
    /// <see cref="CountTokens(string)"/> against <see cref="MaxSequenceLength"/>. The profiles truncate: an input with
    /// <see cref="TokenLimitCheck.ExceedsLimit"/> is embedded from its first <see cref="MaxSequenceLength"/> tokens.
    /// </summary>
    public TokenLimitCheck Validate(string text) => new(CountTokens(text), MaxSequenceLength);

    /// <summary>Embeds supplied tokenized inputs, retaining mask and segment ids.</summary>
    public float[][] EmbedTokens(IReadOnlyList<BertInput> inputs, CancellationToken cancellationToken = default) =>
        _model.Predict(inputs, cancellationToken);

    /// <summary>Disposes the owned native session. Idempotent.</summary>
    public void Dispose() => _model.Dispose();
}
