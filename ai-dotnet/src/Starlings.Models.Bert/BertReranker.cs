using Starlings.Runtime.Onnx;

namespace Starlings.Models.Bert;

/// <summary>Paired query/document relevance scoring. Scores are raw logits with identity activation.</summary>
public sealed class BertReranker : IReranker, IDisposable
{
    private readonly BertSession _model;
    private BertReranker(BertSession model) => _model = model;
    /// <summary>Profile and immutable source/runtime asset identities.</summary>
    public BertModelProfile Profile => _model.Profile;
    /// <summary>Loaded model and native runtime diagnostics.</summary>
    public OnnxSessionDiagnostics Diagnostics => _model.Session.Diagnostics;
    /// <summary>Advanced backend API for intermediate diagnostics. Owned by this reranker.</summary>
    public OnnxModelSession Session => _model.Session;

    /// <summary>Verifies local assets and creates a CPU scoring session; never downloads.</summary>
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
    public static BertReranker Create(BertModelProfile profile, string snapshot, int threads = 2, int maxBatchSize = 32, int? maxBatchTokens = null, bool sortByLength = false)
    {
        ArgumentNullException.ThrowIfNull(profile);
        if (profile.Task != BertTask.Reranking)
        {
            throw new ArgumentException("Use BertTextEmbedder for vector embedding profiles.", nameof(profile));
        }

        return new BertReranker(new BertSession(profile, snapshot, threads, maxBatchSize, maxBatchTokens, sortByLength));
    }

    /// <summary>Tokenizes a pair with segment ids and longest-first right truncation.</summary>
    public BertInput Tokenize(BertPair pair)
    {
        ArgumentNullException.ThrowIfNull(pair);
        ArgumentNullException.ThrowIfNull(pair.Query);
        ArgumentNullException.ThrowIfNull(pair.Document);
        return _model.Tokenize(pair.Query, pair.Document);
    }

    /// <summary>Returns a raw logit for one pair; not a probability.</summary>
    public float Score(BertPair pair, CancellationToken cancellationToken = default) =>
        ScoreTokens([Tokenize(pair)], cancellationToken)[0];

    /// <summary>Scores pairs in input order, retaining pair segment ids.</summary>
    public float[] ScoreBatch(IReadOnlyList<BertPair> pairs, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pairs);
        cancellationToken.ThrowIfCancellationRequested();
        return ScoreTokens(pairs.Select(Tokenize).ToArray(), cancellationToken);
    }

    /// <summary>Scores supplied tokenized pair inputs, retaining masks and segment ids.</summary>
    public float[] ScoreTokens(IReadOnlyList<BertInput> inputs, CancellationToken cancellationToken = default) =>
        _model.Predict(inputs, cancellationToken).Select(row => row[0]).ToArray();

    /// <summary>
    /// Scores each document against <paramref name="query"/>, in document order: raw logits, as
    /// <see cref="ScoreBatch"/> returns them for the pairs. The <see cref="IReranker"/> form.
    /// </summary>
    public float[] Score(string query, IReadOnlyList<string> documents, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(documents);
        return ScoreBatch(documents.Select(document => new BertPair(query, document)).ToArray(), cancellationToken);
    }

    /// <summary>Ranks documents by descending logit, with stable original-index ties.</summary>
    public IReadOnlyList<BertRankedDocument> Rank(string query, IReadOnlyList<string> documents, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(documents);
        float[] scores = ScoreBatch(documents.Select(document => new BertPair(query, document)).ToArray(), cancellationToken);
        return documents.Select((document, index) => new BertRankedDocument(index, document, scores[index]))
            .OrderByDescending(row => row.Score).ThenBy(row => row.Index).ToArray();
    }

    /// <summary>Disposes the owned native session. Idempotent.</summary>
    public void Dispose() => _model.Dispose();
}
