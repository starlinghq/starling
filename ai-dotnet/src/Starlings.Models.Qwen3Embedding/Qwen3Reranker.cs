using Starlings.Runtime.Onnx;

namespace Starlings.Models.Qwen3Embedding;

/// <summary>A document and its reranker score, as <see cref="Qwen3Reranker.Rank"/> returns it.</summary>
/// <param name="Index">Position of the document in the call.</param>
/// <param name="Document">The document.</param>
/// <param name="Score">Raw log-odds <c>logit[yes] - logit[no]</c>.</param>
public sealed record Qwen3RankedDocument(int Index, string Document, float Score);

/// <summary>
/// <c>Qwen/Qwen3-Reranker-0.6B</c> query-document relevance on ONNX Runtime CPU (fp32). The input is the model card's
/// template, <c>prefix + "&lt;Instruct&gt;: {instruction}\n&lt;Query&gt;: {query}\n&lt;Document&gt;: {document}" + suffix</c>
/// (<see cref="Qwen3EmbeddingProfile.RerankPrefix"/>, <see cref="Qwen3EmbeddingProfile.RerankSuffix"/>), and the score is
/// the raw log-odds <c>logit[yes] - logit[no]</c> at the last position (sentence-transformers <c>LogitScore</c>), not a
/// probability: <see cref="Sigmoid"/> turns it into the card's <c>P(yes)</c>. Experimental
/// (<see cref="Qwen3EmbeddingCapabilities.Reranking"/>).
/// </summary>
/// <remarks>
/// A pair longer than 8192 tokens has its instruction, query and document text truncated from the right (longest
/// first); the template's prefix and suffix are always kept. Concurrent calls are safe; <see cref="Dispose"/> is
/// idempotent.
/// </remarks>
public sealed class Qwen3Reranker : IReranker, IDisposable
{
    private readonly Qwen3Session _session;

    private Qwen3Reranker(Qwen3Session session) => _session = session;

    /// <summary>The profile.</summary>
    public Qwen3EmbeddingProfile Profile => _session.Profile;

    /// <summary>Tokens per pair at most (8192), template included.</summary>
    public int MaxSequenceLength => Profile.MaxTokens;

    /// <summary>Loaded model and native runtime diagnostics.</summary>
    public OnnxSessionDiagnostics Diagnostics => _session.Session.Diagnostics;

    /// <summary>
    /// Verifies a released directory (<see cref="Qwen3EmbeddingProfile.OpenDirectory"/>) and creates a CPU session.
    /// Never downloads.
    /// </summary>
    /// <param name="directory">The directory <c>release_qwen3_onnx.py</c> wrote for <see cref="Qwen3EmbeddingProfile.Reranker06B"/>.</param>
    /// <param name="options">Threads, batching and verification.</param>
    /// <param name="profile">The profile; <see cref="Qwen3EmbeddingProfile.Reranker06B"/> when null.</param>
    public static Qwen3Reranker Create(string directory, Qwen3EmbeddingOptions? options = null, Qwen3EmbeddingProfile? profile = null)
    {
        profile ??= Qwen3EmbeddingProfile.Reranker06B;
        if (profile.Task != Qwen3EmbeddingTask.Reranking)
        {
            throw new ArgumentException("Use Qwen3TextEmbedder for an embedding profile.", nameof(profile));
        }

        return new Qwen3Reranker(new Qwen3Session(profile, directory, options));
    }

    /// <summary><c>1 / (1 + e^-score)</c>: the probability of <c>"yes"</c> over the two tokens, as the model card computes it.</summary>
    public static float Sigmoid(float score) => (float)(1.0 / (1.0 + Math.Exp(-score)));

    /// <summary>Tokenizes one pair with the template (only the content is truncated).</summary>
    /// <param name="query">The query.</param>
    /// <param name="document">The document.</param>
    /// <param name="instruction">The task; <see cref="Qwen3EmbeddingProfile.DefaultRerankInstruction"/> when null.</param>
    public Qwen3Input Tokenize(string query, string document, string? instruction = null) =>
        _session.Tokenization.TokenizePair(instruction ?? Qwen3EmbeddingProfile.DefaultRerankInstruction, query, document);

    /// <summary>Tokens a pair needs, template included, before truncation.</summary>
    public int CountTokens(string query, string document, string? instruction = null) =>
        _session.Tokenization.CountPair(instruction ?? Qwen3EmbeddingProfile.DefaultRerankInstruction, query, document);

    /// <summary>Scores each document against <paramref name="query"/> with the default instruction, in document order. The <see cref="IReranker"/> form.</summary>
    public float[] Score(string query, IReadOnlyList<string> documents, CancellationToken cancellationToken = default) =>
        Score(query, documents, instruction: null, cancellationToken);

    /// <summary>Scores each document against <paramref name="query"/>, in document order: raw log-odds.</summary>
    /// <param name="query">The query.</param>
    /// <param name="documents">The documents; none may be null.</param>
    /// <param name="instruction">The task; <see cref="Qwen3EmbeddingProfile.DefaultRerankInstruction"/> when null.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    public float[] Score(string query, IReadOnlyList<string> documents, string? instruction, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(documents);
        var inputs = new Qwen3Input[documents.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            inputs[i] = Tokenize(query, documents[i] ?? throw new ArgumentException($"Document {i} is null.", nameof(documents)), instruction);
        }

        return ScoreTokens(inputs, cancellationToken);
    }

    /// <summary>Scores tokenized rows (from <see cref="Tokenize"/>).</summary>
    public float[] ScoreTokens(IReadOnlyList<Qwen3Input> inputs, CancellationToken cancellationToken = default) =>
        _session.Predict(inputs, cancellationToken).Select(row => row[0]).ToArray();

    /// <summary>Documents by descending score; ties keep their input order.</summary>
    public IReadOnlyList<Qwen3RankedDocument> Rank(string query, IReadOnlyList<string> documents, string? instruction = null, CancellationToken cancellationToken = default)
    {
        float[] scores = Score(query, documents, instruction, cancellationToken);
        return documents.Select((document, index) => new Qwen3RankedDocument(index, document, scores[index]))
            .OrderByDescending(row => row.Score).ThenBy(row => row.Index).ToArray();
    }

    /// <summary>Disposes the native session. Idempotent.</summary>
    public void Dispose() => _session.Dispose();
}
