namespace Starlings.Models;

/// <summary>
/// Query-document relevance scoring: one score per document, in document order. Higher means more relevant.
/// </summary>
/// <remarks>
/// Scores are the model's raw output (for a cross-encoder, a logit), not probabilities, and are comparable only
/// between documents scored for the same query by the same model. Each implementation states its activation.
/// </remarks>
public interface IReranker
{
    /// <summary>Scores each of <paramref name="documents"/> against <paramref name="query"/>.</summary>
    /// <param name="query">The query.</param>
    /// <param name="documents">The documents; none may be null.</param>
    /// <param name="cancellationToken">Cancels the call; nothing is returned for a cancelled call.</param>
    /// <returns>One score per document, in document order.</returns>
    float[] Score(string query, IReadOnlyList<string> documents, CancellationToken cancellationToken = default);
}
