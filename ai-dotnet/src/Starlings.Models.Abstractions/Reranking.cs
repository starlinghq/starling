namespace Starlings.Models;

/// <summary>One candidate after reranking: the candidate, its position before reranking, and its score.</summary>
/// <typeparam name="T">The candidate type, for example a vector-search result.</typeparam>
/// <param name="Item">The candidate.</param>
/// <param name="Index">The candidate's zero-based position in the input, before reranking.</param>
/// <param name="Score">The reranker's raw score for the candidate (see <see cref="IReranker"/>).</param>
public sealed record RerankedItem<T>(T Item, int Index, float Score);

/// <summary>
/// Helpers that put an <see cref="IReranker"/> after a first-stage retriever, for example the top-k of a
/// <c>Microsoft.Extensions.VectorData</c> <c>SearchAsync</c> call over a Starlings embedding generator.
/// Microsoft.Extensions.AI has no reranker abstraction, so the helpers take the candidates as any sequence plus a text
/// selector, and need no reference to either library. Experimental: the API may change.
/// </summary>
/// <remarks>
/// <see cref="Rerank{T}"/> scores on the calling thread. The delegate from <see cref="AsScoringFunction"/> scores
/// synchronously on the thread that invokes it and returns a completed (or faulted, or cancelled) task; failures are
/// never thrown synchronously by it, they are in the returned task. The asynchronous forms read the candidates
/// asynchronously, then score them all with one <see cref="IReranker.Score"/> call after the last candidate arrives,
/// on the thread that resumed it (a thread-pool thread when the source completes asynchronously, as a vector store's
/// <c>SearchAsync</c> usually does), so their task is not necessarily complete when they return. They validate their
/// arguments synchronously. Ordering is by descending score, ties in input order, a NaN score last.
/// </remarks>
public static class Reranking
{
    /// <summary>
    /// Wraps <paramref name="reranker"/> as a <c>Func&lt;string, IReadOnlyList&lt;string&gt;, Task&lt;float[]&gt;&gt;</c>,
    /// the delegate shape a pipeline can take without depending on <see cref="IReranker"/>. The task's result is the
    /// array <see cref="IReranker.Score"/> returns: one raw score per document, in document order.
    /// </summary>
    /// <param name="reranker">The reranker; the caller keeps ownership.</param>
    /// <param name="cancellationToken">Passed to every <see cref="IReranker.Score"/> call the delegate makes.</param>
    /// <returns>The delegate. It runs on the calling thread and returns a completed (or faulted, or cancelled) task.</returns>
    public static Func<string, IReadOnlyList<string>, Task<float[]>> AsScoringFunction(this IReranker reranker, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(reranker);
        return (query, documents) =>
        {
            try
            {
                return Task.FromResult(reranker.Score(query, documents, cancellationToken));
            }
            catch (OperationCanceledException ex) when (ex.CancellationToken.IsCancellationRequested || cancellationToken.IsCancellationRequested)
            {
                return Task.FromCanceled<float[]>(ex.CancellationToken.IsCancellationRequested ? ex.CancellationToken : cancellationToken);
            }
            catch (Exception ex)
            {
                return Task.FromException<float[]>(ex);
            }
        };
    }

    /// <summary>Scores <paramref name="candidates"/> against <paramref name="query"/> and orders them by descending score.</summary>
    /// <typeparam name="T">The candidate type.</typeparam>
    /// <param name="reranker">The reranker.</param>
    /// <param name="query">The query.</param>
    /// <param name="candidates">The candidates, for example a first-stage retriever's top-k.</param>
    /// <param name="text">Selects the text to score from a candidate; it must not return null.</param>
    /// <param name="top">Keep at most this many (at least 1), or null (default) to keep all.</param>
    /// <param name="cancellationToken">Passed to <see cref="IReranker.Score"/>.</param>
    /// <returns>The candidates by descending score, ties in input order; empty for no candidates (the reranker is not called).</returns>
    /// <exception cref="ArgumentException"><paramref name="text"/> returned null for a candidate.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="top"/> is below 1.</exception>
    /// <exception cref="InvalidOperationException">The reranker returned a score count other than the candidate count.</exception>
    public static IReadOnlyList<RerankedItem<T>> Rerank<T>(this IReranker reranker, string query, IEnumerable<T> candidates, Func<T, string> text, int? top = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(reranker);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(text);
        ValidateTop(top);
        var items = candidates as IReadOnlyList<T> ?? candidates.ToList();
        return Order(items, Texts(items, text), query, (q, d) => reranker.Score(q, d, cancellationToken), top);
    }

    /// <summary>
    /// Reads <paramref name="candidates"/> (for example the results of a vector store's <c>SearchAsync</c>), scores
    /// them against <paramref name="query"/> with one <see cref="IReranker.Score"/> call and orders them by descending score.
    /// </summary>
    /// <typeparam name="T">The candidate type, for example <c>VectorSearchResult&lt;TRecord&gt;</c>.</typeparam>
    /// <param name="reranker">The reranker.</param>
    /// <param name="query">The query, without any embedding prefix.</param>
    /// <param name="candidates">The candidates; read to the end before scoring.</param>
    /// <param name="text">Selects the text to score from a candidate; it must not return null.</param>
    /// <param name="top">Keep at most this many (at least 1), or null (default) to keep all.</param>
    /// <param name="cancellationToken">Cancels reading the candidates and scoring.</param>
    /// <returns>The candidates by descending score, ties in input order; empty for no candidates (the reranker is not called).</returns>
    /// <exception cref="ArgumentException"><paramref name="text"/> returned null for a candidate.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="top"/> is below 1.</exception>
    /// <exception cref="InvalidOperationException">The reranker returned a score count other than the candidate count.</exception>
    public static Task<IReadOnlyList<RerankedItem<T>>> RerankAsync<T>(this IReranker reranker, string query, IAsyncEnumerable<T> candidates, Func<T, string> text, int? top = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(reranker);
        return RerankAsync(query, candidates, text, reranker.AsScoringFunction(cancellationToken), top, cancellationToken);
    }

    /// <summary>
    /// Reads <paramref name="candidates"/>, scores them against <paramref name="query"/> with one call of
    /// <paramref name="score"/> and orders them by descending score. Any scorer with the delegate shape works, for
    /// example <see cref="AsScoringFunction"/> of an <see cref="IReranker"/> or a remote service.
    /// </summary>
    /// <typeparam name="T">The candidate type.</typeparam>
    /// <param name="query">The query.</param>
    /// <param name="candidates">The candidates; read to the end before scoring.</param>
    /// <param name="text">Selects the text to score from a candidate; it must not return null.</param>
    /// <param name="score">Returns one score per document, in document order; higher means more relevant.</param>
    /// <param name="top">Keep at most this many (at least 1), or null (default) to keep all.</param>
    /// <param name="cancellationToken">Cancels reading the candidates.</param>
    /// <returns>The candidates by descending score, ties in input order; empty for no candidates (the scorer is not called).</returns>
    /// <exception cref="ArgumentException"><paramref name="text"/> returned null for a candidate.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="top"/> is below 1.</exception>
    /// <exception cref="InvalidOperationException">The scorer returned null or a score count other than the candidate count.</exception>
    public static Task<IReadOnlyList<RerankedItem<T>>> RerankAsync<T>(string query, IAsyncEnumerable<T> candidates, Func<T, string> text, Func<string, IReadOnlyList<string>, Task<float[]>> score, int? top = null, CancellationToken cancellationToken = default)
    {
        // Validated here, synchronously; the async core only reads, scores and orders.
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(score);
        ValidateTop(top);
        return RerankCoreAsync(query, candidates, text, score, top, cancellationToken);
    }

    private static async Task<IReadOnlyList<RerankedItem<T>>> RerankCoreAsync<T>(string query, IAsyncEnumerable<T> candidates, Func<T, string> text, Func<string, IReadOnlyList<string>, Task<float[]>> score, int? top, CancellationToken cancellationToken)
    {
        var items = new List<T>();
        await foreach (var item in candidates.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            items.Add(item);
        }

        var texts = Texts(items, text);
        float[]? scores = items.Count == 0 ? [] : await score(query, texts).ConfigureAwait(false);
        return Order(items, texts, query, (_, _) => scores, top);
    }

    private static void ValidateTop(int? top)
    {
        if (top is { } value)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(value, 1, nameof(top));
        }
    }

    private static string[] Texts<T>(IReadOnlyList<T> items, Func<T, string> text)
    {
        var texts = new string[items.Count];
        for (int i = 0; i < texts.Length; i++)
        {
            texts[i] = text(items[i]) ?? throw new ArgumentException($"The text selector returned null for candidate {i}.", nameof(text));
        }

        return texts;
    }

    private static RerankedItem<T>[] Order<T>(IReadOnlyList<T> items, string[] texts, string query, Func<string, IReadOnlyList<string>, float[]?> score, int? top)
    {
        if (items.Count == 0)
        {
            return [];
        }

        var scores = score(query, texts);
        if (scores is null || scores.Length != items.Count)
        {
            throw new InvalidOperationException(
                $"The reranker returned {(scores is null ? "null" : $"{scores.Length} scores")} for {items.Count} candidates.");
        }

        var ranked = new RerankedItem<T>[items.Count];
        for (int i = 0; i < ranked.Length; i++)
        {
            ranked[i] = new RerankedItem<T>(items[i], i, scores[i]);
        }

        // Descending score, ties in input order (Array.Sort is unstable, so the index breaks ties). NaN sorts last.
        Array.Sort(ranked, static (a, b) =>
        {
            int byScore = b.Score.CompareTo(a.Score);
            return byScore != 0 ? byScore : a.Index.CompareTo(b.Index);
        });
        return top is { } keep && keep < ranked.Length ? ranked[..keep] : ranked;
    }
}
