using Starlings.Models;

namespace Starlings.Models.Bert.Tests;

public sealed class RerankingTests
{
    // Scores each document by a fixed table, so the expected order is known without a model.
    private sealed class TableReranker(Func<string, float> score) : IReranker
    {
        public int Calls { get; private set; }
        public string? LastQuery { get; private set; }
        public CancellationToken LastToken { get; private set; }

        public float[] Score(string query, IReadOnlyList<string> documents, CancellationToken cancellationToken = default)
        {
            Calls++;
            LastQuery = query;
            LastToken = cancellationToken;
            cancellationToken.ThrowIfCancellationRequested();
            return documents.Select(d => d is null ? throw new ArgumentException("null document") : score(d)).ToArray();
        }
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private sealed record Hit(string Id, string Text);

    private static readonly Hit[] Hits =
    [
        new("a", "low"),
        new("b", "high"),
        new("c", "mid"),
        new("d", "high"),
    ];

    private static readonly string[] Letters = ["bb", "a", "ccc"];

    private static float Table(string text) => text switch { "high" => 2f, "mid" => 1f, _ => -1f };

    private static async IAsyncEnumerable<T> Stream<T>(IEnumerable<T> items)
    {
        foreach (var item in items)
        {
            await Task.Yield();
            yield return item;
        }
    }

    [Fact]
    public void RerankOrdersByDescendingScoreWithTiesInInputOrder()
    {
        var reranker = new TableReranker(Table);
        var ranked = reranker.Rerank("q", Hits, h => h.Text, cancellationToken: Ct);
        Assert.Equal(["b", "d", "c", "a"], ranked.Select(r => r.Item.Id));
        Assert.Equal([1, 3, 2, 0], ranked.Select(r => r.Index));
        Assert.Equal([2f, 2f, 1f, -1f], ranked.Select(r => r.Score));
        Assert.Equal(1, reranker.Calls);
        Assert.Equal("q", reranker.LastQuery);
    }

    [Fact]
    public void TopKeepsTheBestAndIsClampedToTheCandidateCount()
    {
        var reranker = new TableReranker(Table);
        Assert.Equal(["b", "d"], reranker.Rerank("q", Hits, h => h.Text, top: 2, cancellationToken: Ct).Select(r => r.Item.Id));
        Assert.Equal(4, reranker.Rerank("q", Hits, h => h.Text, top: 10, cancellationToken: Ct).Count);
        Assert.Throws<ArgumentOutOfRangeException>(() => reranker.Rerank("q", Hits, h => h.Text, top: 0, cancellationToken: Ct));
    }

    [Fact]
    public async Task NoCandidatesReturnsEmptyWithoutCallingTheReranker()
    {
        var reranker = new TableReranker(Table);
        Assert.Empty(reranker.Rerank("q", Array.Empty<Hit>(), h => h.Text, cancellationToken: Ct));
        Assert.Empty(await reranker.RerankAsync("q", Stream(Array.Empty<Hit>()), h => h.Text, cancellationToken: Ct));
        Assert.Equal(0, reranker.Calls);
    }

    [Fact]
    public async Task RerankAsyncReadsTheStreamAndMatchesTheSynchronousForm()
    {
        var reranker = new TableReranker(Table);
        var ranked = await reranker.RerankAsync("q", Stream(Hits), h => h.Text, top: 3, cancellationToken: Ct);
        Assert.Equal(reranker.Rerank("q", Hits, h => h.Text, top: 3, cancellationToken: Ct), ranked);
        Assert.Equal(2, reranker.Calls);
    }

    [Fact]
    public async Task ANullTextIsRejected()
    {
        var reranker = new TableReranker(Table);
        Assert.Throws<ArgumentException>("text", () => reranker.Rerank("q", Hits, h => h.Id == "c" ? null! : h.Text, cancellationToken: Ct));
        await Assert.ThrowsAsync<ArgumentException>("text", () => reranker.RerankAsync("q", Stream(Hits), h => h.Id == "c" ? null! : h.Text, cancellationToken: Ct));
        Assert.Equal(0, reranker.Calls);
    }

    [Fact]
    public async Task AScoreCountMismatchIsRejected()
    {
        Func<string, IReadOnlyList<string>, Task<float[]>> shortScorer = (_, _) => Task.FromResult<float[]>([1f]);
        await Assert.ThrowsAsync<InvalidOperationException>(() => Reranking.RerankAsync("q", Stream(Hits), h => h.Text, shortScorer, cancellationToken: Ct));
        Func<string, IReadOnlyList<string>, Task<float[]>> nullScorer = (_, _) => Task.FromResult<float[]>(null!);
        await Assert.ThrowsAsync<InvalidOperationException>(() => Reranking.RerankAsync("q", Stream(Hits), h => h.Text, nullScorer, cancellationToken: Ct));
    }

    [Fact]
    public async Task TheScoringFunctionReturnsTheRerankersScoresInDocumentOrder()
    {
        var reranker = new TableReranker(Table);
        var score = reranker.AsScoringFunction(Ct);
        float[] scores = await score("q", ["mid", "high", "low"]);
        Assert.Equal([1f, 2f, -1f], scores);
        Assert.Equal("q", reranker.LastQuery);
    }

    [Fact]
    public async Task TheScoringFunctionPutsFailuresInTheTask()
    {
        var reranker = new TableReranker(Table);
        var score = reranker.AsScoringFunction(Ct);
        Task<float[]> failed = score("q", [null!]);
        Assert.True(failed.IsFaulted);
        await Assert.ThrowsAsync<ArgumentException>(() => failed);
    }

    [Fact]
    public async Task CancellationReachesTheRerankerAndCancelsTheTask()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var reranker = new TableReranker(Table);

        Task<float[]> cancelled = reranker.AsScoringFunction(cts.Token)("q", ["high"]);
        Assert.True(cancelled.IsCanceled);
        Assert.Equal(cts.Token, reranker.LastToken);

        Assert.Throws<OperationCanceledException>(() => reranker.Rerank("q", Hits, h => h.Text, cancellationToken: cts.Token));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => reranker.RerankAsync("q", Stream(Hits), h => h.Text, cancellationToken: cts.Token));
    }

    [Fact]
    public async Task ANaNScoreSortsLast()
    {
        Func<string, IReadOnlyList<string>, Task<float[]>> nanFirst = (_, _) => Task.FromResult<float[]>([float.NaN, 1f, -1f]);
        var ranked = await Reranking.RerankAsync("q", Stream(Letters), s => s, nanFirst, cancellationToken: Ct);
        Assert.Equal([1, 2, 0], ranked.Select(r => r.Index));
        Assert.True(float.IsNaN(ranked[^1].Score));
        var reranker = new TableReranker(t => t == "a" ? float.NaN : t.Length);
        Assert.Equal(["ccc", "bb", "a"], reranker.Rerank("q", Letters, s => s, cancellationToken: Ct).Select(r => r.Item));
    }

    [Fact]
    public void TheAsynchronousFormsValidateArgumentsSynchronously()
    {
        var reranker = new TableReranker(Table);
        Func<string, IReadOnlyList<string>, Task<float[]>> score = reranker.AsScoringFunction(Ct);
        var hits = Stream(Hits);
        Assert.Throws<ArgumentNullException>("reranker", () => { _ = Reranking.RerankAsync(null!, "q", hits, h => h.Text, cancellationToken: Ct); });
        Assert.Throws<ArgumentNullException>("query", () => { _ = reranker.RerankAsync(null!, hits, h => h.Text, cancellationToken: Ct); });
        Assert.Throws<ArgumentNullException>("candidates", () => { _ = Reranking.RerankAsync("q", (IAsyncEnumerable<Hit>)null!, h => h.Text, score, cancellationToken: Ct); });
        Assert.Throws<ArgumentNullException>("text", () => { _ = Reranking.RerankAsync("q", hits, null!, score, cancellationToken: Ct); });
        Assert.Throws<ArgumentNullException>("score", () => { _ = Reranking.RerankAsync("q", hits, h => h.Text, null!, cancellationToken: Ct); });
        Assert.Throws<ArgumentOutOfRangeException>("top", () => { _ = reranker.RerankAsync("q", hits, h => h.Text, top: 0, cancellationToken: Ct); });
        Assert.Equal(0, reranker.Calls);
    }

    [Fact]
    public async Task AnyScorerWithTheDelegateShapeCanRerank()
    {
        Func<string, IReadOnlyList<string>, Task<float[]>> lengthScorer =
            async (_, docs) => { await Task.Yield(); return docs.Select(d => (float)d.Length).ToArray(); };
        var ranked = await Reranking.RerankAsync("q", Stream(Letters), s => s, lengthScorer, top: 2, cancellationToken: Ct);
        Assert.Equal(["ccc", "bb"], ranked.Select(r => r.Item));
    }
}
