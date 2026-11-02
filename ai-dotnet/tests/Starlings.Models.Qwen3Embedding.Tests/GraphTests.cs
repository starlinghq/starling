using Microsoft.Extensions.AI;
using Starlings.Models.Qwen3Embedding.Extensions.AI;

namespace Starlings.Models.Qwen3Embedding.Tests;

/// <summary>
/// The embedder and the reranker on released graphs (built locally with the export recipe; skipped without them),
/// against the PyTorch fp32 reference in TestData and the model card's published cosines. The bounds are the issue's
/// provisional ones (starlinghq/starling#144): embedding 1e-4, score 1e-3 (log-odds), card cosines 1e-3.
/// </summary>
public sealed class GraphTests
{
    private const float EmbeddingBound = 1e-4f;
    private const float ScoreBound = 1e-3f;
    private static readonly string[] CardQueries = ["What is the capital of China?", "Explain gravity"];
    private static readonly string[] CardDocuments =
    [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ];

    private static readonly double[][] CardCosines = [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]];

    private static readonly Lazy<Qwen3TextEmbedder?> SharedEmbedder = new(() =>
        Qwen3TestAssets.Release(Qwen3EmbeddingProfile.Embedding06B).Directory is { } dir ? Qwen3TextEmbedder.Create(dir, new Qwen3EmbeddingOptions { MaxBatchSize = 4 }) : null);

    private static readonly Lazy<Qwen3Reranker?> SharedReranker = new(() =>
        Qwen3TestAssets.Release(Qwen3EmbeddingProfile.Reranker06B).Directory is { } dir ? Qwen3Reranker.Create(dir, new Qwen3EmbeddingOptions { MaxBatchSize = 4 }) : null);

    private static Qwen3TextEmbedder Embedder()
    {
        var embedder = SharedEmbedder.Value;
        if (embedder is null)
        {
            Assert.Skip(Qwen3TestAssets.Release(Qwen3EmbeddingProfile.Embedding06B).Reason);
        }

        return embedder;
    }

    private static Qwen3Reranker Reranker()
    {
        var reranker = SharedReranker.Value;
        if (reranker is null)
        {
            Assert.Skip(Qwen3TestAssets.Release(Qwen3EmbeddingProfile.Reranker06B).Reason);
        }

        return reranker;
    }

    [Fact]
    public void EmbeddingsMatchTheReference()
    {
        var embedder = Embedder();
        Assert.Equal(1024, embedder.Dimension);
        foreach (var c in Qwen3TestAssets.Reference("embedding.json").EnumerateArray())
        {
            float[] e = embedder.Embed(c.GetProperty("text").GetString()!, c.GetProperty("prompt").GetString()!, TestContext.Current.CancellationToken);
            Assert.Equal(1024, e.Length);
            Assert.Equal(1.0, Math.Sqrt(e.Sum(v => (double)v * v)), 5);
            float[] head = c.GetProperty("head").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            for (int i = 0; i < head.Length; i++)
            {
                Assert.True(Math.Abs(e[i] - head[i]) <= EmbeddingBound, $"{c.GetProperty("text")}[{i}]: {e[i]} vs {head[i]}");
            }

            Assert.True(Math.Abs(e.Sum() - c.GetProperty("sum").GetSingle()) <= 1e-3f, $"sum {e.Sum()} vs {c.GetProperty("sum")}");
        }
    }

    [Fact]
    public void CardCosinesMatchThePublishedValues()
    {
        var embedder = Embedder();
        float[][] queries = embedder.EmbedBatch(CardQueries, "query", TestContext.Current.CancellationToken);
        float[][] documents = embedder.EmbedBatch(CardDocuments, "none", TestContext.Current.CancellationToken);
        for (int q = 0; q < 2; q++)
        {
            for (int d = 0; d < 2; d++)
            {
                double cosine = queries[q].Zip(documents[d], (a, b) => (double)a * b).Sum();
                Assert.True(Math.Abs(cosine - CardCosines[q][d]) <= 1e-3, $"cosine[{q},{d}] {cosine} vs {CardCosines[q][d]}");
            }
        }
    }

    [Fact]
    public void PaddedBatchEqualsSingleRows()
    {
        var embedder = Embedder();
        string[] texts = [.. CardDocuments, string.Empty, "short"];
        var counts = new int[texts.Length];
        float[][] batch = embedder.EmbedBatch(texts, counts, TestContext.Current.CancellationToken);
        Assert.Equal(embedder.CountTokens(texts), counts);
        for (int i = 0; i < texts.Length; i++)
        {
            float[] single = embedder.Embed(texts[i], cancellationToken: TestContext.Current.CancellationToken);
            Assert.True(batch[i].Zip(single, (a, b) => Math.Abs(a - b)).Max() <= 1e-5f, $"row {i}");
        }

        Assert.Equal(8192, embedder.MaxSequenceLength);
        Assert.False(embedder.Validate("short").ExceedsLimit);
    }

    [Fact]
    public async Task GeneratorUsesTheNamedPrompt()
    {
        var embedder = Embedder();
        using var generator = new Qwen3EmbeddingGenerator(embedder);
        var query = await generator.GenerateAsync([CardQueries[0]], new EmbeddingGenerationOptions
        {
            AdditionalProperties = new() { [Qwen3EmbeddingGenerator.PromptPropertyName] = "query" },
        }, TestContext.Current.CancellationToken);
        float[] expected = embedder.Embed(CardQueries[0], "query", TestContext.Current.CancellationToken);
        Assert.Equal(expected, query[0].Vector.ToArray());
        Assert.Equal(embedder.CountTokens(CardQueries[0], "query"), query.Usage!.InputTokenCount);
        Assert.Equal(1024, generator.Metadata.DefaultModelDimensions);

        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { Dimensions = 256 }, TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions
        {
            AdditionalProperties = new() { [Qwen3EmbeddingGenerator.PromptPropertyName] = "passage" },
        }, TestContext.Current.CancellationToken));
        Assert.Same(embedder, generator.GetService(typeof(Qwen3TextEmbedder)));
    }

    [Fact]
    public void RerankerScoresMatchTheReference()
    {
        var reranker = Reranker();
        var cases = Qwen3TestAssets.Reference("reranker.json").GetProperty("cases").EnumerateArray().ToArray();
        var inputs = cases.Select(c => reranker.Tokenize(c.GetProperty("query").GetString()!, c.GetProperty("document").GetString()!, c.GetProperty("instruction").GetString())).ToArray();
        float[] batch = reranker.ScoreTokens(inputs, TestContext.Current.CancellationToken);
        for (int i = 0; i < cases.Length; i++)
        {
            float expected = cases[i].GetProperty("score").GetSingle();
            Assert.True(Math.Abs(batch[i] - expected) <= ScoreBound, $"case {i}: {batch[i]} vs {expected}");
            float single = reranker.ScoreTokens([inputs[i]], TestContext.Current.CancellationToken)[0];
            Assert.True(Math.Abs(single - batch[i]) <= 1e-4f, $"case {i}: single {single} vs batch {batch[i]}");
        }
    }

    [Fact]
    public void RerankerRanksTheCardExamples()
    {
        var reranker = Reranker();
        float[] capital = reranker.Score(CardQueries[0], CardDocuments, TestContext.Current.CancellationToken);
        float[] gravity = reranker.Score(CardQueries[1], CardDocuments, TestContext.Current.CancellationToken);
        Assert.True(capital[0] > 0 && capital[1] < 0);
        Assert.True(gravity[1] > 0 && gravity[0] < 0);
        var ranked = reranker.Rank(CardQueries[1], CardDocuments, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal([1, 0], ranked.Select(r => r.Index));
    }
}
