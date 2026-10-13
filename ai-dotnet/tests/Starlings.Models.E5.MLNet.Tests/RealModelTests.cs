using Starlings.TestSupport;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.E5.MLNet.Tests;

/// <summary>
/// One transformer over the real e5 snapshot per test class, fitted from <see cref="E5ModelFiles"/> through
/// <c>mlContext.Transforms.ApplyE5Embedding</c> (model and tokenizer digests enforced; the transformer owns its
/// session), or a skip reason when the snapshot is absent.
/// </summary>
public sealed class E5TransformerFixture : IDisposable
{
    private readonly E5EmbeddingTransformer? _transformer;

    public E5TransformerFixture()
    {
        var (files, reason) = E5TestAssets.FindFiles();
        if (files is null)
        {
            SkipReason = reason;
            return;
        }

        Files = files;
        var ml = new MLContext(seed: 1);
        var empty = ml.Data.LoadFromEnumerable(new List<FixtureRow>());
        _transformer = ml.Transforms.ApplyE5Embedding(files, "Embedding", "Text", new E5EmbeddingOptions { TokenCountColumnName = "TokenCount" }).Fit(empty);
    }

    public string? SkipReason { get; }

    public E5ModelFiles? Files { get; }

    public E5EmbeddingTransformer Transformer
    {
        get
        {
            Assert.SkipWhen(_transformer is null, SkipReason ?? "model unavailable");
            return _transformer!;
        }
    }

    public void Dispose() => _transformer?.Dispose();
}

public sealed class FixtureRow
{
    public string Id { get; set; } = string.Empty;

    public string Text { get; set; } = string.Empty;
}

public sealed class FixtureEmbedding
{
    public string Id { get; set; } = string.Empty;

    [VectorType(384)]
    public float[] Embedding { get; set; } = [];

    public int TokenCount { get; set; }
}

/// <summary>
/// The transform on the real model: embeddings equal <see cref="E5TextEmbedder"/> exactly and meet the protected
/// binding tolerance (<c>model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/tolerances.yaml</c>, read only);
/// look-ahead batching equals the batched embedder; owned disposal. Skips when the snapshot is absent.
/// </summary>
public class RealModelTests(E5TransformerFixture fixture, ITestOutputHelper output) : IClassFixture<E5TransformerFixture>
{
    private static readonly E5Tolerances Tolerances = E5Tolerances.Load(E5TestAssets.CpuFixtureDirectory);

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static IDataView FixtureView(MLContext ml) =>
        ml.Data.LoadFromEnumerable(E5Fixtures.Cases.Select(c => new FixtureRow { Id = c.Id, Text = E5Fixtures.Texts[c.Id] }).ToList());

    [Fact]
    public void RowWiseEmbeddingsEqualTheTextEmbedderAndMeetTheProtectedTolerance()
    {
        var transformer = fixture.Transformer;
        var ml = new MLContext(seed: 1);
        Assert.Equal(384, transformer.Dimension);
        Assert.Equal(E5TestAssets.ModelSha256, transformer.Embedder.Encoder.Session.Diagnostics.ModelSha256);

        var rows = ml.Data.CreateEnumerable<FixtureEmbedding>(transformer.Transform(FixtureView(ml)), reuseRowObject: false).ToList();
        var cases = E5Fixtures.Cases;
        Assert.Equal(46, rows.Count);
        var vsProtected = new ErrorStats();
        for (int i = 0; i < cases.Count; i++)
        {
            Assert.Equal(cases[i].Id, rows[i].Id);
            Assert.Equal(transformer.Embedder.Embed(E5Fixtures.Texts[cases[i].Id], Ct), rows[i].Embedding);
            Assert.Equal(cases[i].InputIds.Length, rows[i].TokenCount);
            vsProtected.Add(rows[i].Embedding, cases[i].Embedding);
        }

        output.WriteLine($"ML.NET transform, row-wise, {cases.Count} texts vs protected: {vsProtected}");
        output.WriteLine($"bound: embedding_max_abs {Tolerances.EmbeddingMaxAbs:E1} from {Tolerances.Source}");
        Assert.False(vsProtected.NonFinite);
        Assert.True(vsProtected.Max <= Tolerances.EmbeddingMaxAbs, $"transform vs protected {vsProtected.Max:E3}");
    }

    [Fact]
    public void LookAheadBatchingEqualsTheBatchedEmbedderAndMeetsTheProtectedTolerance()
    {
        var single = fixture.Transformer;
        var ml = new MLContext(seed: 1);
        const int batchSize = 16;
        using var batched = new E5EmbeddingTransformer(single.Embedder, "Embedding", "Text", new E5EmbeddingOptions { BatchSize = batchSize, TokenCountColumnName = "TokenCount" });
        // Wall time of one pass each (one machine, warm session; an observation, not a benchmark).
        var rowWiseTime = System.Diagnostics.Stopwatch.StartNew();
        _ = ml.Data.CreateEnumerable<FixtureEmbedding>(single.Transform(FixtureView(ml)), reuseRowObject: false).ToList();
        rowWiseTime.Stop();
        var batchedTime = System.Diagnostics.Stopwatch.StartNew();
        var rows = ml.Data.CreateEnumerable<FixtureEmbedding>(batched.Transform(FixtureView(ml)), reuseRowObject: false).ToList();
        batchedTime.Stop();
        output.WriteLine($"46 fixture texts (2 to 512 tokens): row-wise {rowWiseTime.Elapsed.TotalMilliseconds:F0} ms, batches of {batchSize} {batchedTime.Elapsed.TotalMilliseconds:F0} ms");
        var similar = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 256).Select(i => new FixtureRow { Id = $"s{i}", Text = $"query: how much protein should a person eat on day {i}" }).ToList());
        foreach (var transformer in new[] { single, batched, single, batched })
        {
            var time = System.Diagnostics.Stopwatch.StartNew();
            _ = ml.Data.CreateEnumerable<FixtureEmbedding>(transformer.Transform(similar), reuseRowObject: false).Count();
            output.WriteLine($"256 similar short texts, batch size {transformer.Options.BatchSize}: {time.Elapsed.TotalMilliseconds:F0} ms");
        }

        var cases = E5Fixtures.Cases;
        var texts = cases.Select(c => E5Fixtures.Texts[c.Id]).ToList();

        var vsProtected = new ErrorStats();
        var vsRowWise = new ErrorStats();
        for (int start = 0; start < texts.Count; start += batchSize)
        {
            var chunk = texts.Skip(start).Take(batchSize).ToList();
            var expected = single.Embedder.EmbedBatch(chunk, Ct);
            for (int j = 0; j < chunk.Count; j++)
            {
                int i = start + j;
                Assert.Equal(expected[j], rows[i].Embedding);
                Assert.Equal(cases[i].InputIds.Length, rows[i].TokenCount);
                vsProtected.Add(rows[i].Embedding, cases[i].Embedding);
                vsRowWise.Add(rows[i].Embedding, single.Embedder.Embed(texts[i], Ct));
            }
        }

        output.WriteLine($"ML.NET transform, batches of {batchSize}, vs protected: {vsProtected}");
        output.WriteLine($"batched vs row-wise: {vsRowWise}");
        Assert.True(vsProtected.Max <= Tolerances.EmbeddingMaxAbs, $"batched vs protected {vsProtected.Max:E3}");
        Assert.True(vsRowWise.Max <= Tolerances.PaddedBatchVsSingleMaxAbs, $"batched vs row-wise {vsRowWise.Max:E3}");
    }

    [Fact]
    public void FittedChainOwnsTheSessionAndDisposingTheChainReleasesIt()
    {
        _ = fixture.Transformer;
        var ml = new MLContext(seed: 1);
        var view = FixtureView(ml);
        E5Encoder encoder;
        var options = new E5EmbeddingOptions { Prefix = E5TextPrefix.Query, TokenCountColumnName = "TokenCount" };
        using (var chain = ml.Transforms.CopyColumns("Copy", "Text").Append(ml.Transforms.ApplyE5Embedding(fixture.Files!, "Embedding", "Copy", options)).Fit(view))
        {
            encoder = chain.LastTransformer.Embedder.Encoder;
            Assert.NotSame(fixture.Transformer.Embedder.Encoder, encoder);
            var first = ml.Data.CreateEnumerable<FixtureEmbedding>(chain.Transform(view), reuseRowObject: false).First();
            Assert.Equal(fixture.Transformer.Embedder.Embed("query: " + E5Fixtures.Texts[first.Id], Ct), first.Embedding);
        }

        // TransformerChain.Dispose disposed the fitted transformer, which owned its encoder and session.
        Assert.Throws<ObjectDisposedException>(() => encoder.Embed(new EncoderInput(new long[] { 0, 2 }), Ct));
    }
}
