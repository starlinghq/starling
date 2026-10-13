using Starlings.TestSupport;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.EmbeddingGemma2.MLNet.Tests;

/// <summary>
/// One transformer over the real graph per test class, fitted from <see cref="EmbeddingGemma2ModelFiles"/> through
/// <c>mlContext.Transforms.ApplyEmbeddingGemma2Embedding</c> (graph digest enforced; the transformer owns its session)
/// with the owned tokenizer and the <c>document</c> prompt, or a skip reason when the graph is
/// absent.
/// </summary>
public sealed class EmbeddingGemma2TransformerFixture : IDisposable
{
    private readonly EmbeddingGemma2EmbeddingTransformer? _transformer;

    public EmbeddingGemma2TransformerFixture()
    {
        Rows = EmbeddingGemma2Fixtures.LoadRows();
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        if (files is null)
        {
            SkipReason = reason;
            return;
        }

        var ml = new MLContext(seed: 1);
        var empty = ml.Data.LoadFromEnumerable(new List<FixtureRow>());
        _transformer = ml.Transforms.ApplyEmbeddingGemma2Embedding(
            files,
            "Embedding",
            "Text",
            new EmbeddingGemma2EmbeddingOptions { TokenCountColumnName = "TokenCount", Prompt = EmbeddingGemma2Prompt.Document, BatchSize = 4 }).Fit(empty);
    }

    public string? SkipReason { get; }

    public IReadOnlyList<EmbeddingGemma2Row> Rows { get; }

    public EmbeddingGemma2EmbeddingTransformer Transformer
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

    [VectorType(768)]
    public float[] Embedding { get; set; } = [];

    public int TokenCount { get; set; }
}

/// <summary>
/// The transform on the real graph against the protected <c>document</c>-prompt rows with the binding tolerance of
/// <c>onnx-fp32-cpu</c> (look-ahead batches of 4, padded). Skips when the graph is absent.
/// </summary>
public class RealModelTests(EmbeddingGemma2TransformerFixture fixture, ITestOutputHelper output) : IClassFixture<EmbeddingGemma2TransformerFixture>
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("onnx-fp32-cpu");

    [Fact]
    public void DocumentRowsMeetTheProtectedTolerance()
    {
        var transformer = fixture.Transformer;
        var rows = fixture.Rows.Where(r => r.Prompt == "document" && r.InputIds.Length < 400).ToList();
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(rows.Select(r => new FixtureRow { Id = r.Key, Text = r.Text }).ToList());
        var embedded = ml.Data.CreateEnumerable<FixtureEmbedding>(transformer.Transform(data), reuseRowObject: false).ToList();
        var stats = new EmbeddingGemma2ErrorStats();
        for (int i = 0; i < rows.Count; i++)
        {
            Assert.Equal(rows[i].Key, embedded[i].Id);
            Assert.Equal(rows[i].InputIds.Length, embedded[i].TokenCount);
            stats.Add(rows[i].Key, embedded[i].Embedding, rows[i].Embedding);
        }

        output.WriteLine($"ML.NET transform, {rows.Count} document rows in look-ahead batches of 4 vs protected: {stats}");
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs, $"{stats.Max:E3}");
        Assert.True(stats.MinCosine >= Tolerances.MinCosine);
    }

    [Fact]
    public void SessionIsTheLockedGraph() =>
        Assert.Equal(EmbeddingGemma2ModelAssets.ModelSha256, fixture.Transformer.Embedder.Encoder.Session.Diagnostics.ModelSha256);
}
