using Starlings.TestSupport;
using Microsoft.Extensions.AI;

namespace Starlings.Models.EmbeddingGemma2.Extensions.AI.Tests;

/// <summary>
/// One generator over the real graph per test class, from <see cref="EmbeddingGemma2EmbeddingGenerator.Create"/> (graph
/// and tokenizer digests enforced, owned tokenizer), or a skip reason when the graph is absent.
/// </summary>
public sealed class EmbeddingGemma2GeneratorFixture : IDisposable
{
    private readonly EmbeddingGemma2EmbeddingGenerator? _generator;

    public EmbeddingGemma2GeneratorFixture()
    {
        Rows = EmbeddingGemma2Fixtures.LoadRows();
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        if (files is null)
        {
            SkipReason = reason;
            return;
        }

        _generator = EmbeddingGemma2EmbeddingGenerator.Create(files);
    }

    public string? SkipReason { get; }

    public IReadOnlyList<EmbeddingGemma2Row> Rows { get; }

    public EmbeddingGemma2EmbeddingGenerator Generator
    {
        get
        {
            Assert.SkipWhen(_generator is null, SkipReason ?? "model unavailable");
            return _generator!;
        }
    }

    public void Dispose() => _generator?.Dispose();
}

/// <summary>
/// The adapter on the real graph: embeddings equal <see cref="EmbeddingGemma2TextEmbedder"/> exactly and meet the
/// protected binding tolerance of <c>onnx-fp32-cpu</c>, per prompt, and per Matryoshka size. Skips when the graph is absent.
/// </summary>
public class RealModelTests(EmbeddingGemma2GeneratorFixture fixture, ITestOutputHelper output) : IClassFixture<EmbeddingGemma2GeneratorFixture>
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("onnx-fp32-cpu");

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void MetadataReportsThePinnedModel()
    {
        var generator = fixture.Generator;
        var metadata = ((IEmbeddingGenerator)generator).GetService<EmbeddingGeneratorMetadata>()!;
        Assert.Equal($"{EmbeddingGemma2ModelAssets.Repository}@{EmbeddingGemma2ModelAssets.Revision}", metadata.DefaultModelId);
        Assert.Equal(768, metadata.DefaultModelDimensions);
        Assert.Equal(EmbeddingGemma2ModelAssets.ModelSha256, generator.Embedder.Encoder.Session.Diagnostics.ModelSha256);
    }

    [Theory]
    [InlineData("none")]
    [InlineData("query")]
    [InlineData("document")]
    public async Task EmbeddingsPerPromptEqualTheTextEmbedderAndMeetTheProtectedTolerance(string prompt)
    {
        var generator = fixture.Generator;
        var rows = fixture.Rows.Where(r => r.Prompt == prompt && r.InputIds.Length < 400).ToList();
        var texts = rows.Select(r => r.Text).ToList();
        var options = new EmbeddingGenerationOptions { AdditionalProperties = new() { [EmbeddingGemma2EmbeddingGenerator.PromptPropertyName] = prompt } };
        var result = await generator.GenerateAsync(texts, options, Ct);
        var direct = generator.Embedder.EmbedBatch(texts, EmbeddingGemma2Prompt.FromName(prompt), Ct);
        var stats = new EmbeddingGemma2ErrorStats();
        for (int i = 0; i < rows.Count; i++)
        {
            Assert.Equal(direct[i], result[i].Vector.ToArray());
            stats.Add(rows[i].Key, result[i].Vector.Span, rows[i].Embedding);
        }

        Assert.Equal(rows.Sum(r => (long)r.InputIds.Length), result.Usage!.InputTokenCount);
        output.WriteLine($"prompt {prompt}, {rows.Count} rows through IEmbeddingGenerator vs protected: {stats}");
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs, $"{stats.Max:E3}");
        Assert.True(stats.MinCosine >= Tolerances.MinCosine);
    }

    [Fact]
    public async Task MicroBatchedSingleCallsMeetThePaddedBatchTolerance()
    {
        // Experimental micro-batching over the fixture's encoder: query rows as concurrent one-value calls, coalesced into
        // mixed-length padded batches of up to 8 rows. Each must equal a single call within the protected
        // padded_batch_vs_single bound and the protected embedding within embedding_max_abs.
        var embedder = fixture.Generator.Embedder;
        using var generator = new EmbeddingGemma2EmbeddingGenerator(embedder, new EmbeddingGemma2EmbeddingGeneratorOptions
        {
            Prompt = EmbeddingGemma2Prompt.Query,
            MicroBatching = new() { MaxRows = 8, MaxWait = TimeSpan.FromMilliseconds(20) },
        });
        var rows = fixture.Rows.Where(r => r.Prompt == "query" && r.InputIds.Length < 400).Take(24).ToList();
        var calls = rows.Select(r => Task.Run(() => generator.GenerateVectorAsync(r.Text, cancellationToken: Ct), Ct)).ToArray();
        var vectors = await Task.WhenAll(calls);

        var vsSingle = new EmbeddingGemma2ErrorStats();
        var vsProtected = new EmbeddingGemma2ErrorStats();
        for (int i = 0; i < rows.Count; i++)
        {
            vsSingle.Add(rows[i].Key, vectors[i].Span, embedder.Embed(rows[i].Text, EmbeddingGemma2Prompt.Query, Ct));
            vsProtected.Add(rows[i].Key, vectors[i].Span, rows[i].Embedding);
        }

        var stats = generator.MicroBatchingStatistics!.Value;
        output.WriteLine($"micro-batched, {rows.Count} concurrent calls in {stats.Batches} batches (padding efficiency {(double)stats.Tokens / stats.TokenCells:P1}); vs single: {vsSingle}; vs protected: {vsProtected}");
        Assert.Equal(rows.Count, stats.Items);
        Assert.False(vsSingle.NonFinite);
        Assert.True(vsSingle.Max <= Tolerances.PaddedBatchVsSingleMaxAbs, $"micro-batched vs single {vsSingle.Max:E3}");
        Assert.True(vsProtected.Max <= Tolerances.EmbeddingMaxAbs, $"micro-batched vs protected {vsProtected.Max:E3}");
    }

    [Theory]
    [InlineData(512)]
    [InlineData(256)]
    [InlineData(128)]
    public async Task DimensionsGiveTheTruncatedReference(int dimensions)
    {
        var rows = fixture.Rows.Where(r => r.Prompt == "query" && r.InputIds.Length < 200).Take(12).ToList();
        var options = new EmbeddingGenerationOptions
        {
            Dimensions = dimensions,
            AdditionalProperties = new() { [EmbeddingGemma2EmbeddingGenerator.PromptPropertyName] = EmbeddingGemma2Prompt.Query },
        };
        var result = await fixture.Generator.GenerateAsync(rows.Select(r => r.Text), options, Ct);
        var stats = new EmbeddingGemma2ErrorStats();
        for (int i = 0; i < rows.Count; i++)
        {
            Assert.Equal(dimensions, result[i].Dimensions);
            stats.Add(rows[i].Key, result[i].Vector.Span, EmbeddingGemma2Encoder.Truncate(rows[i].Embedding, dimensions));
        }

        output.WriteLine($"Dimensions = {dimensions} vs sliced+renormalized reference: {stats}");
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs, $"{stats.Max:E3}");
    }
}
