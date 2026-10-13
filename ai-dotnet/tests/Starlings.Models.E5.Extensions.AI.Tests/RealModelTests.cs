using System.Diagnostics;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

namespace Starlings.Models.E5.Extensions.AI.Tests;

/// <summary>
/// One generator over the real e5 snapshot per test class, from <see cref="E5EmbeddingGenerator.Create"/> (model and
/// tokenizer digests enforced), or a skip reason when the snapshot is absent.
/// </summary>
public sealed class E5GeneratorFixture : IDisposable
{
    private readonly E5EmbeddingGenerator? _generator;

    public E5GeneratorFixture()
    {
        var (files, reason) = E5TestAssets.FindFiles();
        if (files is null)
        {
            SkipReason = reason;
            return;
        }

        Files = files;
        _generator = E5EmbeddingGenerator.Create(files);
    }

    public string? SkipReason { get; }

    public E5ModelFiles? Files { get; }

    public E5EmbeddingGenerator Generator
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
/// The adapter on the real model: embeddings equal <see cref="E5TextEmbedder"/> exactly and meet the protected
/// binding tolerance (<c>model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/tolerances.yaml</c>, read only);
/// metadata, cancellation during a run, owned disposal and DI. Skips when the snapshot is absent.
/// </summary>
public class RealModelTests(E5GeneratorFixture fixture, ITestOutputHelper output) : IClassFixture<E5GeneratorFixture>
{
    private static readonly E5Tolerances Tolerances = E5Tolerances.Load(E5TestAssets.CpuFixtureDirectory);

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void MetadataReportsThePinnedModel()
    {
        var generator = fixture.Generator;
        var metadata = ((IEmbeddingGenerator)generator).GetService<EmbeddingGeneratorMetadata>()!;
        Assert.Equal(E5EmbeddingGenerator.ProviderName, metadata.ProviderName);
        Assert.Equal($"{E5ModelAssets.Repository}@{E5ModelAssets.Revision}", metadata.DefaultModelId);
        Assert.Equal(384, metadata.DefaultModelDimensions);
        Assert.Equal(E5TestAssets.ModelSha256, generator.Embedder.Encoder.Session.Diagnostics.ModelSha256);
    }

    [Fact]
    public async Task EmbeddingsEqualTheTextEmbedderAndMeetTheProtectedTolerance()
    {
        var generator = fixture.Generator;
        var cases = E5Fixtures.Cases;
        var texts = cases.Select(c => E5Fixtures.Texts[c.Id]).ToList();

        // All 46 texts in one call: two ONNX Runtime chunks of MaxBatchSize = 32 and 14 rows.
        var batch = await generator.GenerateAsync(texts, cancellationToken: Ct);
        var direct = generator.Embedder.EmbedBatch(texts, Ct);
        var vsProtected = new ErrorStats();
        var vsSingle = new ErrorStats();
        long tokens = 0;
        for (int i = 0; i < cases.Count; i++)
        {
            Assert.Equal(direct[i], batch[i].Vector.ToArray());
            Assert.Equal(384, batch[i].Dimensions);
            vsProtected.Add(batch[i].Vector.Span, cases[i].Embedding);

            // One value per call equals E5TextEmbedder.Embed exactly.
            var single = await generator.GenerateVectorAsync(texts[i], cancellationToken: Ct);
            Assert.Equal(generator.Embedder.Embed(texts[i], Ct), single.ToArray());
            vsSingle.Add(batch[i].Vector.Span, single.Span);
            tokens += cases[i].InputIds.Length;
        }

        output.WriteLine($"adapter, {cases.Count} texts in one call vs protected: {vsProtected}");
        output.WriteLine($"adapter batch vs one text per call: {vsSingle}");
        output.WriteLine($"bounds: embedding_max_abs {Tolerances.EmbeddingMaxAbs:E1}, padded_batch_vs_single {Tolerances.PaddedBatchVsSingleMaxAbs:E1} from {Tolerances.Source}");
        Assert.False(vsProtected.NonFinite);
        Assert.True(vsProtected.Max <= Tolerances.EmbeddingMaxAbs, $"adapter vs protected {vsProtected.Max:E3}");
        Assert.True(vsSingle.Max <= Tolerances.PaddedBatchVsSingleMaxAbs, $"adapter batch vs single {vsSingle.Max:E3}");

        // Usage counts the tokens fed to the model: the protected (truncated) ids.
        Assert.Equal(tokens, batch.Usage!.InputTokenCount);
    }

    [Fact]
    public async Task CancellationInterruptsARunAndTheGeneratorStaysUsable()
    {
        var generator = fixture.Generator;
        string longText = E5Fixtures.Texts["long-english"];
        var texts = Enumerable.Repeat(longText, 32).ToList();

        using var cts = new CancellationTokenSource();
        var sw = Stopwatch.StartNew();
        TimeSpan cancelledAt = TimeSpan.Zero;
        cts.Token.Register(() => cancelledAt = sw.Elapsed);
        cts.CancelAfter(TimeSpan.FromMilliseconds(200));

        // Synchronous: the call blocks this thread until ONNX Runtime stops at the next node boundary.
        var task = generator.GenerateAsync(texts, cancellationToken: cts.Token);
        var latency = sw.Elapsed - cancelledAt;
        Assert.True(task.IsCanceled, $"task status {task.Status}");
        output.WriteLine($"32 x 512 tokens: cancelled at {cancelledAt.TotalMilliseconds:F0} ms, task returned {latency.TotalMilliseconds:F1} ms later");
        Assert.True(latency < TimeSpan.FromSeconds(2), $"cancellation latency {latency.TotalMilliseconds:F0} ms");

        var empty = E5Fixtures.Cases[0];
        var after = await generator.GenerateVectorAsync(E5Fixtures.Texts[empty.Id], cancellationToken: Ct);
        var stats = new ErrorStats();
        stats.Add(after.Span, empty.Embedding);
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs, $"after cancellation: {stats}");
    }

    [Fact]
    public async Task CreatedGeneratorOwnsAndDisposesItsEncoderAndDependencyInjectionDisposesIt()
    {
        _ = fixture.Generator;
        var files = fixture.Files!;

        var services = new ServiceCollection();
        services.AddE5EmbeddingGenerator(files, new E5EmbeddingGeneratorOptions { InputType = E5InputType.Query });
        E5Encoder encoder;
        await using (var provider = services.BuildServiceProvider())
        {
            var contract = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
            encoder = contract.GetService<E5Encoder>()!;
            var viaDi = await contract.GenerateVectorAsync("how much protein should a female eat", cancellationToken: Ct);
            var explicitPrefix = await fixture.Generator.GenerateVectorAsync("query: how much protein should a female eat", cancellationToken: Ct);
            Assert.Equal(explicitPrefix.ToArray(), viaDi.ToArray());
        }

        Assert.Throws<ObjectDisposedException>(() => encoder.Embed(new EncoderInput(new long[] { 0, 2 }), Ct));
    }
}
