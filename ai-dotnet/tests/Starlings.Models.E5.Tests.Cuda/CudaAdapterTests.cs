using Starlings.Models.E5.Extensions.AI;
using Starlings.Models.E5.MLNet;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.E5.Tests.Cuda;

public sealed class CudaAdapterRow
{
    public string Id { get; set; } = string.Empty;

    public string Text { get; set; } = string.Empty;
}

public sealed class CudaAdapterEmbedding
{
    public string Id { get; set; } = string.Empty;

    [VectorType(384)]
    public float[] Embedding { get; set; } = [];
}

/// <summary>
/// Profile <c>multilingual-e5-small/onnx-fp32-cuda</c>, capabilities <c>extensions_ai</c> and <c>mlnet</c>: the
/// Microsoft.Extensions.AI generator and the ML.NET transform, each created from the verified files with a CUDA session
/// (<c>use_tf32=0</c>), on the 46 protected texts: equal to the text embedder over the same session, within the CUDA
/// profile's protected tolerances. Opt-in like the other CUDA tests.
/// </summary>
public class CudaAdapterTests(ITestOutputHelper output)
{
    private static readonly E5Tolerances Tolerances = E5Tolerances.Load(E5TestAssets.CudaFixtureDirectory);

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static OnnxSessionOptions Cuda => new() { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 };

    private static E5ModelFiles Files()
    {
        string? gate = CudaGate.Check();
        Assert.SkipWhen(gate is not null, gate ?? string.Empty);
        var (files, reason) = E5TestAssets.FindFiles();
        Assert.SkipWhen(files is null, reason);
        return files!;
    }

    private static void AssertCuda(OnnxModelSession session)
    {
        var d = session.Diagnostics;
        Assert.Equal(OnnxRuntimeInfo.CudaProviderName, d.SessionProviders[0]);
        Assert.Equal("0", d.ProviderOptions["use_tf32"]);
        Assert.Equal(E5TestAssets.ModelSha256, d.ModelSha256);
    }

    [Fact]
    public async Task ExtensionsAIGeneratorOverCudaMeetsTheProtectedTolerance()
    {
        using var generator = E5EmbeddingGenerator.Create(Files(), new E5EmbeddingGeneratorOptions { Encoder = new E5EncoderOptions { Session = Cuda } });
        AssertCuda(generator.Embedder.Encoder.Session);
        var cases = E5Fixtures.Cases;
        var texts = cases.Select(c => E5Fixtures.Texts[c.Id]).ToList();
        var batch = await generator.GenerateAsync(texts, cancellationToken: Ct);
        var direct = generator.Embedder.EmbedBatch(texts, Ct);
        var vsProtected = new ErrorStats();
        var vsSingle = new ErrorStats();
        for (int i = 0; i < cases.Count; i++)
        {
            Assert.Equal(direct[i], batch[i].Vector.ToArray());
            vsProtected.Add(batch[i].Vector.Span, cases[i].Embedding);
            var single = await generator.GenerateVectorAsync(texts[i], cancellationToken: Ct);
            Assert.Equal(generator.Embedder.Embed(texts[i], Ct), single.ToArray());
            vsSingle.Add(batch[i].Vector.Span, single.Span);
        }

        output.WriteLine($"E5EmbeddingGenerator over CUDA, {cases.Count} texts in one call vs protected: {vsProtected}");
        output.WriteLine($"batch vs one text per call: {vsSingle}; bounds from {Tolerances.Source}");
        Assert.False(vsProtected.NonFinite);
        Assert.True(vsProtected.Max <= Tolerances.EmbeddingMaxAbs, $"vs protected {vsProtected.Max:E3}");
        Assert.True(vsSingle.Max <= Tolerances.PaddedBatchVsSingleMaxAbs, $"batch vs single {vsSingle.Max:E3}");
    }

    [Fact]
    public void MLNetTransformOverCudaMeetsTheProtectedTolerance()
    {
        var files = Files();
        var ml = new MLContext(seed: 1);
        var view = ml.Data.LoadFromEnumerable(E5Fixtures.Cases.Select(c => new CudaAdapterRow { Id = c.Id, Text = E5Fixtures.Texts[c.Id] }).ToList());
        foreach (int batchSize in new[] { 1, 16 })
        {
            using var transformer = ml.Transforms.ApplyE5Embedding(files, "Embedding", "Text", new E5EmbeddingOptions { BatchSize = batchSize, Encoder = new E5EncoderOptions { Session = Cuda } }).Fit(view);
            AssertCuda(transformer.Embedder.Encoder.Session);
            var rows = ml.Data.CreateEnumerable<CudaAdapterEmbedding>(transformer.Transform(view), reuseRowObject: false).ToList();
            Assert.Equal(46, rows.Count);
            var vsProtected = new ErrorStats();
            for (int i = 0; i < rows.Count; i++)
            {
                var c = E5Fixtures.Cases[i];
                Assert.Equal(c.Id, rows[i].Id);
                vsProtected.Add(rows[i].Embedding, c.Embedding);
            }

            output.WriteLine($"ML.NET transform over CUDA, batch size {batchSize}, 46 texts vs protected: {vsProtected}");
            Assert.False(vsProtected.NonFinite);
            Assert.True(vsProtected.Max <= Tolerances.EmbeddingMaxAbs, $"batch size {batchSize}: vs protected {vsProtected.Max:E3}");
        }
    }
}
