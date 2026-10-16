using Starlings.Models.EmbeddingGemma2.Extensions.AI;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using Starlings.Models.EmbeddingGemma2.MLNet;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.EmbeddingGemma2.Tests.Cuda;

public sealed class CudaAdapterRow
{
    public string Id { get; set; } = string.Empty;

    public string Text { get; set; } = string.Empty;
}

public sealed class CudaAdapterEmbedding
{
    public string Id { get; set; } = string.Empty;

    [VectorType(768)]
    public float[] Embedding { get; set; } = [];
}

/// <summary>
/// Capabilities <c>extensions_ai</c> and <c>mlnet</c> of <c>embeddinggemma-2/onnx-fp32-cuda</c> and <c>extensions_ai</c>
/// of <c>image-fp32-cuda</c> and <c>video-fp32-cuda</c>: the Microsoft.Extensions.AI generators and the ML.NET transform,
/// each created from the verified files with CUDA sessions (<c>use_tf32=0</c>), against the protected references with
/// each profile's protected tolerances. Opt-in like the other CUDA tests.
/// </summary>
[Collection("cuda")]
public class CudaAdapterTests(ITestOutputHelper output)
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static OnnxSessionOptions Cuda => new() { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 };

    private static EmbeddingGemma2ModelFiles TextFiles()
    {
        string? gate = CudaGate.Check();
        Assert.SkipWhen(gate is not null, gate ?? string.Empty);
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        return files!;
    }

    private static EmbeddingGemma2ImageModelFiles ImageFiles()
    {
        string? gate = CudaGate.Check();
        Assert.SkipWhen(gate is not null, gate ?? string.Empty);
        var (files, reason) = EmbeddingGemma2ImageTestAssets.FindImageModel();
        Assert.SkipWhen(files is null, reason);
        return files!;
    }

    private static void AssertCuda(OnnxModelSession session)
    {
        Assert.Equal(OnnxRuntimeInfo.CudaProviderName, session.Diagnostics.SessionProviders[0]);
        Assert.Equal("0", session.Diagnostics.ProviderOptions["use_tf32"]);
    }

    [Fact]
    public async Task TextGeneratorOverCudaMeetsTheProtectedTolerance()
    {
        var tolerances = EmbeddingGemma2Tolerances.Load("onnx-fp32-cuda");
        using var generator = EmbeddingGemma2EmbeddingGenerator.Create(TextFiles(), new EmbeddingGemma2EmbeddingGeneratorOptions { Encoder = new EmbeddingGemma2EncoderOptions { Session = Cuda } });
        AssertCuda(generator.Embedder.Encoder.Session);
        var all = new EmbeddingGemma2ErrorStats();
        foreach (string prompt in new[] { "none", "query", "document" })
        {
            var rows = EmbeddingGemma2Fixtures.LoadRows().Where(r => r.Prompt == prompt && r.InputIds.Length < 400).ToList();
            var options = new EmbeddingGenerationOptions { AdditionalProperties = new() { [EmbeddingGemma2EmbeddingGenerator.PromptPropertyName] = prompt } };
            var result = await generator.GenerateAsync(rows.Select(r => r.Text).ToList(), options, Ct);
            var direct = generator.Embedder.EmbedBatch(rows.Select(r => r.Text).ToList(), EmbeddingGemma2Prompt.FromName(prompt), Ct);
            for (int i = 0; i < rows.Count; i++)
            {
                Assert.Equal(direct[i], result[i].Vector.ToArray());
                all.Add(rows[i].Key, result[i].Vector.Span, rows[i].Embedding);
            }
        }

        output.WriteLine($"EmbeddingGemma2EmbeddingGenerator over CUDA, prompts none/query/document, vs protected: {all}; bounds from {tolerances.Source}");
        Assert.False(all.NonFinite);
        Assert.True(all.Max <= tolerances.EmbeddingMaxAbs, $"{all.Max:E3}");
        Assert.True(all.MinCosine >= tolerances.MinCosine);
    }

    [Fact]
    public void MLNetTransformOverCudaMeetsTheProtectedTolerance()
    {
        var tolerances = EmbeddingGemma2Tolerances.Load("onnx-fp32-cuda");
        var files = TextFiles();
        var rows = EmbeddingGemma2Fixtures.LoadRows().Where(r => r.Prompt == "document" && r.InputIds.Length < 400).ToList();
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(rows.Select(r => new CudaAdapterRow { Id = r.Key, Text = r.Text }).ToList());
        using var transformer = ml.Transforms.ApplyEmbeddingGemma2Embedding(
            files,
            "Embedding",
            "Text",
            new EmbeddingGemma2EmbeddingOptions { Prompt = EmbeddingGemma2Prompt.Document, BatchSize = 4, Encoder = new EmbeddingGemma2EncoderOptions { Session = Cuda } }).Fit(data);
        AssertCuda(transformer.Embedder.Encoder.Session);
        var embedded = ml.Data.CreateEnumerable<CudaAdapterEmbedding>(transformer.Transform(data), reuseRowObject: false).ToList();
        var stats = new EmbeddingGemma2ErrorStats();
        for (int i = 0; i < rows.Count; i++)
        {
            Assert.Equal(rows[i].Key, embedded[i].Id);
            stats.Add(rows[i].Key, embedded[i].Embedding, rows[i].Embedding);
        }

        output.WriteLine($"ML.NET transform over CUDA, {rows.Count} document rows in look-ahead batches of 4 vs protected: {stats}");
        Assert.False(stats.NonFinite);
        Assert.True(stats.Max <= tolerances.EmbeddingMaxAbs, $"{stats.Max:E3}");
        Assert.True(stats.MinCosine >= tolerances.MinCosine);
    }

    [Fact]
    public async Task MultimodalGeneratorOverCudaMeetsTheProtectedImageAndVideoTolerances()
    {
        var files = ImageFiles();
        Assert.SkipWhen(EmbeddingGemma2ImageTestAssets.ImagesSkipReason is not null, EmbeddingGemma2ImageTestAssets.ImagesSkipReason ?? string.Empty);
        var videoDecoder = EmbeddingGemma2VideoTestAssets.Decoder; // skips without the video set or FFmpeg
        using var generator = EmbeddingGemma2MultimodalEmbeddingGenerator.Create(files, new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions
        {
            Embedder = new EmbeddingGemma2MultimodalOptions { Session = Cuda },
            ImageDecoder = (bytes, _) => EmbeddingGemma2ImageDecoder.Decode(bytes.Span),
            VideoDecoder = (bytes, _) => videoDecoder.Decode(bytes, Ct).Video,
        });
        AssertCuda(generator.Embedder.VisionSession);
        AssertCuda(generator.Embedder.MultimodalSession);

        // Images: every image-only row without a prompt.
        var imageTolerances = EmbeddingGemma2Tolerances.Load("image-fp32-cuda");
        var imageRows = EmbeddingGemma2ImageTestAssets.Rows.Where(r => r.InContract && r.Kind == "image" && r.Prompt == "none" && r.Images.Count == 1).ToList();
        var imageContents = imageRows.Select(r => new DataContent(EmbeddingGemma2ImageTestAssets.ReadImageFile(r.Images[0].Image), "image/png")).ToList();
        var images = await ((IEmbeddingGenerator<DataContent, Embedding<float>>)generator).GenerateAsync(imageContents, cancellationToken: Ct);
        var imageStats = new EmbeddingGemma2ErrorStats();
        for (int i = 0; i < imageRows.Count; i++)
        {
            imageStats.Add(imageRows[i].Id, images[i].Vector.Span, imageRows[i].Embedding);
        }

        output.WriteLine($"multimodal generator over CUDA, {imageRows.Count} images vs protected: {imageStats}; bounds from {imageTolerances.Source}");
        Assert.False(imageStats.NonFinite);
        Assert.True(imageStats.Max <= imageTolerances.EmbeddingMaxAbs, $"images {imageStats.Max:E3} ({imageStats.WorstKey})");
        Assert.True(imageStats.MinCosine >= imageTolerances.MinCosine);

        // Videos: the video-only rows from a file path without a prompt (decoded from the file's bytes).
        var videoTolerances = EmbeddingGemma2Tolerances.Load("video-fp32-cuda");
        var videoRows = EmbeddingGemma2VideoTestAssets.Rows.Where(r => r.Kind == "video" && r.InputForm == "path" && r.Prompt == "none").Take(6).ToList();
        var videoContents = videoRows.Select(r => new DataContent(File.ReadAllBytes(EmbeddingGemma2VideoTestAssets.PathOf(r.Videos[0].Video)), "video/mp4")).ToList();
        var videos = await ((IEmbeddingGenerator<DataContent, Embedding<float>>)generator).GenerateAsync(videoContents, cancellationToken: Ct);
        var videoStats = new EmbeddingGemma2ErrorStats();
        for (int i = 0; i < videoRows.Count; i++)
        {
            videoStats.Add(videoRows[i].Id, videos[i].Vector.Span, videoRows[i].Embedding);
        }

        output.WriteLine($"multimodal generator over CUDA, {videoRows.Count} videos vs protected: {videoStats}; bounds from {videoTolerances.Source}");
        Assert.False(videoStats.NonFinite);
        Assert.True(videoStats.Max <= videoTolerances.EmbeddingMaxAbs, $"videos {videoStats.Max:E3} ({videoStats.WorstKey})");
        Assert.True(videoStats.MinCosine >= videoTolerances.MinCosine);
    }
}
