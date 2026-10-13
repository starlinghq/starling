// The tests call the generator through each IEmbeddingGenerator interface on purpose.
#pragma warning disable CA1859

using Starlings.TestSupport;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

namespace Starlings.Models.EmbeddingGemma2.Extensions.AI.Tests;

/// <summary>
/// The image generator's contract on the weight-free graphs (FakeMultimodal): both input shapes, the decoder hook,
/// rejected content, options, usage and DI. Runs in CI.
/// </summary>
public sealed class MultimodalGeneratorContractTests : IDisposable
{
    private static readonly byte[] PngSignature = [0x89, 0x50, 0x4E, 0x47];
    private static readonly int[] Widths = [61, 62, 63];

    private readonly EmbeddingGemma2MultimodalEmbedder _embedder = FakeMultimodal.Create();
    private readonly List<(int Length, string MediaType)> _decoded = [];

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _embedder.Dispose();

    /// <summary>A stand-in decoder: the first byte selects a synthetic image.</summary>
    private EmbeddingGemma2Image Decode(ReadOnlyMemory<byte> bytes, string mediaType)
    {
        lock (_decoded)
        {
            _decoded.Add((bytes.Length, mediaType));
        }

        return FakeMultimodal.Image(60 + bytes.Span[^1], 40, bytes.Span[^1]);
    }

    private EmbeddingGemma2MultimodalEmbeddingGenerator Generator(EmbeddingGemma2MultimodalEmbeddingGeneratorOptions? options = null) =>
        new(_embedder, (options ?? new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions()) with { ImageDecoder = options?.ImageDecoder ?? Decode });

    private static DataContent Image(byte seed, string mediaType = "image/png") => new(new ReadOnlyMemory<byte>([.. PngSignature, seed]), mediaType);

    [Fact]
    public async Task ImagesEqualTheEmbedderAndKeepOrder()
    {
        using var generator = Generator();
        IEmbeddingGenerator<DataContent, Embedding<float>> images = generator;
        var result = await images.GenerateAsync([Image(1), Image(2), Image(3, "image/jpeg")], cancellationToken: Ct);
        Assert.Equal(3, result.Count);
        for (int i = 0; i < 3; i++)
        {
            var expected = _embedder.Embed(FakeMultimodal.Image(61 + i, 40, (byte)(i + 1)), Ct);
            Assert.Equal(expected, result[i].Vector.ToArray());
            Assert.Equal(EmbeddingGemma2EmbeddingGeneratorOptions.PinnedModelId, result[i].ModelId);
        }

        Assert.Equal("image/jpeg", _decoded[2].MediaType);
        long tokens = Widths.Sum(w => (long)_embedder.Tokenize(EmbeddingGemma2MultimodalInput.FromImage(FakeMultimodal.Image(w, 40, 1))).Length);
        Assert.Equal(tokens, result.Usage!.InputTokenCount);
    }

    [Fact]
    public async Task InterleavedContentEqualsTheEmbedder()
    {
        using var generator = Generator();
        IEmbeddingGenerator<IEnumerable<AIContent>, Embedding<float>> contents = generator;
        var result = await contents.GenerateAsync(
            [
                [new TextContent("A photo: "), Image(5)],
                [new TextContent("text only")],
                [Image(6), new TextContent(" next to "), Image(7)],
            ],
            cancellationToken: Ct);
        Assert.Equal(_embedder.Embed(EmbeddingGemma2MultimodalInput.Create("A photo: ", FakeMultimodal.Image(65, 40, 5)), Ct), result[0].Vector.ToArray());
        Assert.Equal(_embedder.Embed(EmbeddingGemma2MultimodalInput.FromText("text only"), Ct), result[1].Vector.ToArray());
        Assert.Equal(_embedder.Embed(EmbeddingGemma2MultimodalInput.Create(FakeMultimodal.Image(66, 40, 6), " next to ", FakeMultimodal.Image(67, 40, 7)), Ct), result[2].Vector.ToArray());
    }

    [Fact]
    public async Task UnsupportedContentIsRejectedNotIgnored()
    {
        using var generator = Generator();
        IEmbeddingGenerator<DataContent, Embedding<float>> images = generator;
        IEmbeddingGenerator<IEnumerable<AIContent>, Embedding<float>> contents = generator;
        await Assert.ThrowsAsync<NotSupportedException>(() => images.GenerateAsync([new DataContent(new byte[] { 1, 2 }, "audio/wav")], cancellationToken: Ct));
        await Assert.ThrowsAsync<NotSupportedException>(() => images.GenerateAsync([new DataContent(new byte[] { 1, 2 }, "video/mp4")], cancellationToken: Ct));
        await Assert.ThrowsAsync<NotSupportedException>(() => contents.GenerateAsync([[new UriContent("https://example.com/cat.png", "image/png")]], cancellationToken: Ct));
        await Assert.ThrowsAsync<NotSupportedException>(() => contents.GenerateAsync([[new TextReasoningContent("x")]], cancellationToken: Ct));
        await Assert.ThrowsAsync<ArgumentException>(() => contents.GenerateAsync([[new TextContent("caption <|image|>"), Image(1)]], cancellationToken: Ct));

        // No decoder: images are rejected with a pointer to the SkiaSharp adapter.
        using var noDecoder = new EmbeddingGemma2MultimodalEmbeddingGenerator(_embedder);
        var ex = await Assert.ThrowsAsync<NotSupportedException>(() => ((IEmbeddingGenerator<DataContent, Embedding<float>>)noDecoder).GenerateAsync([Image(1)], cancellationToken: Ct));
        Assert.Contains("ImageDecoder", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>A stand-in video decoder: the last byte selects the frame count of a synthetic 64x48 video.</summary>
    private static EmbeddingGemma2Video DecodeVideo(ReadOnlyMemory<byte> bytes, string mediaType) =>
        EmbeddingGemma2Video.FromSampledFrames(Enumerable.Range(0, bytes.Span[^1]).Select(i => FakeMultimodal.Image(64, 48, 100 + i)).ToArray());

    [Fact]
    public async Task VideosUseTheVideoDecoderAndEqualTheEmbedder()
    {
        using var generator = Generator(new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions { VideoDecoder = DecodeVideo });
        IEmbeddingGenerator<DataContent, Embedding<float>> media = generator;
        IEmbeddingGenerator<IEnumerable<AIContent>, Embedding<float>> contents = generator;
        var clip = new DataContent(new byte[] { 0, 0, 0, 3 }, "video/mp4");
        var result = await media.GenerateAsync([clip, Image(2)], cancellationToken: Ct);
        var video = DecodeVideo(clip.Data, clip.MediaType);
        Assert.Equal(_embedder.Embed(video, Ct), result[0].Vector.ToArray());
        Assert.Equal(_embedder.Embed(FakeMultimodal.Image(62, 40, 2), Ct), result[1].Vector.ToArray());
        Assert.Equal((long)_embedder.Tokenize(EmbeddingGemma2MultimodalInput.FromVideo(video)).Length + _embedder.Tokenize(EmbeddingGemma2MultimodalInput.FromImage(FakeMultimodal.Image(62, 40, 2))).Length, result.Usage!.InputTokenCount);

        var mixed = await contents.GenerateAsync([[new TextContent("clip: "), new DataContent(new byte[] { 2 }, "video/webm"), new TextContent(" still: "), Image(5)]], cancellationToken: Ct);
        var expected = _embedder.Embed(EmbeddingGemma2MultimodalInput.Create("clip: ", DecodeVideo(new byte[] { 2 }, "video/webm"), " still: ", FakeMultimodal.Image(65, 40, 5)), Ct);
        Assert.Equal(expected, mixed[0].Vector.ToArray());

        // Without a video decoder, video values are rejected with a pointer to the FFmpeg adapter; frames without timing need the opt-in.
        var ex = await Assert.ThrowsAsync<NotSupportedException>(() => ((IEmbeddingGenerator<DataContent, Embedding<float>>)Generator()).GenerateAsync([clip], cancellationToken: Ct));
        Assert.Contains("VideoDecoder", ex.Message, StringComparison.Ordinal);
        using var untimed = Generator(new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions { VideoDecoder = (b, m) => EmbeddingGemma2Video.FromFramesWithoutTiming(DecodeVideo(b, m).Frames) });
        await Assert.ThrowsAsync<EmbeddingGemma2VideoTimingRequiredException>(() => ((IEmbeddingGenerator<DataContent, Embedding<float>>)untimed).GenerateAsync([clip], cancellationToken: Ct));
        await Assert.ThrowsAsync<ArgumentException>(() => contents.GenerateAsync([[new TextContent("a <|video|>"), clip]], cancellationToken: Ct));
    }

    [Fact]
    public async Task OptionsSelectDimensionsAndPromptAndRejectTheRest()
    {
        using var generator = Generator();
        IEmbeddingGenerator<DataContent, Embedding<float>> images = generator;
        var full = (await images.GenerateAsync([Image(4)], cancellationToken: Ct))[0].Vector.ToArray();
        var small = (await images.GenerateAsync([Image(4)], new EmbeddingGenerationOptions { Dimensions = 256 }, Ct))[0].Vector.ToArray();
        Assert.Equal(EmbeddingGemma2Encoder.Truncate(full, 256), small);
        await Assert.ThrowsAsync<NotSupportedException>(() => images.GenerateAsync([Image(4)], new EmbeddingGenerationOptions { Dimensions = 300 }, Ct));
        await Assert.ThrowsAsync<NotSupportedException>(() => images.GenerateAsync([Image(4)], new EmbeddingGenerationOptions { ModelId = "other" }, Ct));
        await Assert.ThrowsAsync<NotSupportedException>(() => images.GenerateAsync([Image(4)], new EmbeddingGenerationOptions { AdditionalProperties = new() { ["x"] = 1 } }, Ct));

        var prompted = (await images.GenerateAsync([Image(4)], new EmbeddingGenerationOptions { AdditionalProperties = new() { [EmbeddingGemma2EmbeddingGenerator.PromptPropertyName] = "query" } }, Ct))[0].Vector.ToArray();
        Assert.Equal(_embedder.Embed(EmbeddingGemma2MultimodalInput.FromImage(FakeMultimodal.Image(64, 40, 4)), EmbeddingGemma2Prompt.Query, Ct), prompted);
    }

    [Fact]
    public async Task CancellationFaultsTheTaskAndServicesResolve()
    {
        using var generator = Generator();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var task = ((IEmbeddingGenerator<DataContent, Embedding<float>>)generator).GenerateAsync([Image(1)], cancellationToken: cts.Token);
        Assert.True(task.IsCanceled);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);

        Assert.Same(generator.Metadata, generator.GetService(typeof(EmbeddingGeneratorMetadata)));
        Assert.Same(_embedder, generator.GetService(typeof(EmbeddingGemma2MultimodalEmbedder)));
        Assert.Same(generator, generator.GetService(typeof(IEmbeddingGenerator<DataContent, Embedding<float>>)));
        Assert.Same(generator, generator.GetService(typeof(IEmbeddingGenerator<IEnumerable<AIContent>, Embedding<float>>)));
        Assert.Null(generator.GetService(typeof(IEmbeddingGenerator<string, Embedding<float>>)));
    }

    [Fact]
    public void TheGeneratorRegistersBothShapes()
    {
        var services = new ServiceCollection();
        services.AddSingleton(_ => Generator());
        services.AddSingleton<IEmbeddingGenerator<DataContent, Embedding<float>>>(sp => sp.GetRequiredService<EmbeddingGemma2MultimodalEmbeddingGenerator>());
        using var provider = services.BuildServiceProvider();
        Assert.IsType<EmbeddingGemma2MultimodalEmbeddingGenerator>(provider.GetRequiredService<IEmbeddingGenerator<DataContent, Embedding<float>>>());
    }
}
