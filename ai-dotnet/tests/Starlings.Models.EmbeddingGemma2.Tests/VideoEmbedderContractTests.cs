using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// The video path's plumbing on weight-free graphs (soft tokens = mean of each 3x3 block's patch values; the multimodal
/// graph adds 1000 x the soft token where the mask is set): soft tokens per frame in order at the <c>&lt;|video|&gt;</c>
/// positions, frame batching, the three frame forms, interleaving with images, rejection. Runs in CI; numbers against
/// the real model are in <see cref="VideoParityTests"/>.
/// </summary>
public sealed class VideoEmbedderContractTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static EmbeddingGemma2Image[] Frames(int count, int width, int height, int seed) =>
        Enumerable.Range(0, count).Select(i => FakeMultimodal.Image(width, height, seed + i)).ToArray();

    [Fact]
    public void FrameSoftTokensFillTheVideoPositionsInOrder()
    {
        using var embedder = FakeMultimodal.Create(new EmbeddingGemma2MultimodalOptions { MaxVideoFramesPerVisionCall = 2 });
        var frames = Frames(3, 64, 48, 10);
        var image = FakeMultimodal.Image(40, 30, 99);
        var video = EmbeddingGemma2Video.FromSampledFrames(frames);
        var input = EmbeddingGemma2MultimodalInput.Create("clip ", video, " still ", image);
        var collector = new Collector();
        embedder.EmbedCore([input], EmbeddingGemma2Prompt.None, new float[768], default, collector, Ct);

        // Per frame: the 140-token preprocessing, soft token s = mean of block s.
        var videoSoft = Assert.Single(collector.Video);
        int n = EmbeddingGemma2ImageProcessor.GetSoftTokenCount(64, 48, EmbeddingGemma2MultimodalEmbedder.VideoFrameSoftTokens);
        Assert.Equal(3 * n * 512, videoSoft.Length);
        for (int f = 0; f < 3; f++)
        {
            var p = EmbeddingGemma2ImageProcessor.Preprocess(frames[f], EmbeddingGemma2MultimodalEmbedder.VideoFrameSoftTokens);
            Assert.Equal(n, p.SoftTokenCount);
            for (int s = 0; s < n; s++)
            {
                double mean = 0;
                foreach (float v in p.PixelValues.Span.Slice(s * 9 * 768, 9 * 768))
                {
                    mean += v;
                }

                Assert.Equal(mean / (9 * 768), videoSoft[((f * n) + s) * 512], tolerance: 1e-5);
            }
        }

        // Token embeddings: the k-th <|video|> carries video soft token k; images keep their own tokens.
        var ids = embedder.Tokenize(input).InputIds.ToArray();
        int videoSlot = 0, imageSlot = 0;
        for (int t = 0; t < ids.Length; t++)
        {
            if (ids[t] == EmbeddingGemma2MultimodalEmbedder.VideoTokenId)
            {
                Assert.Equal((ids[t] * 1.0) + 1.0 + (videoSoft[videoSlot * 512] * 1000.0), collector.Tokens[t * 768], tolerance: 0.1);
                videoSlot++;
            }
            else if (ids[t] == EmbeddingGemma2MultimodalEmbedder.ImageTokenId)
            {
                Assert.Equal((ids[t] * 1.0) + 1.0 + (collector.Images[0][imageSlot * 512] * 1000.0), collector.Tokens[t * 768], tolerance: 0.1);
                imageSlot++;
            }
        }

        Assert.Equal(3 * n, videoSlot);
        Assert.Equal(collector.Images[0].Length / 512, imageSlot);
    }

    [Fact]
    public void FrameFormsBatchingAndPaddingGiveTheSameEmbeddings()
    {
        var all = Frames(150, 64, 48, 20); // 5 s at 30 fps: frames 0, 30, 60, 90, 120
        var timed = EmbeddingGemma2Video.FromDecodedFrames(all, 30, 5);
        var sampled = EmbeddingGemma2Video.FromSampledFrames([all[0], all[30], all[60], all[90], all[120]], [0, 30, 60, 90, 120]);
        var inputs = new[]
        {
            EmbeddingGemma2MultimodalInput.FromVideo(timed),
            EmbeddingGemma2MultimodalInput.FromText("text"),
            EmbeddingGemma2MultimodalInput.FromVideo(sampled),
            EmbeddingGemma2MultimodalInput.Create("two ", EmbeddingGemma2Video.FromSampledFrames(Frames(2, 330, 186, 300)), " and ", EmbeddingGemma2Video.FromSampledFrames(Frames(1, 321, 179, 400))),
            EmbeddingGemma2MultimodalInput.Create(FakeMultimodal.Image(30, 20, 7), EmbeddingGemma2Video.FromSampledFrames(Frames(4, 48, 96, 500))),
        };

        using var single = FakeMultimodal.Create();
        var singles = inputs.Select(i => single.Embed(i, Ct)).ToArray();
        Assert.Equal(singles[0], singles[2]); // the package's sampling equals pre-sampled upstream frames

        foreach (var options in new[]
        {
            new EmbeddingGemma2MultimodalOptions { MaxVideoFramesPerVisionCall = 1 },
            new EmbeddingGemma2MultimodalOptions { MaxVideoFramesPerVisionCall = 8, MaxVisionBatchPatches = 8 * 1_260, MaxBatchSize = 3 },
            new EmbeddingGemma2MultimodalOptions { PadImagesToMaxPatches = true, MaxVideoFramesPerVisionCall = 3, MaxVisionBatchPatches = 3 * 2_520 },
        })
        {
            using var batched = FakeMultimodal.Create(options);
            var batch = batched.EmbedBatch(inputs, Ct);
            for (int i = 0; i < inputs.Length; i++)
            {
                for (int d = 0; d < 768; d++)
                {
                    Assert.Equal(singles[i][d], batch[i][d], tolerance: 1e-5);
                }
            }
        }
    }

    [Fact]
    public void FramesWithoutTimingAreRejectedUnlessAllowed()
    {
        var all = Frames(150, 32, 32, 30);
        var input = EmbeddingGemma2MultimodalInput.FromVideo(EmbeddingGemma2Video.FromFramesWithoutTiming(all));
        using var strict = FakeMultimodal.Create();
        Assert.Throws<EmbeddingGemma2VideoTimingRequiredException>(() => strict.Embed(input, Ct));

        using var optIn = FakeMultimodal.Create(new EmbeddingGemma2MultimodalOptions { AllowVideosWithoutTiming = true });
        int[] upstream = EmbeddingGemma2VideoSampling.SampleIndices(150, null, null);
        var expected = strict.Embed(EmbeddingGemma2MultimodalInput.FromVideo(EmbeddingGemma2Video.FromSampledFrames(upstream.Select(i => all[i]).ToArray())), Ct);
        Assert.Equal(expected, optIn.Embed(input, Ct));
        Assert.NotEqual(expected, strict.Embed(EmbeddingGemma2MultimodalInput.FromVideo(EmbeddingGemma2Video.FromDecodedFrames(all, 30, 5)), Ct));
    }

    [Fact]
    public void TokenCountsIncludeEveryFrame()
    {
        using var embedder = FakeMultimodal.Create();
        var video = EmbeddingGemma2Video.FromSampledFrames(Frames(3, 64, 48, 40));
        int n = EmbeddingGemma2ImageProcessor.GetSoftTokenCount(64, 48, EmbeddingGemma2MultimodalEmbedder.VideoFrameSoftTokens);
        var counts = new int[1];
        embedder.EmbedBatch([EmbeddingGemma2MultimodalInput.FromVideo(video)], EmbeddingGemma2Prompt.None, counts, Ct);
        Assert.Equal((3 * (n + 2)) + 2, counts[0]);
        Assert.Equal(768, embedder.Embed(video, Ct).Length);
    }

    private sealed class Collector : IEmbeddingGemma2Diagnostics
    {
        public List<float[]> Images { get; } = [];

        public List<float[]> Video { get; } = [];

        public float[] Tokens { get; private set; } = [];

        public void OnSoftTokens(int input, int image, float[] softTokens) => Images.Add(softTokens);

        public void OnVideoSoftTokens(int input, int video, float[] softTokens) => Video.Add(softTokens);

        public void OnTokenEmbeddings(int input, ReadOnlySpan<float> tokenEmbeddings) => Tokens = tokenEmbeddings.ToArray();
    }
}
