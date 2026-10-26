using Starlings.Models.EmbeddingGemma2;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// The multimodal embedder's plumbing on weight-free graphs (soft tokens = mean of each 3x3 block's patch values; the
/// multimodal graph adds 1000 x the soft token where the mask is set): graph routing, soft-token order, batching,
/// padding, limits and lifetime. Runs in CI; numbers against the real model are in <see cref="ImageParityTests"/>.
/// </summary>
public sealed class MultimodalEmbedderContractTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void SoftTokensAreTheVisionOutputForEachImageInOrder()
    {
        using var embedder = FakeMultimodal.Create();
        var a = FakeMultimodal.Image(40, 30, 1);
        var b = FakeMultimodal.Image(100, 20, 2);
        var collector = new Collector();
        var input = EmbeddingGemma2MultimodalInput.Create(a, " and ", b);
        var destination = new float[768];
        embedder.EmbedCore([input], EmbeddingGemma2Prompt.None, destination, default, collector, Ct);

        foreach (var (image, k) in new[] { (a, 0), (b, 1) })
        {
            var p = EmbeddingGemma2ImageProcessor.Preprocess(image);
            var soft = collector.Soft[k];
            Assert.Equal(p.SoftTokenCount * 512, soft.Length);
            for (int s = 0; s < p.SoftTokenCount; s++)
            {
                double mean = 0;
                foreach (float v in p.PixelValues.Span.Slice(s * 9 * 768, 9 * 768))
                {
                    mean += v;
                }

                mean /= 9 * 768;
                Assert.Equal(mean, soft[s * 512], tolerance: 1e-5);
                Assert.Equal(soft[s * 512], soft[(s * 512) + 511]);
            }
        }

        // The token embeddings carry soft token s of image k at the s-th <|image|> of that image's expansion.
        var ids = embedder.Tokenize(input).InputIds.ToArray();
        var tokens = collector.Tokens;
        int imageIndex = -1, slot = 0;
        for (int t = 0; t < ids.Length; t++)
        {
            if (ids[t] == EmbeddingGemma2MultimodalEmbedder.ImageStartTokenId)
            {
                imageIndex++;
                slot = 0;
            }
            else if (ids[t] == EmbeddingGemma2MultimodalEmbedder.ImageTokenId)
            {
                double expected = (ids[t] * 1.0) + 1.0 + (collector.Soft[imageIndex][slot * 512] * 1000.0);
                Assert.Equal(expected, tokens[t * 768], tolerance: 0.1);
                slot++;
            }
        }

        Assert.Equal(1, imageIndex);
    }

    [Fact]
    public void BatchesOfDifferentSizesPaddingAndVisionBatchingGiveTheSameEmbeddings()
    {
        var inputs = new[]
        {
            EmbeddingGemma2MultimodalInput.FromImage(FakeMultimodal.Image(451, 300, 3)),
            EmbeddingGemma2MultimodalInput.FromText("a text"),
            EmbeddingGemma2MultimodalInput.Create("caption ", FakeMultimodal.Image(2000, 40, 4)),
            EmbeddingGemma2MultimodalInput.Create(FakeMultimodal.Image(8, 6, 5), "x", FakeMultimodal.Image(30, 1500, 6)),
            EmbeddingGemma2MultimodalInput.FromText("another"),
        };

        using var single = FakeMultimodal.Create();
        var expected = inputs.Select(i => single.Embed(i, Ct)).ToArray();
        foreach (var options in new[]
        {
            new EmbeddingGemma2MultimodalOptions(),
            new EmbeddingGemma2MultimodalOptions { MaxVisionBatchSize = 8, MaxVisionBatchPatches = 8 * 2_520, MaxBatchSize = 2 },
            new EmbeddingGemma2MultimodalOptions { PadImagesToMaxPatches = true, MaxVisionBatchSize = 3, MaxVisionBatchPatches = 3 * 2_520 },
        })
        {
            using var embedder = FakeMultimodal.Create(options);
            var counts = new int[inputs.Length];
            var batch = embedder.EmbedBatch(inputs, EmbeddingGemma2Prompt.None, counts, Ct);
            for (int i = 0; i < inputs.Length; i++)
            {
                Assert.Equal(expected[i], batch[i]);
                Assert.Equal(embedder.Tokenize(inputs[i]).Length, counts[i]);
            }
        }
    }

    [Fact]
    public void TextOnlyInputsRunOnTheTextEncoder()
    {
        using var embedder = FakeMultimodal.Create();
        using var encoder = new EmbeddingGemma2Encoder(OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2, new OnnxSessionOptions { ComputeModelSha256 = false }), ownsSession: true);
        var text = new EmbeddingGemma2TextEmbedder(new MarkerTokenizer(), encoder);
        Assert.Equal(text.Embed("hello", EmbeddingGemma2Prompt.Query, Ct), embedder.Embed(EmbeddingGemma2MultimodalInput.FromText("hello"), EmbeddingGemma2Prompt.Query, Ct));

        using var noText = FakeMultimodal.Create(withTextEncoder: false);
        Assert.Throws<InvalidOperationException>(() => noText.Embed(EmbeddingGemma2MultimodalInput.FromText("hello"), Ct));
        Assert.Equal(768, noText.Embed(FakeMultimodal.Image(48, 48, 1), Ct).Length);
    }

    [Fact]
    public void PromptsAreRenderedFirst()
    {
        using var embedder = FakeMultimodal.Create();
        var image = FakeMultimodal.Image(48, 48, 9);
        var ids = embedder.Tokenize(EmbeddingGemma2MultimodalInput.FromImage(image), EmbeddingGemma2Prompt.Query).InputIds.ToArray();
        var prompt = EmbeddingGemma2Prompt.SearchQueryText.Select(c => (long)c + 10).ToArray();
        Assert.Equal(prompt, ids[1..(1 + prompt.Length)]);
        Assert.Equal(EmbeddingGemma2MultimodalEmbedder.ImageStartTokenId, ids[1 + prompt.Length]);
        Assert.Equal(256, ids.Count(i => i == EmbeddingGemma2MultimodalEmbedder.ImageTokenId)); // 48x48 upscales to 768x768
        Assert.NotEqual(embedder.Embed(image, Ct), embedder.Embed(EmbeddingGemma2MultimodalInput.FromImage(image), EmbeddingGemma2Prompt.Query, Ct));
    }

    [Fact]
    public void MatryoshkaDimensionTruncatesTheNativeEmbedding()
    {
        using var native = FakeMultimodal.Create();
        using var small = FakeMultimodal.Create(new EmbeddingGemma2MultimodalOptions { Dimension = 128 });
        var image = FakeMultimodal.Image(64, 64, 7);
        var e128 = small.Embed(image, Ct);
        Assert.Equal(128, e128.Length);
        var expected = EmbeddingGemma2Encoder.Truncate(native.Embed(image, Ct), 128);
        for (int d = 0; d < 128; d++)
        {
            Assert.Equal(expected[d], e128[d], 6);
        }
    }

    [Fact]
    public void LimitsAreCheckedBeforeAnyInference()
    {
        using var embedder = FakeMultimodal.Create(new EmbeddingGemma2MultimodalOptions { MaxSequenceLength = 300, MaxImagePixels = 1_000_000 });
        var big = FakeMultimodal.Image(1001, 1000, 1);
        var ok = FakeMultimodal.Image(64, 64, 2);
        Assert.Throws<EmbeddingGemma2ImageTooLargeException>(() => embedder.EmbedBatch([ok, big], Ct));
        var twoImages = EmbeddingGemma2MultimodalInput.Create(ok, ok);
        var ex = Assert.Throws<EmbeddingGemma2InputTooLongException>(() => embedder.EmbedBatch([EmbeddingGemma2MultimodalInput.FromImage(ok), twoImages], Ct));
        Assert.Equal(1, ex.InputIndex);
    }

    [Fact]
    public void WrongGraphsAreRejected()
    {
        var session = new OnnxSessionOptions { ComputeModelSha256 = false };
        using var text = OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2, session);
        using var vision = OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2Vision, session);
        using var multimodal = OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2Multimodal, session);
        Assert.Throws<NotSupportedException>(() => new EmbeddingGemma2MultimodalEmbedder(new MarkerTokenizer(), text, multimodal));
        Assert.Throws<NotSupportedException>(() => new EmbeddingGemma2MultimodalEmbedder(new MarkerTokenizer(), vision, text));
        Assert.Throws<NotSupportedException>(() => new EmbeddingGemma2Encoder(multimodal));
    }

    [Fact]
    public void TextEncoderMustShareTheDimensionAndTheSequenceLimit()
    {
        var session = new OnnxSessionOptions { ComputeModelSha256 = false };
        using var vision = OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2Vision, session);
        using var multimodal = OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2Multimodal, session);
        using var text = OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2, session);
        using var shorter = new EmbeddingGemma2Encoder(text, new EmbeddingGemma2EncoderOptions { MaxSequenceLength = 300 });
        using var narrower = new EmbeddingGemma2Encoder(text, new EmbeddingGemma2EncoderOptions { Dimension = 128 });

        var length = Assert.Throws<ArgumentException>(() => new EmbeddingGemma2MultimodalEmbedder(new MarkerTokenizer(), vision, multimodal, shorter));
        Assert.Equal("textEncoder", length.ParamName);
        Assert.Contains("300", length.Message, StringComparison.Ordinal);
        var dimension = Assert.Throws<ArgumentException>(() => new EmbeddingGemma2MultimodalEmbedder(new MarkerTokenizer(), vision, multimodal, narrower));
        Assert.Equal("textEncoder", dimension.ParamName);

        // The same limit on both sides is accepted, and text-only rows are checked against it.
        using var embedder = new EmbeddingGemma2MultimodalEmbedder(new MarkerTokenizer(), vision, multimodal, shorter, new EmbeddingGemma2MultimodalOptions { MaxSequenceLength = 300 });
        Assert.Equal(768, embedder.Embed(EmbeddingGemma2MultimodalInput.FromText("hello"), Ct).Length);
        Assert.Throws<EmbeddingGemma2InputTooLongException>(() => embedder.Embed(EmbeddingGemma2MultimodalInput.FromText(new string('x', 300)), Ct));
    }

    [Fact]
    public void CancellationAndDisposal()
    {
        var embedder = FakeMultimodal.Create();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => embedder.Embed(FakeMultimodal.Image(48, 48, 1), cts.Token));
        embedder.Dispose();
        embedder.Dispose();
        Assert.Throws<ObjectDisposedException>(() => embedder.Embed(FakeMultimodal.Image(48, 48, 1), Ct));
    }

    private sealed class Collector : IEmbeddingGemma2Diagnostics
    {
        public List<float[]> Soft { get; } = [];

        public float[] Tokens { get; private set; } = [];

        public void OnSoftTokens(int input, int image, float[] softTokens) => Soft.Add(softTokens);

        public void OnTokenEmbeddings(int input, ReadOnlySpan<float> tokenEmbeddings) => Tokens = tokenEmbeddings.ToArray();
    }
}
