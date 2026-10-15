using System.Runtime.Intrinsics.X86;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// The AVX2 and multi-threaded resize and patch paths against the scalar single-threaded loops (the port the
/// torchvision cases in <see cref="ImageProcessorTests"/> pinned): equal bytes and values on random sizes, contents,
/// budgets and arbitrary target sizes, including 1-pixel sides, extreme aspects and saturating contents. Runs in CI.
/// </summary>
public sealed class ImageProcessorPathTests
{
    [Fact]
    public void FastPathsEqualTheScalarLoops()
    {
        Assert.SkipUnless(Avx2.IsSupported, "AVX2 is not available; only the scalar loops run on this machine.");
        var random = new Random(20261015);
        int[] budgets = [70, 140, 280, 560, 1_120];
        int checkedSizes = 0;
        for (int i = 0; i < 100; i++)
        {
            int w = random.Next(1, i % 25 == 0 ? 1_600 : 400);
            int h = random.Next(1, i % 20 == 0 ? 1_600 : 400);
            byte[] rgb = Content(random, i, w, h);
            int budget = i % 10 == 9 ? budgets[random.Next(budgets.Length)] : budgets[random.Next(2)];
            (int Width, int Height) target;
            try
            {
                target = EmbeddingGemma2ImageProcessor.GetTargetSize(w, h, budget);
            }
            catch (ArgumentException)
            {
                continue; // resizes to nothing at this budget
            }

            CheckResize(rgb, w, h, target.Width, target.Height);
            CheckResize(rgb, w, h, random.Next(1, 300), random.Next(1, 300));
            CheckResize(rgb, w, h, w, random.Next(1, 300));
            CheckResize(rgb, w, h, random.Next(1, 300), h);
            checkedSizes++;

            byte[] resized = TorchAntialiasResampler.Resize(rgb, rgb, w, h, target.Width, target.Height, vectorized: false, maxThreads: 1);
            int patches = (target.Width / 16) * (target.Height / 16);
            var scalar = new float[patches * 768];
            var fast = new float[patches * 768];
            EmbeddingGemma2ImageProcessor.WritePatches(resized, target.Width, target.Height, scalar, new long[patches * 2], vectorized: false);
            EmbeddingGemma2ImageProcessor.WritePatches(resized, target.Width, target.Height, fast, new long[patches * 2], vectorized: true);
            Assert.True(scalar.AsSpan().SequenceEqual(fast), $"{w}x{h}: AVX2 patch values differ from the scalar loop");
        }

        Assert.True(checkedSizes > 60, $"only {checkedSizes} sizes resized");
    }

    [Fact]
    public void BatchPreprocessingEqualsSingleImages()
    {
        var random = new Random(140);
        var images = Enumerable.Range(0, 7).Select(i =>
        {
            int w = random.Next(1, 700), h = random.Next(1, 500);
            return EmbeddingGemma2Image.FromRgb24(w, h, Content(random, i, w, h));
        }).ToArray();
        foreach (int budget in new[] { 140, 280 })
        {
            var batch = EmbeddingGemma2ImageProcessor.Preprocess(images, budget);
            Assert.Equal(images.Length, batch.Length);
            for (int i = 0; i < images.Length; i++)
            {
                var single = EmbeddingGemma2ImageProcessor.Preprocess(images[i], budget);
                Assert.Equal((single.ResizedWidth, single.ResizedHeight), (batch[i].ResizedWidth, batch[i].ResizedHeight));
                Assert.True(single.ResizedPixels.Span.SequenceEqual(batch[i].ResizedPixels.Span), $"image {i} at {budget}: resized bytes");
                Assert.True(single.PixelValues.Span.SequenceEqual(batch[i].PixelValues.Span), $"image {i} at {budget}: pixel values");
                Assert.True(single.Positions.Span.SequenceEqual(batch[i].Positions.Span), $"image {i} at {budget}: positions");
            }
        }

        Assert.Empty(EmbeddingGemma2ImageProcessor.Preprocess(Array.Empty<EmbeddingGemma2Image>()));
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2ImageProcessor.Preprocess([images[0], null!]));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2ImageProcessor.Preprocess(images, 100));
    }

    private static void CheckResize(byte[] rgb, int w, int h, int tw, int th)
    {
        byte[] expected = TorchAntialiasResampler.Resize(rgb, rgb, w, h, tw, th, vectorized: false, maxThreads: 1);
        byte[] vectorized = TorchAntialiasResampler.Resize(rgb, default, w, h, tw, th, vectorized: true, maxThreads: 1);
        byte[] parallel = TorchAntialiasResampler.Resize(rgb, rgb, w, h, tw, th, vectorized: true, maxThreads: 8);
        byte[] parallelScalar = TorchAntialiasResampler.Resize(rgb, rgb, w, h, tw, th, vectorized: false, maxThreads: 8);
        Assert.True(expected.AsSpan().SequenceEqual(vectorized), $"{w}x{h} -> {tw}x{th}: AVX2 resize differs from the scalar loop");
        Assert.True(expected.AsSpan().SequenceEqual(parallel), $"{w}x{h} -> {tw}x{th}: parallel AVX2 resize differs from the scalar loop");
        Assert.True(expected.AsSpan().SequenceEqual(parallelScalar), $"{w}x{h} -> {tw}x{th}: parallel scalar resize differs from the scalar loop");
    }

    private static byte[] Content(Random random, int i, int w, int h)
    {
        var rgb = new byte[w * h * 3];
        switch (i % 4)
        {
            case 0:
                random.NextBytes(rgb);
                break;
            case 1:
                Array.Fill(rgb, (byte)255); // saturates high after overshoot
                break;
            case 2:
                for (int k = 0; k < rgb.Length; k++)
                {
                    rgb[k] = (byte)(((k / 3) + (k / (3 * w))) % 2 * 255); // checkerboard: overshoot on both sides
                }

                break;
            default:
                for (int k = 0; k < rgb.Length; k++)
                {
                    rgb[k] = (byte)((k * 7) + (k / (3 * w) * 13));
                }

                break;
        }

        return rgb;
    }
}
