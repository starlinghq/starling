using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Models.EmbeddingGemma2;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// The C# image preprocessing against weight-free torchvision/transformers cases (Data/image-processor-cases.json,
/// made by Data/make_image_processor_cases.py): resized bytes, target sizes and patch tensors equal bit for bit. Runs
/// in CI (no model, no images).
/// </summary>
public sealed class ImageProcessorTests
{
    private static readonly JsonElement Cases = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        EmbeddingGemma2TestAssets.RepositoryRoot, "ai-dotnet", "tests", "Starlings.Models.EmbeddingGemma2.Tests", "Data", "image-processor-cases.json"))).RootElement;

    [Fact]
    public void ResizeEqualsTorchvisionUint8AntialiasBicubicBitForBit()
    {
        int n = 0;
        foreach (var c in Cases.GetProperty("resize").EnumerateArray())
        {
            int w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32();
            int tw = c.GetProperty("target_width").GetInt32(), th = c.GetProperty("target_height").GetInt32();
            byte[] input = Convert.FromBase64String(c.GetProperty("input").GetString()!);
            byte[] output = EmbeddingGemma2ImageProcessor.Resize(input, w, h, tw, th);
            Assert.Equal(tw * th * 3, output.Length);
            Assert.True(
                c.GetProperty("output_sha256").GetString() == Sha(output),
                $"{w}x{h} -> {tw}x{th}: resized bytes differ from torchvision");
            n++;
        }

        Assert.Equal(15, n);
    }

    [Fact]
    public void TargetSizeEqualsTransformersForEveryBudget()
    {
        int n = 0;
        foreach (var c in Cases.GetProperty("target_size").EnumerateArray())
        {
            int w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32(), budget = c.GetProperty("budget").GetInt32();
            var (tw, th) = EmbeddingGemma2ImageProcessor.GetTargetSize(w, h, budget);
            Assert.True(
                (c.GetProperty("target_width").GetInt32(), c.GetProperty("target_height").GetInt32()) == (tw, th),
                $"{w}x{h} at {budget}: got {tw}x{th}");
            Assert.True(tw % 48 == 0 && th % 48 == 0);
            Assert.True((tw / 16) * (th / 16) <= budget * 9);
            Assert.Equal((tw / 48) * (th / 48), EmbeddingGemma2ImageProcessor.GetSoftTokenCount(w, h, budget));
            n++;
        }

        Assert.Equal(120, n);
    }

    [Fact]
    public void PatchesAndPositionsEqualTheProcessor()
    {
        foreach (var c in Cases.GetProperty("patches").EnumerateArray())
        {
            int w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32();
            byte[] input = Convert.FromBase64String(c.GetProperty("input").GetString()!);
            int patches = (w / 16) * (h / 16);
            var pixelValues = new float[patches * 768];
            var positions = new long[patches * 2];
            EmbeddingGemma2ImageProcessor.WritePatches(input, w, h, pixelValues, positions);
            Assert.Equal(c.GetProperty("pixel_values_sha256").GetString(), Sha(pixelValues));
            Assert.Equal(c.GetProperty("positions_sha256").GetString(), Sha(positions));
        }
    }

    [Fact]
    public void PreprocessSkipsTheResizeAtTheTargetSize()
    {
        var image = EmbeddingGemma2Image.FromRgb24(864, 672, new byte[864 * 672 * 3]);
        var p = EmbeddingGemma2ImageProcessor.Preprocess(image);
        Assert.Equal((864, 672), (p.ResizedWidth, p.ResizedHeight));
        Assert.Equal(54 * 42, p.PatchCount);
        Assert.Equal(252, p.SoftTokenCount);
    }

    [Fact]
    public void RescaleIsFloat32ProductNotDivision()
    {
        // processor_config rescale_factor 0.00392156862745098 in float32, multiplied (not v / 255f).
        int differ = 0;
        for (int v = 0; v < 256; v++)
        {
            differ += v * EmbeddingGemma2ImageProcessor.RescaleFactor != v / 255f ? 1 : 0;
        }

        Assert.True(differ > 0, "the product and the division agree on every byte; the test lost its point");
        byte[] bytes = Enumerable.Range(0, 48 * 48 * 3).Select(i => (byte)(i % 256)).ToArray();
        var pixelValues = new float[9 * 768];
        EmbeddingGemma2ImageProcessor.WritePatches(bytes, 48, 48, pixelValues, new long[18]);
        Assert.Equal(1 * EmbeddingGemma2ImageProcessor.RescaleFactor, pixelValues[1]);
    }

    [Theory]
    [InlineData(451, 300, 960, 624, 260)]
    [InlineData(512, 512, 768, 768, 256)]
    [InlineData(4000, 3000, 912, 672, 266)]
    [InlineData(2000, 40, 5664, 96, 236)]
    [InlineData(4000, 8, 13440, 48, 280)]
    [InlineData(1, 1, 768, 768, 256)]
    public void SpikeExamplesAtTheDefaultBudget(int w, int h, int tw, int th, int soft)
    {
        Assert.Equal((tw, th), EmbeddingGemma2ImageProcessor.GetTargetSize(w, h));
        Assert.Equal(soft, EmbeddingGemma2ImageProcessor.GetSoftTokenCount(w, h));
    }

    [Fact]
    public void UnsupportedBudgetsAndBadSizesAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2ImageProcessor.GetTargetSize(10, 10, 100));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2ImageProcessor.GetTargetSize(0, 10));
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2ImageProcessor.Resize(new byte[5], 2, 1, 4, 4));
        Assert.Equal([70, 140, 280, 560, 1120], EmbeddingGemma2ImageProcessor.SupportedSoftTokenBudgets);
    }

    [Fact]
    public void ImageTypeValidatesAndCopiesStridedAndRgbaInput()
    {
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2Image.FromRgb24(2, 2, new byte[11]));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2Image.FromRgb24(0, 2, Array.Empty<byte>()));

        // Two rows of 2 RGBA pixels with 3 bytes of row padding; alpha is dropped, colours kept as stored.
        byte[] rgba = [1, 2, 3, 0, 4, 5, 6, 255, 9, 9, 9, 10, 11, 12, 128, 13, 14, 15, 0, 9, 9, 9];
        var image = EmbeddingGemma2Image.FromRgba32(2, 2, rgba, 11);
        Assert.Equal([1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15], image.Pixels.ToArray());
        Assert.Equal(((byte)13, (byte)14, (byte)15), image.GetPixel(1, 1));

        byte[] rgb = [1, 2, 3, 7, 4, 5, 6, 7];
        Assert.Equal([1, 2, 3, 4, 5, 6], EmbeddingGemma2Image.FromRgb24(1, 2, rgb, 4).Pixels.ToArray());
    }

    [Fact]
    public void VisionCallPlanRespectsImageAndPatchBudgets()
    {
        Assert.Equal([1, 1, 1], EmbeddingGemma2MultimodalEmbedder.PlanVisionCalls([2520, 2340, 2520], 1, 2520));
        Assert.Equal([2, 1], EmbeddingGemma2MultimodalEmbedder.PlanVisionCalls([900, 1260, 2520], 8, 2520));
        Assert.Equal([3], EmbeddingGemma2MultimodalEmbedder.PlanVisionCalls([2520, 2340, 2520], 8, 3 * 2520));
        Assert.Equal([1, 1], EmbeddingGemma2MultimodalEmbedder.PlanVisionCalls([5000, 9], 8, 2520));
    }

    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    private static string Sha(ReadOnlySpan<float> values) => Sha(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values));

    private static string Sha(ReadOnlySpan<long> values) => Sha(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values));
}
