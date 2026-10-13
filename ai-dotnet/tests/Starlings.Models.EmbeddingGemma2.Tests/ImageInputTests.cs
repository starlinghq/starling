using Starlings.Models.EmbeddingGemma2;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// Image inputs without the graphs: rendered ids against the accepted fixtures (tokenizer only; CI), preprocessing of
/// the fixture images bit for bit (image set only), and validation.
/// </summary>
public sealed class ImageInputTests(ITestOutputHelper output)
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("image-fp32-cpu");

    [Fact]
    public void RenderedIdsEqualTheFixtureOnEveryRow()
    {
        Assert.True(Tolerances.IsExact("token_ids"));
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        var options = new EmbeddingGemma2MultimodalOptions();
        int n = 0;
        foreach (var row in EmbeddingGemma2ImageTestAssets.Rows)
        {
            // Rendering depends only on each image's decoded size: zero images of the fixture sizes.
            var input = EmbeddingGemma2ImageTestAssets.BuildInput(row, EmbeddingGemma2ImageTestAssets.SizedPlaceholder);
            var prepared = EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, options, input, row.PackagePrompt, 0);
            Assert.True(row.InputIds.AsSpan().SequenceEqual(prepared.Tokens.InputIds.Span), $"{row.Id}: rendered ids differ from the fixture");
            Assert.Equal(row.Images.Select(i => i.SoftTokens), prepared.SoftTokenCounts);
            n++;
        }

        output.WriteLine($"{n} rows (29 image-only, 6 interleaved or prompted, 8 text): ids exact with the owned tokenizer");
        Assert.Equal(43, n);
    }

    [Fact]
    public void PreprocessingOfTheFixtureImagesIsBitExact()
    {
        Assert.True(Tolerances.IsExact("resized_pixels") && Tolerances.IsExact("pixel_values") && Tolerances.IsExact("image_positions"));
        int n = 0;
        foreach (var image in EmbeddingGemma2ImageTestAssets.Rows.Where(r => r.Kind == "image" && r.Prompt == "none").SelectMany(r => r.Images))
        {
            if (image.Image == EmbeddingGemma2ImageTestAssets.SixteenBitImageId)
            {
                continue; // rejected by the decoder (contract); upstream renders it white
            }

            var decoded = EmbeddingGemma2ImageTestAssets.Decode(image.Image);
            Assert.Equal(image.RgbSha256, EmbeddingGemma2ImageTestAssets.Sha(decoded.Pixels.Span));
            var p = EmbeddingGemma2ImageProcessor.Preprocess(decoded);
            Assert.Equal((image.ResizedWidth, image.ResizedHeight), (p.ResizedWidth, p.ResizedHeight));
            Assert.True(image.ResizedSha256 == EmbeddingGemma2ImageTestAssets.Sha(p.ResizedPixels.Span), $"{image.Image}: resized bytes differ");
            Assert.True(image.PixelValuesSha256 == EmbeddingGemma2ImageTestAssets.Sha(p.PixelValues.Span), $"{image.Image}: pixel values differ");
            Assert.True(image.PositionsSha256 == EmbeddingGemma2ImageTestAssets.Sha(p.Positions.Span), $"{image.Image}: positions differ");
            Assert.Equal(image.SoftTokens, p.SoftTokenCount);
            n++;
        }

        output.WriteLine($"{n} images: decoded RGB, resized bytes, pixel values and positions equal the reference digests");
        Assert.Equal(28, n);
    }

    [Fact]
    public void InputsOverTheContextAreRejectedBeforeInference()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        var image = EmbeddingGemma2Image.FromRgb24(512, 512, new byte[512 * 512 * 3]); // 256 soft tokens
        var parts = Enumerable.Repeat<object>(image, 32).ToList(); // 32 x 258 = 8,256 > 8,192
        var input = EmbeddingGemma2MultimodalInput.Create(parts);
        var ex = Assert.Throws<EmbeddingGemma2InputTooLongException>(() =>
            EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions(), input, EmbeddingGemma2Prompt.None, 3));
        Assert.Equal(3, ex.InputIndex);
        Assert.Equal(8_192, ex.MaximumTokens);

        // 31 images fit: 31 x 258 + 2 = 8,000 tokens.
        var ok = EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions(), EmbeddingGemma2MultimodalInput.Create(parts.Take(31)), EmbeddingGemma2Prompt.None, 0);
        Assert.Equal(8_000, ok.Tokens.Length);
    }

    [Fact]
    public void ImagesAboveThePixelLimitAreRejected()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        var image = EmbeddingGemma2Image.FromRgb24(100, 100, new byte[100 * 100 * 3]);
        var ex = Assert.Throws<EmbeddingGemma2ImageTooLargeException>(() =>
            EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions { MaxImagePixels = 9_999 }, EmbeddingGemma2MultimodalInput.FromImage(image), EmbeddingGemma2Prompt.None, 0));
        Assert.Equal((100, 100, 9_999L), (ex.Width, ex.Height, ex.MaximumPixels));
        Assert.Equal(50_000_000, new EmbeddingGemma2MultimodalOptions().MaxImagePixels);
    }

    [Theory]
    [InlineData("<|image|>")]
    [InlineData("caption <|image>")]
    [InlineData("<image|>")]
    [InlineData("a <|video|> b")]
    [InlineData("<|audio|>")]
    public void MediaPlaceholdersInTextAreRejected(string text)
    {
        var image = EmbeddingGemma2Image.FromRgb24(1, 1, new byte[3]);
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2MultimodalInput.Create(text, image));
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2MultimodalInput.FromText(text));
    }

    [Fact]
    public void PartsAreTextOrImages()
    {
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2MultimodalInput.Create(42));
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2MultimodalInput.Create(Array.Empty<object>()));
        var image = EmbeddingGemma2Image.FromRgb24(1, 1, new byte[3]);
        var input = EmbeddingGemma2MultimodalInput.Create("a", image, string.Empty, "b", image);
        Assert.Equal(4, input.Parts.Count); // the empty string is dropped
        Assert.Equal(2, input.ImageCount);
        Assert.True(input.HasImages);
        Assert.False(EmbeddingGemma2MultimodalInput.FromText("x").HasImages);
    }

    [Fact]
    public void OptionsValidate()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2MultimodalOptions { MaxImageSoftTokens = 100 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2MultimodalOptions { Dimension = 300 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2MultimodalOptions { MaxVisionBatchSize = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2MultimodalOptions { MaxSequenceLength = 8_193 }.Validate());
        var defaults = new EmbeddingGemma2MultimodalOptions();
        Assert.Equal((1, 2_520, 280, false), (defaults.MaxVisionBatchSize, defaults.MaxVisionBatchPatches, defaults.MaxImageSoftTokens, defaults.PadImagesToMaxPatches));
    }
}
