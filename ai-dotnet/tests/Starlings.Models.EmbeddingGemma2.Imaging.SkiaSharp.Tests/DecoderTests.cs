using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp.Tests;

/// <summary>
/// The SkiaSharp decoder against PIL (decoder profile of <c>embeddinggemma-2/image-fp32-cpu</c>: decoded RGB exact on
/// the supported fixture images; needs the image set) and its contract rules on hand-made images (CI).
/// </summary>
public sealed class DecoderTests(ITestOutputHelper output)
{
    /// <summary>
    /// SHA-256 of <c>np.asarray(ImageOps.exif_transpose(Image.open("jpeg-exif-orientation.jpg")).convert("RGB"))</c>
    /// (Pillow 12.3.0, the fixture image set; 451x300 after the orientation-6 rotation).
    /// </summary>
    private const string PilExifTransposedSha256 = "2e90876c8313176dffdbe897f87bdc64beace06cef7dec1f84dce0795954bab1";

    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("image-fp32-cpu");

    [Fact]
    public void DecodedRgbEqualsPilOnEverySupportedFixtureImage()
    {
        Assert.True(Tolerances.DecoderSkiaSharpExact);
        var pilRgb = EmbeddingGemma2ImageTestAssets.Rows.SelectMany(r => r.Images).GroupBy(i => i.Image).ToDictionary(g => g.Key, g => g.First().RgbSha256, StringComparer.Ordinal);
        int n = 0;
        foreach (var entry in EmbeddingGemma2ImageTestAssets.Manifest)
        {
            if (entry.Id == EmbeddingGemma2ImageTestAssets.SixteenBitImageId)
            {
                continue;
            }

            var image = EmbeddingGemma2ImageDecoder.Decode(EmbeddingGemma2ImageTestAssets.ReadImageFile(entry.Id));
            Assert.Equal((entry.Width, entry.Height), (image.Width, image.Height));
            Assert.True(pilRgb[entry.Id] == Sha(image.Pixels.Span), $"{entry.Id} ({entry.Format} {entry.Mode}): decoded RGB differs from PIL's convert(\"RGB\")");
            n++;
        }

        output.WriteLine($"{n} images (PNG RGB/L/RGBA/LA/P/P+transparency/1-bit, JPEG baseline/progressive 4:2:0/4:4:4/CMYK/EXIF): decoded RGB equals PIL's");
        Assert.Equal(28, n);
    }

    /// <summary>
    /// Capability <c>image_decode_gif_bmp_webp</c>: GIF (palette, transparent, grayscale, two-frame animated: frame 0),
    /// BMP (24-bit, 32-bit with alpha, 8-bit palette, 8-bit grayscale, 1-bit) and WebP (lossless, lossy, lossy and
    /// lossless with alpha) written by Pillow from fixture images; decoded RGB equal to PIL's <c>convert("RGB")</c>
    /// (proposed <c>image-formats.json</c>). Needs the format files; skips otherwise.
    /// </summary>
    [Fact]
    public void GifBmpAndWebpDecodeLikePil()
    {
        Assert.True(Tolerances.DecoderSkiaSharpExact);
        var formats = EmbeddingGemma2ImageTestAssets.LoadFormats();
        string dir = EmbeddingGemma2ImageTestAssets.FormatsDirectory;
        string first = Path.Combine(dir, formats[0].File);
        Assert.SkipWhen(!File.Exists(first), $"format files not found ({first}); write them with run_embeddinggemma2_image_extra.py --formats-out <dir> and set {EmbeddingGemma2ImageTestAssets.FormatsEnvironmentVariable}.");
        var failures = new List<string>();
        foreach (var f in formats)
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(dir, f.File));
            Assert.True(f.Sha256 == Convert.ToHexStringLower(SHA256.HashData(bytes)), $"{f.File}: not the proposed file");
            if (f.Id == "gif-transparent")
            {
                // PIL keeps the palette colour of transparent pixels; SkiaSharp zeroes them: rejected by contract.
                var ex = Assert.Throws<EmbeddingGemma2UnsupportedImageException>(() => EmbeddingGemma2ImageDecoder.Decode(bytes));
                Assert.Equal(EmbeddingGemma2ImageRejection.GifTransparency, ex.Reason);
                output.WriteLine($"{f.Id} ({f.Format}, {f.Width}x{f.Height}): rejected ({ex.Reason})");
                continue;
            }

            var image = EmbeddingGemma2ImageDecoder.Decode(bytes);
            bool ok = (image.Width, image.Height) == (f.Width, f.Height) && Sha(image.Pixels.Span) == f.RgbSha256;
            output.WriteLine($"{f.Id} ({f.Format}, {f.Width}x{f.Height}, {f.Frames} frame(s)): {(ok ? "equal to PIL" : "DIFFERS from PIL")}");
            if (!ok)
            {
                failures.Add(f.Id);
            }
        }

        Assert.Empty(failures);
        Assert.Equal(13, formats.Count);
    }

    [Fact]
    public void TheSixteenBitFixtureImageIsRejected()
    {
        var ex = Assert.Throws<EmbeddingGemma2UnsupportedImageException>(() =>
            EmbeddingGemma2ImageDecoder.Decode(EmbeddingGemma2ImageTestAssets.ReadImageFile(EmbeddingGemma2ImageTestAssets.SixteenBitImageId)));
        Assert.Equal(EmbeddingGemma2ImageRejection.HighBitDepth, ex.Reason);
        Assert.Equal("png", ex.Format);
    }

    [Fact]
    public void ExifOrientationIsIgnoredByDefaultAndAppliedOnRequest()
    {
        var entry = EmbeddingGemma2ImageTestAssets.Manifest.Single(m => m.Id == "jpeg-exif-orientation");
        Assert.Equal(6, entry.ExifOrientation);
        byte[] file = EmbeddingGemma2ImageTestAssets.ReadImageFile(entry.Id);
        string upstream = EmbeddingGemma2ImageTestAssets.Rows.Single(r => r.Id == entry.Id).Images[0].RgbSha256;

        var parity = EmbeddingGemma2ImageDecoder.Decode(file);
        Assert.Equal((300, 451), (parity.Width, parity.Height));
        Assert.Equal(upstream, Sha(parity.Pixels.Span));

        var upright = EmbeddingGemma2ImageDecoder.Decode(file, new EmbeddingGemma2ImageDecoderOptions { ApplyExifOrientation = true });
        Assert.Equal((451, 300), (upright.Width, upright.Height));
        Assert.Equal(PilExifTransposedSha256, Sha(upright.Pixels.Span));

        // Images without an orientation tag are unchanged by the option.
        var plain = EmbeddingGemma2ImageTestAssets.ReadImageFile("photo-coffee");
        Assert.Equal(
            Sha(EmbeddingGemma2ImageDecoder.Decode(plain).Pixels.Span),
            Sha(EmbeddingGemma2ImageDecoder.Decode(plain, new EmbeddingGemma2ImageDecoderOptions { ApplyExifOrientation = true }).Pixels.Span));
    }

    /// <summary>A GIF with a transparent pixel (the 43-byte one-pixel GIF89a) is rejected with a typed reason (CI).</summary>
    [Fact]
    public void GifWithTransparencyIsRejectedWithATypedException()
    {
        byte[] gif = Convert.FromHexString("47494638396101000100800000000000FFFFFF21F90401000000002C00000000010001000002024401003B");
        var ex = Assert.Throws<EmbeddingGemma2UnsupportedImageException>(() => EmbeddingGemma2ImageDecoder.Decode(gif));
        Assert.Equal(EmbeddingGemma2ImageRejection.GifTransparency, ex.Reason);
        Assert.Equal("gif", ex.Format);
        Assert.Equal(["png", "jpeg", "gif", "bmp", "webp"], EmbeddingGemma2ImageDecoder.ExperimentalFormats);
        Assert.Empty(EmbeddingGemma2ImageDecoder.UntestedFormats);
    }

    [Fact]
    public void SixteenBitPngIsRejectedWithATypedException()
    {
        byte[] png = Png(2, 2, bitDepth: 16, colorType: 0, raw: [0, 0, 0, 255, 255, 0, 1, 0, 2, 0, 3]);
        var ex = Assert.Throws<EmbeddingGemma2UnsupportedImageException>(() => EmbeddingGemma2ImageDecoder.Decode(png));
        Assert.Equal(EmbeddingGemma2ImageRejection.HighBitDepth, ex.Reason);
        Assert.Contains("8 bits", ex.Message, StringComparison.Ordinal);

        byte[] rgb16 = Png(1, 1, bitDepth: 16, colorType: 2, raw: [0, 1, 2, 3, 4, 5, 6]);
        Assert.Equal(EmbeddingGemma2ImageRejection.HighBitDepth, Assert.Throws<EmbeddingGemma2UnsupportedImageException>(() => EmbeddingGemma2ImageDecoder.Decode(rgb16)).Reason);
    }

    [Fact]
    public void AlphaIsDroppedNotComposited()
    {
        // 2x1 RGBA: an opaque red pixel and a fully transparent pixel whose stored colour is (10, 20, 30).
        byte[] png = Png(2, 1, bitDepth: 8, colorType: 6, raw: [0, 255, 0, 0, 255, 10, 20, 30, 0]);
        var image = EmbeddingGemma2ImageDecoder.Decode(png);
        Assert.Equal([255, 0, 0, 10, 20, 30], image.Pixels.ToArray());
    }

    [Fact]
    public void GrayscaleIsReplicated()
    {
        byte[] png = Png(3, 1, bitDepth: 8, colorType: 0, raw: [0, 0, 128, 255]);
        Assert.Equal([0, 0, 0, 128, 128, 128, 255, 255, 255], EmbeddingGemma2ImageDecoder.Decode(png).Pixels.ToArray());
    }

    [Fact]
    public void ThePixelLimitIsCheckedFromTheHeader()
    {
        // 20,000 x 20,000 (400 MP) declared, a few bytes of pixel data: rejected before decoding.
        byte[] png = Png(20_000, 20_000, bitDepth: 8, colorType: 0, raw: [0, 0, 0, 0]);
        var ex = Assert.Throws<EmbeddingGemma2ImageTooLargeException>(() => EmbeddingGemma2ImageDecoder.Decode(png));
        Assert.Equal((20_000, 20_000, 50_000_000L), (ex.Width, ex.Height, ex.MaximumPixels));
        Assert.Throws<EmbeddingGemma2ImageTooLargeException>(() =>
            EmbeddingGemma2ImageDecoder.Decode(Png(3, 1, 8, 0, [0, 0, 128, 255]), new EmbeddingGemma2ImageDecoderOptions { MaxPixels = 2 }));
    }

    [Fact]
    public void TruncatedCorruptAndUnsupportedDataAreRejected()
    {
        byte[] png = Png(64, 64, bitDepth: 8, colorType: 2, raw: new byte[64 * (1 + 64 * 3)]);
        var truncated = png.AsSpan(0, png.Length - 40).ToArray();
        Assert.Equal(EmbeddingGemma2ImageRejection.Undecodable, Assert.Throws<EmbeddingGemma2UnsupportedImageException>(() => EmbeddingGemma2ImageDecoder.Decode(truncated)).Reason);
        Assert.Equal(EmbeddingGemma2ImageRejection.Undecodable, Assert.Throws<EmbeddingGemma2UnsupportedImageException>(() => EmbeddingGemma2ImageDecoder.Decode(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 })).Reason);
        Assert.Equal(EmbeddingGemma2ImageRejection.Undecodable, Assert.Throws<EmbeddingGemma2UnsupportedImageException>(() => EmbeddingGemma2ImageDecoder.Decode(ReadOnlySpan<byte>.Empty)).Reason);

        // WBMP (type 0, 8x1, one row of 8 black pixels): decodable by Skia, not a supported format here.
        var wbmp = Assert.Throws<EmbeddingGemma2UnsupportedImageException>(() => EmbeddingGemma2ImageDecoder.Decode(new byte[] { 0, 0, 8, 1, 0 }));
        Assert.Equal(EmbeddingGemma2ImageRejection.UnsupportedFormat, wbmp.Reason);
    }

    [Fact]
    public void StreamAndFileOverloadsDecodeTheSameImage()
    {
        byte[] png = Png(3, 1, bitDepth: 8, colorType: 0, raw: [0, 0, 128, 255]);
        using var stream = new MemoryStream(png);
        Assert.Equal(EmbeddingGemma2ImageDecoder.Decode(png).Pixels.ToArray(), EmbeddingGemma2ImageDecoder.Decode(stream).Pixels.ToArray());
        string path = Path.Combine(Path.GetTempPath(), $"eg2-decoder-{Guid.NewGuid():N}.png");
        File.WriteAllBytes(path, png);
        try
        {
            Assert.Equal(EmbeddingGemma2ImageDecoder.Decode(png).Pixels.ToArray(), EmbeddingGemma2ImageDecoder.DecodeFile(path).Pixels.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(1, 3, 2, "a b c / d e f")]
    [InlineData(2, 3, 2, "c b a / f e d")]
    [InlineData(3, 3, 2, "f e d / c b a")]
    [InlineData(4, 3, 2, "d e f / a b c")]
    [InlineData(5, 2, 3, "a d / b e / c f")]
    [InlineData(6, 2, 3, "d a / e b / f c")]
    [InlineData(7, 2, 3, "f c / e b / d a")]
    [InlineData(8, 2, 3, "c f / b e / a d")]
    public void OrientationsMatchPillowExifTranspose(int orientation, int width, int height, string expected)
    {
        // 3x2 image "a b c / d e f"; each pixel's red value is its letter. Expected layouts are PIL's exif_transpose:
        // 2 FLIP_LEFT_RIGHT, 3 ROTATE_180, 4 FLIP_TOP_BOTTOM, 5 TRANSPOSE, 6 ROTATE_270, 7 TRANSVERSE, 8 ROTATE_90.
        byte[] rgb = "abcdef".SelectMany(c => new[] { (byte)c, (byte)0, (byte)0 }).ToArray();
        var turned = EmbeddingGemma2ImageDecoder.ApplyOrientation(EmbeddingGemma2Image.FromRgb24(3, 2, rgb), orientation);
        Assert.Equal((width, height), (turned.Width, turned.Height));
        string layout = string.Join(" / ", Enumerable.Range(0, height).Select(y =>
            string.Join(' ', Enumerable.Range(0, width).Select(x => (char)turned.GetPixel(x, y).R))));
        Assert.Equal(expected, layout);
    }

    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    /// <summary>A minimal PNG: IHDR, one zlib IDAT of the given filtered scanlines, IEND.</summary>
    private static byte[] Png(int width, int height, byte bitDepth, byte colorType, byte[] raw)
    {
        using var ms = new MemoryStream();
        ms.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        var ihdr = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(ihdr, width);
        BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(4), height);
        ihdr[8] = bitDepth;
        ihdr[9] = colorType;
        Chunk(ms, "IHDR", ihdr);
        using (var z = new MemoryStream())
        {
            using (var zs = new ZLibStream(z, CompressionLevel.Optimal, leaveOpen: true))
            {
                zs.Write(raw);
            }

            Chunk(ms, "IDAT", z.ToArray());
        }

        Chunk(ms, "IEND", []);
        return ms.ToArray();
    }

    private static void Chunk(Stream s, string type, byte[] data)
    {
        Span<byte> len = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(len, data.Length);
        s.Write(len);
        byte[] typed = [.. type.Select(c => (byte)c), .. data];
        s.Write(typed);
        BinaryPrimitives.WriteUInt32BigEndian(len, Crc32(typed));
        s.Write(len);
    }

    private static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++)
            {
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
            }
        }

        return ~crc;
    }
}
