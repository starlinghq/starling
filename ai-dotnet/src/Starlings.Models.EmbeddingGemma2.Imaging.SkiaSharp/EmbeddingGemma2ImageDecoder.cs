using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SkiaSharp;

namespace Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;

/// <summary>
/// Decodes encoded images (PNG, JPEG, GIF, BMP, WebP) with SkiaSharp into
/// <see cref="EmbeddingGemma2Image"/>, reproducing what upstream's processor receives from PIL's
/// <c>Image.open(...).convert("RGB")</c>.
/// </summary>
/// <remarks>
/// <para>
/// Conversion rules (upstream behaviour, transformers 5.19.0 with Pillow 12.3.0): the first frame; straight
/// (unpremultiplied) 8-bit RGBA with no colour-space conversion (embedded ICC profiles are ignored, as PIL does); alpha
/// dropped, never composited, so a transparent pixel keeps its stored colour; grayscale replicated; palettes looked up;
/// CMYK JPEG converted by Skia (equal to PIL on the fixture image). EXIF orientation is ignored unless
/// <see cref="EmbeddingGemma2ImageDecoderOptions.ApplyExifOrientation"/> is set, a deliberate deviation from upstream.
/// </para>
/// <para>
/// Rejected with <see cref="EmbeddingGemma2UnsupportedImageException"/>: images with more than 8 bits per channel
/// (upstream's <c>convert("RGB")</c> saturates 16-bit values to white; every C# decoder would scale them instead, so the
/// embedding would not be upstream's), truncated or undecodable data, and formats other than PNG, JPEG, GIF, BMP and
/// WebP. Images above <see cref="EmbeddingGemma2ImageDecoderOptions.MaxPixels"/> are rejected from the header, before
/// any pixel is decoded, with <see cref="EmbeddingGemma2ImageTooLargeException"/>. GIF images whose first frame has a
/// transparent colour are rejected too (<see cref="EmbeddingGemma2ImageRejection.GifTransparency"/>): PIL keeps the
/// palette colour of transparent pixels, SkiaSharp zeroes them, so the embedding would not be upstream's.
/// </para>
/// <para>
/// Fidelity (profile <c>embeddinggemma-2/image-fp32-cpu</c>, decoder <c>skiasharp</c>): on the 28 supported fixture
/// images (PNG RGB, grayscale, RGBA, LA, palette with and without transparency, 1-bit; JPEG baseline, progressive 4:2:0,
/// 4:4:4, CMYK, EXIF-rotated) and on 12 GIF, BMP and WebP files written by Pillow from them (GIF palette, grayscale and
/// two-frame animated, frame 0; BMP 24-bit, 32-bit with alpha, 8-bit palette and grayscale, 1-bit; WebP lossless and
/// lossy, with and without alpha) the decoded RGB equals PIL's byte for byte. Linux x64 is tested.
/// </para>
/// </remarks>
public static class EmbeddingGemma2ImageDecoder
{
    /// <summary>Formats with fixture evidence (decoded RGB equal to PIL's).</summary>
    public static IReadOnlyList<string> ExperimentalFormats { get; } = ["png", "jpeg", "gif", "bmp", "webp"];

    /// <summary>Formats that are decoded but have no fixture evidence (none in this version).</summary>
    public static IReadOnlyList<string> UntestedFormats { get; } = [];

    /// <summary>Decodes encoded image bytes.</summary>
    /// <exception cref="EmbeddingGemma2UnsupportedImageException">The data is not a supported, decodable 8-bit image.</exception>
    /// <exception cref="EmbeddingGemma2ImageTooLargeException">The image has more pixels than allowed.</exception>
    public static EmbeddingGemma2Image Decode(ReadOnlySpan<byte> encoded, EmbeddingGemma2ImageDecoderOptions? options = null)
    {
        options ??= new EmbeddingGemma2ImageDecoderOptions();
        options.Validate();
        if (encoded.IsEmpty)
        {
            throw new EmbeddingGemma2UnsupportedImageException(EmbeddingGemma2ImageRejection.Undecodable, null, "The image data is empty.");
        }

        using var data = SKData.CreateCopy(encoded);
        return DecodeData(data, IsSixteenBitPng(encoded), options);
    }

    /// <summary>
    /// Decodes an encoded image from a stream (read to the end; the stream is not disposed). The whole stream is
    /// buffered in memory before the header is read, so <see cref="EmbeddingGemma2ImageDecoderOptions.MaxPixels"/>
    /// bounds the decoded pixels, not the encoded bytes read from the stream; bound the stream's length first when the
    /// source is untrusted.
    /// </summary>
    public static EmbeddingGemma2Image Decode(Stream stream, EmbeddingGemma2ImageDecoderOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        return Decode(buffer.GetBuffer().AsSpan(0, (int)buffer.Length), options);
    }

    /// <summary>Decodes an image file.</summary>
    public static EmbeddingGemma2Image DecodeFile(string path, EmbeddingGemma2ImageDecoderOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        return Decode(File.ReadAllBytes(path), options);
    }

    /// <summary>
    /// Applies an EXIF orientation (1 to 8) to tightly packed RGB24 pixels, as Pillow's <c>ImageOps.exif_transpose</c>:
    /// 2 mirror, 3 rotate 180, 4 flip, 5 transpose, 6 rotate 90 clockwise, 7 transverse, 8 rotate 90 counter-clockwise.
    /// </summary>
    public static EmbeddingGemma2Image ApplyOrientation(EmbeddingGemma2Image image, int exifOrientation)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (exifOrientation is < 1 or > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(exifOrientation), exifOrientation, "EXIF orientation is 1 to 8.");
        }

        if (exifOrientation == 1)
        {
            return image;
        }

        int w = image.Width, h = image.Height;
        bool swap = exifOrientation >= 5;
        int ow = swap ? h : w, oh = swap ? w : h;
        var src = image.Pixels.Span;
        var dst = new byte[(long)ow * oh * 3];
        for (int y = 0; y < oh; y++)
        {
            for (int x = 0; x < ow; x++)
            {
                (int sx, int sy) = exifOrientation switch
                {
                    2 => (w - 1 - x, y),
                    3 => (w - 1 - x, h - 1 - y),
                    4 => (x, h - 1 - y),
                    5 => (y, x),
                    6 => (y, h - 1 - x),
                    7 => (w - 1 - y, h - 1 - x),
                    _ => (w - 1 - y, x),
                };
                int s = (sy * w + sx) * 3, d = (y * ow + x) * 3;
                dst[d] = src[s];
                dst[d + 1] = src[s + 1];
                dst[d + 2] = src[s + 2];
            }
        }

        return EmbeddingGemma2Image.FromRgb24(ow, oh, dst);
    }

    private static EmbeddingGemma2Image DecodeData(SKData data, bool sixteenBitPng, EmbeddingGemma2ImageDecoderOptions options)
    {
        using var codec = SKCodec.Create(data)
            ?? throw new EmbeddingGemma2UnsupportedImageException(EmbeddingGemma2ImageRejection.Undecodable, null, "SkiaSharp cannot read the image data (unknown or corrupt format).");
        string format = codec.EncodedFormat switch
        {
            SKEncodedImageFormat.Png => "png",
            SKEncodedImageFormat.Jpeg => "jpeg",
            SKEncodedImageFormat.Gif => "gif",
            SKEncodedImageFormat.Bmp => "bmp",
            SKEncodedImageFormat.Webp => "webp",
            var other => throw new EmbeddingGemma2UnsupportedImageException(EmbeddingGemma2ImageRejection.UnsupportedFormat, other.ToString().ToLowerInvariant(), $"{other} images are not supported; PNG, JPEG, GIF, BMP and WebP are."),
        };

        var info = codec.Info;
        if (sixteenBitPng || !IsEightBit(info.ColorType))
        {
            throw new EmbeddingGemma2UnsupportedImageException(
                EmbeddingGemma2ImageRejection.HighBitDepth,
                format,
                $"The {format} image has more than 8 bits per channel ({(sixteenBitPng ? "16-bit PNG" : info.ColorType.ToString())}). Upstream turns 16-bit images white; this package rejects them. Convert to 8 bits per channel first.");
        }

        if ((long)info.Width * info.Height > options.MaxPixels)
        {
            throw new EmbeddingGemma2ImageTooLargeException(info.Width, info.Height, options.MaxPixels);
        }

        var target = new SKImageInfo(info.Width, info.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        var rgba = new byte[target.BytesSize64];
        var handle = GCHandle.Alloc(rgba, GCHandleType.Pinned);
        SKCodecResult result;
        try
        {
            result = codec.GetPixels(target, handle.AddrOfPinnedObject());
        }
        finally
        {
            handle.Free();
        }

        if (result != SKCodecResult.Success)
        {
            throw new EmbeddingGemma2UnsupportedImageException(EmbeddingGemma2ImageRejection.Undecodable, format, $"SkiaSharp could not decode the {format} image ({result}); truncated or corrupt data is rejected, as PIL rejects it.");
        }

        if (format == "gif" && HasTransparentPixel(rgba))
        {
            throw new EmbeddingGemma2UnsupportedImageException(
                EmbeddingGemma2ImageRejection.GifTransparency,
                format,
                "The GIF image has transparent pixels. Upstream (PIL) keeps their palette colour and SkiaSharp does not, so this package rejects it. Convert it to PNG first.");
        }

        var image = EmbeddingGemma2Image.FromRgba32(info.Width, info.Height, rgba, target.RowBytes);
        if (options.ApplyExifOrientation)
        {
            image = ApplyOrientation(image, (int)codec.EncodedOrigin is >= 1 and <= 8 ? (int)codec.EncodedOrigin : 1);
        }

        return image;
    }

    private static bool HasTransparentPixel(byte[] rgba)
    {
        for (int i = 3; i < rgba.Length; i += 4)
        {
            if (rgba[i] != 255)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>True for the 8-bit colour types Skia reports for 8-bit sources.</summary>
    private static bool IsEightBit(SKColorType colorType) =>
        colorType is SKColorType.Rgba8888 or SKColorType.Bgra8888 or SKColorType.Gray8 or SKColorType.Alpha8 or SKColorType.Rgb888x or SKColorType.Rgb565 or SKColorType.Argb4444;

    /// <summary>PNG signature, then the IHDR chunk's bit depth (byte 24) is 16.</summary>
    private static bool IsSixteenBitPng(ReadOnlySpan<byte> encoded)
    {
        ReadOnlySpan<byte> signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        return encoded.Length > 24 && encoded[..8].SequenceEqual(signature) &&
               BinaryPrimitives.ReadUInt32BigEndian(encoded[12..16]) == 0x49484452 && // "IHDR"
               encoded[24] == 16;
    }
}

/// <summary>Decoder options.</summary>
public sealed record EmbeddingGemma2ImageDecoderOptions
{
    /// <summary>
    /// When <see langword="true"/>, the EXIF orientation is applied (the image is rotated or mirrored upright, as
    /// Pillow's <c>exif_transpose</c> does). Default <see langword="false"/>: upstream ignores EXIF orientation
    /// (sentence-transformers opens files without <c>exif_transpose</c>), so the default reproduces the reference.
    /// Setting this is a deviation from upstream: the embedding of a rotated photo then differs from
    /// sentence-transformers'.
    /// </summary>
    public bool ApplyExifOrientation { get; init; }

    /// <summary>
    /// Decoded pixels at most, checked from the header before any pixel is decoded. Default 50,000,000. It does not
    /// bound the encoded bytes: <see cref="EmbeddingGemma2ImageDecoder.Decode(Stream, EmbeddingGemma2ImageDecoderOptions?)"/>
    /// reads the whole stream into memory first.
    /// </summary>
    public long MaxPixels { get; init; } = 50_000_000;

    internal void Validate() => ArgumentOutOfRangeException.ThrowIfLessThan(MaxPixels, 1);
}

/// <summary>Why an image was rejected.</summary>
public enum EmbeddingGemma2ImageRejection
{
    /// <summary>Unknown, corrupt or truncated data.</summary>
    Undecodable,

    /// <summary>A format other than PNG, JPEG, GIF, BMP or WebP.</summary>
    UnsupportedFormat,

    /// <summary>More than 8 bits per channel (for example a 16-bit PNG).</summary>
    HighBitDepth,

    /// <summary>A GIF whose first frame has transparent pixels (PIL keeps their palette colour; SkiaSharp cannot).</summary>
    GifTransparency,
}

/// <summary>The image cannot be turned into an <see cref="EmbeddingGemma2Image"/> under the upstream rules.</summary>
public sealed class EmbeddingGemma2UnsupportedImageException : NotSupportedException
{
    /// <summary>Creates the exception.</summary>
    public EmbeddingGemma2UnsupportedImageException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public EmbeddingGemma2UnsupportedImageException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public EmbeddingGemma2UnsupportedImageException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Creates the exception with the reason and the detected format.</summary>
    public EmbeddingGemma2UnsupportedImageException(EmbeddingGemma2ImageRejection reason, string? format, string message)
        : base(message)
    {
        Reason = reason;
        Format = format;
    }

    /// <summary>Why the image was rejected.</summary>
    public EmbeddingGemma2ImageRejection Reason { get; init; }

    /// <summary>The detected format (lower case), or null when unknown.</summary>
    public string? Format { get; init; }
}
