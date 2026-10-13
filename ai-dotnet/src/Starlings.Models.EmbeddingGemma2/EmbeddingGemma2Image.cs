namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// An already-decoded image: 8-bit RGB, three bytes per pixel (red, green, blue), rows top to bottom, no padding between
/// rows. This is what PIL's <c>image.convert("RGB")</c> gives the upstream processor; from these bytes the package
/// reproduces the reference's preprocessing bit for bit (resize, rescale, patches).
/// </summary>
/// <remarks>
/// <para>
/// The package does not decode files: decode with your own stack, or with
/// <c>Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp</c>, which reproduces PIL's decoded RGB on the tested
/// formats. Upstream's conversion rules apply to whatever decoder you use: alpha is dropped (not composited), grayscale
/// is replicated to three channels, palettes are looked up, EXIF orientation is not applied, and images with 16 bits per
/// channel are not representable here (upstream saturates them to white; this package rejects them in the decoder).
/// </para>
/// <para>
/// The image holds a reference to <see cref="Pixels"/> and never copies or modifies it; keep the memory unchanged while
/// an embedding call that uses the image runs. Instances are immutable.
/// </para>
/// </remarks>
public sealed class EmbeddingGemma2Image
{
    /// <summary>Bytes per pixel of <see cref="Pixels"/>.</summary>
    public const int BytesPerPixel = 3;

    private EmbeddingGemma2Image(int width, int height, ReadOnlyMemory<byte> pixels)
    {
        Width = width;
        Height = height;
        Pixels = pixels;
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>Tightly packed RGB24 pixels, <c>Width * Height * 3</c> bytes, row-major.</summary>
    public ReadOnlyMemory<byte> Pixels { get; }

    /// <summary>Number of pixels (<c>Width * Height</c>).</summary>
    public long PixelCount => (long)Width * Height;

    /// <summary>
    /// Wraps tightly packed RGB24 pixels (no copy). <paramref name="pixels"/> must hold exactly
    /// <c>width * height * 3</c> bytes.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is not positive.</exception>
    /// <exception cref="ArgumentException">The buffer length does not match the dimensions.</exception>
    public static EmbeddingGemma2Image FromRgb24(int width, int height, ReadOnlyMemory<byte> pixels)
    {
        ValidateSize(width, height);
        long expected = (long)width * height * BytesPerPixel;
        if (pixels.Length != expected)
        {
            throw new ArgumentException($"A {width}x{height} RGB24 image needs {expected} bytes; got {pixels.Length}.", nameof(pixels));
        }

        return new EmbeddingGemma2Image(width, height, pixels);
    }

    /// <summary>
    /// Copies RGB24 rows that are <paramref name="stride"/> bytes apart (stride at least <c>width * 3</c>) into a new
    /// tightly packed image.
    /// </summary>
    public static EmbeddingGemma2Image FromRgb24(int width, int height, ReadOnlySpan<byte> pixels, int stride)
    {
        return CopyRows(width, height, pixels, stride, 3);
    }

    /// <summary>
    /// Copies 8-bit RGBA (or RGBX) pixels, rows <paramref name="stride"/> bytes apart, into a new RGB24 image, dropping
    /// the fourth byte. This matches upstream's <c>convert("RGB")</c> for straight (unpremultiplied) alpha: the colour of
    /// a transparent pixel is kept as stored, never composited on a background. Pass straight-alpha pixels; premultiplied
    /// input gives other colours than the reference.
    /// </summary>
    public static EmbeddingGemma2Image FromRgba32(int width, int height, ReadOnlySpan<byte> pixels, int stride)
    {
        return CopyRows(width, height, pixels, stride, 4);
    }

    /// <summary>The RGB bytes of pixel (<paramref name="x"/>, <paramref name="y"/>).</summary>
    public (byte R, byte G, byte B) GetPixel(int x, int y)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(x);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(x, Width);
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(y, Height);
        var span = Pixels.Span;
        int i = checked((y * Width + x) * BytesPerPixel);
        return (span[i], span[i + 1], span[i + 2]);
    }

    private static EmbeddingGemma2Image CopyRows(int width, int height, ReadOnlySpan<byte> pixels, int stride, int channels)
    {
        ValidateSize(width, height);
        long rowBytes = (long)width * channels;
        if (stride < rowBytes)
        {
            throw new ArgumentOutOfRangeException(nameof(stride), stride, $"Stride must be at least width * {channels} = {rowBytes}.");
        }

        long needed = (long)stride * (height - 1) + rowBytes;
        if (pixels.Length < needed)
        {
            throw new ArgumentException($"A {width}x{height} image with stride {stride} needs at least {needed} bytes; got {pixels.Length}.", nameof(pixels));
        }

        long total = (long)width * height * BytesPerPixel;
        if (total > Array.MaxLength)
        {
            throw new ArgumentException($"A {width}x{height} image does not fit in one array.", nameof(width));
        }

        var rgb = new byte[total];
        for (int y = 0; y < height; y++)
        {
            var src = pixels.Slice(checked(y * stride), (int)rowBytes);
            var dst = rgb.AsSpan(y * width * BytesPerPixel, width * BytesPerPixel);
            if (channels == 3)
            {
                src.CopyTo(dst);
            }
            else
            {
                for (int x = 0, s = 0, d = 0; x < width; x++, s += 4, d += 3)
                {
                    dst[d] = src[s];
                    dst[d + 1] = src[s + 1];
                    dst[d + 2] = src[s + 2];
                }
            }
        }

        return new EmbeddingGemma2Image(width, height, rgb);
    }

    private static void ValidateSize(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
    }
}
