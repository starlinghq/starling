namespace Starlings.Models.PaddleOcr;

/// <summary>
/// A decoded 8-bit image as PP-OCRv5 reads it: tightly packed BGR, three bytes per pixel, row-major (the channel order
/// of the pinned <c>inference.yml</c>, <c>DecodeImage img_mode: BGR</c>). Build it from RGB, BGR or RGBA pixels; the
/// package decodes no file format (see the README for decoding with SkiaSharp). Each side is at most 32,767 pixels
/// and the image at most 64 Mpx (<see cref="PaddleOcrImageTooLargeException"/>).
/// </summary>
public sealed class PaddleOcrImage
{
    private PaddleOcrImage(int width, int height, byte[] bgr)
    {
        Width = width;
        Height = height;
        Bgr = bgr;
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>The pixels: <c>Width * Height * 3</c> bytes, blue, green, red.</summary>
    internal byte[] Bgr { get; }

    /// <summary>Copies tightly packed RGB pixels (<c>width * height * 3</c> bytes).</summary>
    public static PaddleOcrImage FromRgb24(int width, int height, ReadOnlySpan<byte> rgb) => FromPacked(width, height, rgb, 3, swap: true);

    /// <summary>Copies tightly packed BGR pixels (<c>width * height * 3</c> bytes), as OpenCV holds them.</summary>
    public static PaddleOcrImage FromBgr24(int width, int height, ReadOnlySpan<byte> bgr) => FromPacked(width, height, bgr, 3, swap: false);

    /// <summary>
    /// Copies RGBA pixels with <paramref name="rowBytes"/> bytes per row (default <c>width * 4</c>). Alpha is dropped,
    /// not composited, as OpenCV's <c>imread(IMREAD_COLOR)</c> drops it.
    /// </summary>
    public static PaddleOcrImage FromRgba32(int width, int height, ReadOnlySpan<byte> rgba, int rowBytes = 0)
    {
        Validate(width, height);
        rowBytes = rowBytes == 0 ? width * 4 : rowBytes;
        ArgumentOutOfRangeException.ThrowIfLessThan(rowBytes, width * 4);
        if (rgba.Length < (long)rowBytes * (height - 1) + width * 4)
        {
            throw new ArgumentException("The RGBA buffer is smaller than the image.", nameof(rgba));
        }

        var bgr = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            var row = rgba.Slice(y * rowBytes, width * 4);
            for (int x = 0; x < width; x++)
            {
                int d = (y * width + x) * 3;
                bgr[d] = row[x * 4 + 2];
                bgr[d + 1] = row[x * 4 + 1];
                bgr[d + 2] = row[x * 4];
            }
        }

        return new PaddleOcrImage(width, height, bgr);
    }

    private static PaddleOcrImage FromPacked(int width, int height, ReadOnlySpan<byte> pixels, int channels, bool swap)
    {
        Validate(width, height);
        if (pixels.Length != (long)width * height * channels)
        {
            throw new ArgumentException($"Expected {(long)width * height * channels} bytes for {width}x{height}, got {pixels.Length}.", nameof(pixels));
        }

        var bgr = pixels.ToArray();
        if (swap)
        {
            for (int i = 0; i < bgr.Length; i += 3)
            {
                (bgr[i], bgr[i + 2]) = (bgr[i + 2], bgr[i]);
            }
        }

        return new PaddleOcrImage(width, height, bgr);
    }

    private static void Validate(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        if (width > PaddleOcrContract.MaxSide || height > PaddleOcrContract.MaxSide || (long)width * height > PaddleOcrContract.MaxPixels)
        {
            throw new PaddleOcrImageTooLargeException(width, height);
        }
    }
}
