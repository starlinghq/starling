namespace Starlings.Models.PaddleOcr;

/// <summary>A point of a detected text box, in pixels of the input image.</summary>
/// <param name="X">Column.</param>
/// <param name="Y">Row.</param>
public readonly record struct TextPoint(int X, int Y);

/// <summary>
/// The quadrilateral of a detected text line, in pixels of the input image: top-left, top-right, bottom-right,
/// bottom-left (PaddleOCR's point order for DB boxes).
/// </summary>
/// <param name="TopLeft">First corner.</param>
/// <param name="TopRight">Second corner.</param>
/// <param name="BottomRight">Third corner.</param>
/// <param name="BottomLeft">Fourth corner.</param>
public readonly record struct TextQuad(TextPoint TopLeft, TextPoint TopRight, TextPoint BottomRight, TextPoint BottomLeft)
{
    /// <summary>The corners in order.</summary>
    public IReadOnlyList<TextPoint> Points => [TopLeft, TopRight, BottomRight, BottomLeft];

    /// <summary>The axis-aligned bounds: left, top, right, bottom.</summary>
    public (int Left, int Top, int Right, int Bottom) Bounds => (
        Math.Min(Math.Min(TopLeft.X, TopRight.X), Math.Min(BottomRight.X, BottomLeft.X)),
        Math.Min(Math.Min(TopLeft.Y, TopRight.Y), Math.Min(BottomRight.Y, BottomLeft.Y)),
        Math.Max(Math.Max(TopLeft.X, TopRight.X), Math.Max(BottomRight.X, BottomLeft.X)),
        Math.Max(Math.Max(TopLeft.Y, TopRight.Y), Math.Max(BottomRight.Y, BottomLeft.Y)));

    internal static TextQuad From(ReadOnlySpan<int> xy) =>
        new(new TextPoint(xy[0], xy[1]), new TextPoint(xy[2], xy[3]), new TextPoint(xy[4], xy[5]), new TextPoint(xy[6], xy[7]));

    internal int[] ToArray() => [TopLeft.X, TopLeft.Y, TopRight.X, TopRight.Y, BottomRight.X, BottomRight.Y, BottomLeft.X, BottomLeft.Y];
}

/// <summary>One recognized text line.</summary>
/// <param name="Box">Where the line is, in pixels of the input image.</param>
/// <param name="Text">The recognized text (greedy CTC decoding).</param>
/// <param name="Confidence">Mean of the per-character maximum probabilities, 0 to 1 (0 when no character was decoded).</param>
/// <param name="DetectionScore">The DB box score: mean detector probability inside the box, 0 to 1.</param>
public sealed record TextLine(TextQuad Box, string Text, float Confidence, float DetectionScore);

/// <summary>
/// The image is larger than the package accepts: a side above 32,767 pixels (box corners are int16 upstream, so they
/// would wrap) or more than 64 Mpx in total.
/// </summary>
public sealed class PaddleOcrImageTooLargeException : ArgumentOutOfRangeException
{
    /// <summary>Creates the exception.</summary>
    public PaddleOcrImageTooLargeException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public PaddleOcrImageTooLargeException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and inner exception.</summary>
    public PaddleOcrImageTooLargeException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Creates the exception for an image size.</summary>
    public PaddleOcrImageTooLargeException(int width, int height)
        : base("image", $"The image is {width} x {height}; each side may be at most {PaddleOcrContract.MaxSide} pixels and the image at most {PaddleOcrContract.MaxPixels} pixels.")
    {
        Width = width;
        Height = height;
    }

    /// <summary>Requested width.</summary>
    public int Width { get; }

    /// <summary>Requested height.</summary>
    public int Height { get; }
}

/// <summary>A pinned configuration file differs from the contract this package implements.</summary>
public sealed class PaddleOcrConfigurationException : FormatException
{
    /// <summary>Creates the exception.</summary>
    public PaddleOcrConfigurationException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public PaddleOcrConfigurationException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and inner exception.</summary>
    public PaddleOcrConfigurationException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
