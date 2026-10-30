// Ported from OpenCV at 71d3237a093b60a27601c20e9ee6c3e52154e8b1 (tag 4.10.0; modules/imgproc, modules/core), Copyright (C) 2000-2024
// the OpenCV contributors, licensed under the Apache License, Version 2.0. Modified: translated to C# and reduced to the
// cases PP-OCRv5 uses. See ai-dotnet/THIRD-PARTY-NOTICES.md and ai-dotnet/PROVENANCE.md.
namespace Starlings.Models.PaddleOcr.Geometry;

/// <summary>
/// OpenCV 4.10 <c>cv::resize</c> with <c>INTER_LINEAR</c> on 8-bit images, restated bit for bit (adapted from OpenCV,
/// Apache-2.0; see THIRD-PARTY-NOTICES.md): coefficients <c>fx = (float)((dx + 0.5) * scale - 0.5)</c> quantized to
/// 11 bits, a horizontal pass in integers, and the vertical pass of OpenCV's SIMD kernel
/// (<c>VResizeLinearVec_32s8u</c>: each row term shifted right by 4, multiplied as 16-bit high halves, then rounded by
/// <c>(sum + 2) &gt;&gt; 2</c>). Columns past the last source column use the edge pixel; rows outside the source repeat the
/// edge row with the unclamped vertical weights, as OpenCV does.
/// </summary>
internal static class OpenCvResize
{
    private const int CoefficientBits = 11;
    private const int CoefficientScale = 1 << CoefficientBits;

    /// <summary>Resizes interleaved 8-bit pixels (<paramref name="channels"/> per pixel, row-major, no padding).</summary>
    public static byte[] Linear(byte[] source, int height, int width, int channels, int targetHeight, int targetWidth)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(channels, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(targetHeight, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(targetWidth, 1);
        if (source.Length != checked(height * width * channels))
        {
            throw new ArgumentException("Pixel buffer size does not match the image size.", nameof(source));
        }

        var (xOffsets, xAlpha, xMax) = Coefficients(targetWidth, width, clampEdges: true);
        var (yOffsets, yBeta, _) = Coefficients(targetHeight, height, clampEdges: false);

        // Horizontal pass, one source row at a time (cached: consecutive target rows mostly share source rows).
        int rowLength = targetWidth * channels;
        var rows = new Dictionary<int, int[]>();
        int[] HorizontalRow(int sy)
        {
            if (rows.TryGetValue(sy, out var cached))
            {
                return cached;
            }

            var row = new int[rowLength];
            var src = source.AsSpan(sy * width * channels, width * channels);
            for (int dx = 0; dx < targetWidth; dx++)
            {
                int sx = xOffsets[dx];
                int a0 = xAlpha[dx * 2], a1 = xAlpha[dx * 2 + 1];
                int next = Math.Min(sx + 1, width - 1);
                for (int c = 0; c < channels; c++)
                {
                    row[dx * channels + c] = dx < xMax
                        ? src[sx * channels + c] * a0 + src[next * channels + c] * a1
                        : src[sx * channels + c] * CoefficientScale;
                }
            }

            if (rows.Count > 4)
            {
                rows.Clear();
            }

            rows[sy] = row;
            return row;
        }

        var output = new byte[targetHeight * rowLength];
        for (int dy = 0; dy < targetHeight; dy++)
        {
            int sy = yOffsets[dy];
            var s0 = HorizontalRow(Math.Clamp(sy, 0, height - 1));
            var s1 = HorizontalRow(Math.Clamp(sy + 1, 0, height - 1));
            int b0 = yBeta[dy * 2], b1 = yBeta[dy * 2 + 1];
            int offset = dy * rowLength;
            for (int i = 0; i < rowLength; i++)
            {
                int value = (((b0 * (s0[i] >> 4)) >> 16) + ((b1 * (s1[i] >> 4)) >> 16) + 2) >> 2;
                output[offset + i] = (byte)Math.Clamp(value, 0, 255);
            }
        }

        return output;
    }

    /// <summary>
    /// Source offsets and 11-bit weights per target index. With <paramref name="clampEdges"/> (the horizontal axis) an
    /// offset before the first or at/after the last source index is clamped with weight (1, 0), and the first index
    /// whose right neighbour would leave the source is returned as the end of the interpolated range.
    /// </summary>
    private static (int[] Offsets, int[] Weights, int End) Coefficients(int targetSize, int sourceSize, bool clampEdges)
    {
        double scale = 1.0 / ((double)targetSize / sourceSize);
        var offsets = new int[targetSize];
        var weights = new int[targetSize * 2];
        int end = targetSize;
        for (int d = 0; d < targetSize; d++)
        {
            float f = (float)((d + 0.5) * scale - 0.5);
            int s = (int)MathF.Floor(f);
            f -= s;
            if (clampEdges)
            {
                if (s < 0)
                {
                    f = 0;
                    s = 0;
                }

                if (s + 1 >= sourceSize)
                {
                    end = Math.Min(end, d);
                    if (s >= sourceSize - 1)
                    {
                        f = 0;
                        s = sourceSize - 1;
                    }
                }
            }

            offsets[d] = s;
            weights[d * 2] = SaturateShort((1f - f) * CoefficientScale);
            weights[d * 2 + 1] = SaturateShort(f * CoefficientScale);
        }

        return (offsets, weights, end);
    }

    /// <summary>OpenCV <c>saturate_cast&lt;short&gt;(float)</c>: round to nearest, ties to even, then clamp.</summary>
    internal static int SaturateShort(float value) => (int)Math.Clamp(MathF.Round(value, MidpointRounding.ToEven), short.MinValue, short.MaxValue);
}
