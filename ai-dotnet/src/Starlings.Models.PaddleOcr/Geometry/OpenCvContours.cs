// Ported from OpenCV at 71d3237a093b60a27601c20e9ee6c3e52154e8b1 (tag 4.10.0; modules/imgproc, modules/core), Copyright (C) 2000-2024
// the OpenCV contributors, licensed under the Apache License, Version 2.0. Modified: translated to C# and reduced to the
// cases PP-OCRv5 uses. See ai-dotnet/THIRD-PARTY-NOTICES.md and ai-dotnet/PROVENANCE.md.
namespace Starlings.Models.PaddleOcr.Geometry;

/// <summary>
/// OpenCV 4.10 <c>cv::findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)</c> on a binary map, restated (adapted from
/// OpenCV, Apache-2.0; see THIRD-PARTY-NOTICES.md): Suzuki-Abe border following with 8-connectivity over a zero-padded
/// copy, outer and hole borders alike, each contour keeping only the points where the chain direction changes. Contours
/// are returned in OpenCV's order (the reverse of their discovery in raster order).
/// </summary>
internal static class OpenCvContours
{
    // Chain-code steps 0..7: right, up-right, up, up-left, left, down-left, down, down-right.
    private static readonly int[] StepX = [1, 1, 0, -1, -1, -1, 0, 1];
    private static readonly int[] StepY = [0, -1, -1, -1, 0, 1, 1, 1];

    /// <summary>Contours of the nonzero pixels of a <paramref name="height"/> x <paramref name="width"/> map, as (x, y) points.</summary>
    public static List<int[]> Find(ReadOnlySpan<bool> mask, int height, int width)
    {
        if (mask.Length != checked(height * width))
        {
            throw new ArgumentException("Mask size does not match.", nameof(mask));
        }

        int stride = width + 2;
        var image = new sbyte[(height + 2) * stride];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                image[(y + 1) * stride + x + 1] = mask[y * width + x] ? (sbyte)1 : (sbyte)0;
            }
        }

        Span<int> deltas = stackalloc int[16];
        for (int s = 0; s < 8; s++)
        {
            deltas[s] = deltas[s + 8] = StepY[s] * stride + StepX[s];
        }

        var found = new List<int[]>();
        for (int y = 1; y <= height; y++)
        {
            int prev = 0;
            for (int x = 1; x <= width; x++)
            {
                int p = image[y * stride + x];
                if (p != prev)
                {
                    bool isHole = false;
                    bool start = prev == 0 && p == 1;
                    if (!start && p == 0 && prev >= 1)
                    {
                        isHole = true;
                        start = true;
                    }

                    if (start)
                    {
                        int ox = x - (isHole ? 1 : 0);
                        found.Add(Trace(image, stride, deltas, ox, y, isHole));
                        // Tracing may have marked the current pixel; the scan continues from its new value.
                        p = image[y * stride + x];
                    }
                }

                prev = p;
            }
        }

        found.Reverse();
        return found;
    }

    /// <summary>OpenCV <c>icvFetchContour</c> with CHAIN_APPROX_SIMPLE, marks 2 and 2 | -128 (= -126).</summary>
    private static int[] Trace(sbyte[] image, int stride, ReadOnlySpan<int> deltas, int x, int y, bool isHole)
    {
        const sbyte Mark = 2;
        const sbyte RightMark = unchecked((sbyte)(2 | -128));
        var points = new List<int>();
        int i0 = y * stride + x;
        int px = x - 1, py = y - 1;
        int sEnd = isHole ? 0 : 4;
        int s = sEnd;
        int i1;
        do
        {
            s = (s - 1) & 7;
            i1 = i0 + deltas[s];
        }
        while (image[i1] == 0 && s != sEnd);

        if (s == sEnd)
        {
            image[i0] = RightMark;
            return [px, py];
        }

        int i3 = i0;
        int prevS = s ^ 4;
        while (true)
        {
            sEnd = s;
            int i4 = 0;
            while (s < 15)
            {
                i4 = i3 + deltas[++s];
                if (image[i4] != 0)
                {
                    break;
                }
            }

            s &= 7;
            if ((uint)(s - 1) < (uint)sEnd)
            {
                image[i3] = RightMark;
            }
            else if (image[i3] == 1)
            {
                image[i3] = Mark;
            }

            if (s != prevS)
            {
                points.Add(px);
                points.Add(py);
                prevS = s;
            }

            px += StepX[s];
            py += StepY[s];
            if (i4 == i0 && i3 == i1)
            {
                break;
            }

            i3 = i4;
            s = (s + 4) & 7;
        }

        return [.. points];
    }
}
