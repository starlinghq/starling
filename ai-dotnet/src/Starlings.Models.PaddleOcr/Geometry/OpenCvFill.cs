// Ported from OpenCV 4.10.0 (https://github.com/opencv/opencv, modules/imgproc and modules/core), Copyright (C) 2000-2024
// the OpenCV contributors, licensed under the Apache License, Version 2.0. Modified: translated to C# and reduced to the
// cases PP-OCRv5 uses. See ai-dotnet/THIRD-PARTY-NOTICES.md and ai-dotnet/PROVENANCE.md.
namespace Starlings.Models.PaddleOcr.Geometry;

/// <summary>
/// OpenCV 4.10 <c>cv::fillPoly</c> of one integer polygon with <c>LINE_8</c> and no shift, restated (adapted from
/// OpenCV, Apache-2.0; see THIRD-PARTY-NOTICES.md): the outline drawn with the 8-connected line iterator, then the
/// interior filled by the active-edge scanline algorithm in 16.16 fixed point (<c>CollectPolyEdges</c> and
/// <c>FillEdgeCollection</c>).
/// </summary>
internal static class OpenCvFill
{
    private const int Shift = 16;
    private const long One = 1L << Shift;

    private sealed class Edge
    {
        public int Y0;
        public int Y1;
        public long X;
        public long Dx;
        public Edge? Next;
    }

    /// <summary>Sets the pixels of the polygon (x, y interleaved) in a <paramref name="height"/> x <paramref name="width"/> mask.</summary>
    public static void FillPolygon(Span<bool> mask, int height, int width, ReadOnlySpan<int> points)
    {
        int count = points.Length / 2;
        var edges = new List<Edge>(count);
        long p0x = (long)points[(count - 1) * 2] << Shift, p0y = points[(count - 1) * 2 + 1];
        for (int i = 0; i < count; i++)
        {
            long p1x = (long)points[i * 2] << Shift, p1y = points[i * 2 + 1];
            long t0x = (p0x + (One >> 1)) >> Shift, t1x = (p1x + (One >> 1)) >> Shift, t0y = p0y, t1y = p1y;
            Line(mask, height, width, (int)t0x, (int)t0y, (int)t1x, (int)t1y);

            // CollectPolyEdges: an edge leaving the image runs between its clipped endpoints; any other is moved half a
            // pixel right (rounding the scanline crossings).
            long c0x = p0x, c0y = p0y, c1x = p1x, c1y = p1y;
            if ((ulong)t0x >= (ulong)width || (ulong)t0y >= (ulong)height || (ulong)t1x >= (ulong)width || (ulong)t1y >= (ulong)height)
            {
                ClipLine(width, height, ref t0x, ref t0y, ref t1x, ref t1y);
                if (t0y != t1y)
                {
                    (c0x, c0y, c1x, c1y) = (t0x << Shift, t0y, t1x << Shift, t1y);
                }
            }
            else
            {
                c0x += One >> 1;
                c1x += One >> 1;
            }

            if (p0y != p1y)
            {
                long dx = (c1x - c0x) / (c1y - c0y);
                var edge = p0y < p1y
                    ? new Edge { Y0 = (int)p0y, Y1 = (int)p1y, X = c0x + (p0y - c0y) * dx, Dx = dx }
                    : new Edge { Y0 = (int)p1y, Y1 = (int)p0y, X = c1x + (p1y - c1y) * dx, Dx = dx };
                edges.Add(edge);
            }

            p0x = p1x;
            p0y = p1y;
        }

        FillEdges(mask, height, width, edges);
    }

    /// <summary>OpenCV <c>Line</c> with 8-connectivity (<c>LineIterator</c>, left to right), clipped to the image.</summary>
    private static void Line(Span<bool> mask, int height, int width, int x1, int y1, int x2, int y2)
    {
        if ((uint)x1 >= (uint)width || (uint)y1 >= (uint)height || (uint)x2 >= (uint)width || (uint)y2 >= (uint)height)
        {
            long lx1 = x1, ly1 = y1, lx2 = x2, ly2 = y2;
            if (!ClipLine(width, height, ref lx1, ref ly1, ref lx2, ref ly2))
            {
                return;
            }

            (x1, y1, x2, y2) = ((int)lx1, (int)ly1, (int)lx2, (int)ly2);
        }

        if (x1 > x2)
        {
            (x1, x2) = (x2, x1);
            (y1, y2) = (y2, y1);
        }

        int dx = x2 - x1, dy = y2 - y1;
        int stepY = dy < 0 ? -1 : 1;
        dy = Math.Abs(dy);
        bool steep = dy > dx;
        int major = steep ? dy : dx, minor = steep ? dx : dy;
        int err = major - (minor + minor);
        int plusDelta = major + major, minusDelta = -(minor + minor);
        int x = x1, y = y1;
        for (int i = 0; i <= major; i++)
        {
            if ((uint)x < (uint)width && (uint)y < (uint)height)
            {
                mask[y * width + x] = true;
            }

            int maskBit = err < 0 ? -1 : 0;
            err += minusDelta + (plusDelta & maskBit);
            if (steep)
            {
                y += stepY;
                if (maskBit != 0)
                {
                    x += 1;
                }
            }
            else
            {
                x += 1;
                if (maskBit != 0)
                {
                    y += stepY;
                }
            }
        }
    }

    /// <summary>OpenCV <c>clipLine</c>: clips the segment to the image, moving the endpoints along it (integer steps).</summary>
    private static bool ClipLine(int width, int height, ref long x1, ref long y1, ref long x2, ref long y2)
    {
        long right = width - 1, bottom = height - 1;
        int c1 = (x1 < 0 ? 1 : 0) + (x1 > right ? 2 : 0) + (y1 < 0 ? 4 : 0) + (y1 > bottom ? 8 : 0);
        int c2 = (x2 < 0 ? 1 : 0) + (x2 > right ? 2 : 0) + (y2 < 0 ? 4 : 0) + (y2 > bottom ? 8 : 0);
        if ((c1 & c2) == 0 && (c1 | c2) != 0)
        {
            long a;
            if ((c1 & 12) != 0)
            {
                a = c1 < 8 ? 0 : bottom;
                x1 += (long)((double)(a - y1) * (x2 - x1) / (y2 - y1));
                y1 = a;
                c1 = (x1 < 0 ? 1 : 0) + (x1 > right ? 2 : 0);
            }

            if ((c2 & 12) != 0)
            {
                a = c2 < 8 ? 0 : bottom;
                x2 += (long)((double)(a - y2) * (x2 - x1) / (y2 - y1));
                y2 = a;
                c2 = (x2 < 0 ? 1 : 0) + (x2 > right ? 2 : 0);
            }

            if ((c1 & c2) == 0 && (c1 | c2) != 0)
            {
                if (c1 != 0)
                {
                    a = c1 == 1 ? 0 : right;
                    y1 += (long)((double)(a - x1) * (y2 - y1) / (x2 - x1));
                    x1 = a;
                    c1 = 0;
                }

                if (c2 != 0)
                {
                    a = c2 == 1 ? 0 : right;
                    y2 += (long)((double)(a - x2) * (y2 - y1) / (x2 - x1));
                    x2 = a;
                    c2 = 0;
                }
            }
        }

        return (c1 | c2) == 0;
    }

    private static void FillEdges(Span<bool> mask, int height, int width, List<Edge> edges)
    {
        int total = edges.Count;
        if (total < 2)
        {
            return;
        }

        int yMax = int.MinValue, yMin = int.MaxValue;
        long xMax = long.MinValue, xMin = long.MaxValue;
        foreach (var e in edges)
        {
            long x1 = e.X + (e.Y1 - e.Y0) * e.Dx;
            yMin = Math.Min(yMin, e.Y0);
            yMax = Math.Max(yMax, e.Y1);
            xMin = Math.Min(xMin, Math.Min(e.X, x1));
            xMax = Math.Max(xMax, Math.Max(e.X, x1));
        }

        if (yMax < 0 || yMin >= height || xMax < 0 || xMin >= ((long)width << Shift))
        {
            return;
        }

        edges.Sort((a, b) => a.Y0 != b.Y0 ? a.Y0.CompareTo(b.Y0) : a.X != b.X ? a.X.CompareTo(b.X) : a.Dx.CompareTo(b.Dx));
        edges.Add(new Edge { Y0 = int.MaxValue });
        var head = new Edge();
        int index = 0;
        var e0 = edges[0];
        yMax = Math.Min(yMax, height);
        for (int y = e0.Y0; y < yMax; y++)
        {
            bool draw = false;
            bool clip = y < 0;
            Edge prelast = head;
            Edge? last = head.Next;
            while (last is not null || e0.Y0 == y)
            {
                if (last is not null && last.Y1 == y)
                {
                    prelast.Next = last.Next;
                    last = last.Next;
                    continue;
                }

                Edge keepPrelast = prelast;
                if (last is not null && (e0.Y0 > y || last.X < e0.X))
                {
                    prelast = last;
                    last = last.Next;
                }
                else if (index < total)
                {
                    prelast.Next = e0;
                    e0.Next = last;
                    prelast = e0;
                    e0 = edges[++index];
                }
                else
                {
                    break;
                }

                if (draw)
                {
                    if (!clip)
                    {
                        int x1, x2;
                        if (keepPrelast.X > prelast.X)
                        {
                            x1 = (int)(prelast.X >> Shift);
                            x2 = (int)(keepPrelast.X >> Shift);
                        }
                        else
                        {
                            x1 = (int)(keepPrelast.X >> Shift);
                            x2 = (int)(prelast.X >> Shift);
                        }

                        if (x1 < width && x2 >= 0)
                        {
                            x1 = Math.Max(x1, 0);
                            x2 = Math.Min(x2, width - 1);
                            mask.Slice(y * width + x1, x2 - x1 + 1).Fill(true);
                        }
                    }

                    keepPrelast.X += keepPrelast.Dx;
                    prelast.X += prelast.Dx;
                }

                draw = !draw;
            }

            // Bubble-sort the active edges by x.
            Edge? keep = null;
            do
            {
                prelast = head;
                last = head.Next;
                Edge? lastExchange = null;
                while (last != keep && last!.Next is not null)
                {
                    var te = last.Next;
                    if (last.X > te.X)
                    {
                        prelast.Next = te;
                        last.Next = te.Next;
                        te.Next = last;
                        prelast = te;
                        lastExchange = prelast;
                    }
                    else
                    {
                        prelast = last;
                        last = te;
                    }
                }

                if (lastExchange is null)
                {
                    break;
                }

                keep = lastExchange;
            }
            while (keep != head.Next && keep != head);
        }
    }
}
