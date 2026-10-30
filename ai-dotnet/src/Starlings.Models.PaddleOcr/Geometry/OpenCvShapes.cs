// Ported from OpenCV 4.10.0 (https://github.com/opencv/opencv, modules/imgproc and modules/core), Copyright (C) 2000-2024
// the OpenCV contributors, licensed under the Apache License, Version 2.0. Modified: translated to C# and reduced to the
// cases PP-OCRv5 uses. See ai-dotnet/THIRD-PARTY-NOTICES.md and ai-dotnet/PROVENANCE.md.
namespace Starlings.Models.PaddleOcr.Geometry;

/// <summary>An OpenCV <c>RotatedRect</c>: center, size and angle in degrees, all single precision.</summary>
internal readonly record struct RotatedRect(float CenterX, float CenterY, float Width, float Height, float Angle);

/// <summary>
/// OpenCV 4.10 shape measures restated (adapted from OpenCV, Apache-2.0; see THIRD-PARTY-NOTICES.md):
/// <c>convexHull</c> (Sklansky), <c>minAreaRect</c> (rotating calipers), <c>boxPoints</c>, <c>contourArea</c> and
/// <c>arcLength</c>, with OpenCV's single- and double-precision steps.
/// </summary>
internal static class OpenCvShapes
{
    /// <summary><c>cv::minAreaRect</c> of integer points (x, y interleaved).</summary>
    public static RotatedRect MinAreaRect(ReadOnlySpan<int> points)
    {
        var hull = ConvexHull(ToDoubles(points), isFloat: false);
        return MinAreaRectOfHull(hull);
    }

    /// <summary><c>cv::minAreaRect</c> of float points (x, y interleaved).</summary>
    public static RotatedRect MinAreaRect(ReadOnlySpan<float> points)
    {
        var values = new double[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            values[i] = points[i];
        }

        var hull = ConvexHull(values, isFloat: true);
        return MinAreaRectOfHull(hull);
    }

    private static double[] ToDoubles(ReadOnlySpan<int> points)
    {
        var values = new double[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            values[i] = points[i];
        }

        return values;
    }

    private static RotatedRect MinAreaRectOfHull(float[] hull)
    {
        int n = hull.Length / 2;
        if (n > 2)
        {
            Span<float> o = stackalloc float[6];
            RotatingCalipers(hull, n, o);
            float cx = o[0] + (o[2] + o[4]) * 0.5f;
            float cy = o[1] + (o[3] + o[5]) * 0.5f;
            float width = (float)Math.Sqrt((double)o[2] * o[2] + (double)o[3] * o[3]);
            float height = (float)Math.Sqrt((double)o[4] * o[4] + (double)o[5] * o[5]);
            float angle = (float)Math.Atan2(o[3], o[2]);
            return new RotatedRect(cx, cy, width, height, (float)(angle * 180 / Math.PI));
        }

        if (n == 2)
        {
            float cx = (hull[0] + hull[2]) * 0.5f, cy = (hull[1] + hull[3]) * 0.5f;
            double dx = hull[2] - hull[0], dy = hull[3] - hull[1];
            float angle = (float)Math.Atan2(dy, dx);
            return new RotatedRect(cx, cy, (float)Math.Sqrt(dx * dx + dy * dy), 0, (float)(angle * 180 / Math.PI));
        }

        return n == 1 ? new RotatedRect(hull[0], hull[1], 0, 0, 0) : default;
    }

    /// <summary><c>cv::boxPoints</c> (<c>RotatedRect::points</c>): four corners, x and y interleaved.</summary>
    public static float[] BoxPoints(RotatedRect box)
    {
        double angle = box.Angle * Math.PI / 180.0;
        float b = (float)Math.Cos(angle) * 0.5f;
        float a = (float)Math.Sin(angle) * 0.5f;
        var p = new float[8];
        p[0] = box.CenterX - a * box.Height - b * box.Width;
        p[1] = box.CenterY + b * box.Height - a * box.Width;
        p[2] = box.CenterX + a * box.Height - b * box.Width;
        p[3] = box.CenterY - b * box.Height - a * box.Width;
        p[4] = 2 * box.CenterX - p[0];
        p[5] = 2 * box.CenterY - p[1];
        p[6] = 2 * box.CenterX - p[2];
        p[7] = 2 * box.CenterY - p[3];
        return p;
    }

    /// <summary><c>cv::contourArea(points, oriented=false)</c> of float points.</summary>
    public static double ContourArea(ReadOnlySpan<float> points)
    {
        int n = points.Length / 2;
        if (n < 3)
        {
            return 0;
        }

        double area = 0;
        float prevX = points[(n - 1) * 2], prevY = points[(n - 1) * 2 + 1];
        for (int i = 0; i < n; i++)
        {
            float x = points[i * 2], y = points[i * 2 + 1];
            area += (double)prevX * y - (double)prevY * x;
            prevX = x;
            prevY = y;
        }

        return Math.Abs(area * 0.5);
    }

    /// <summary><c>cv::arcLength(points, closed=true)</c> of float points.</summary>
    public static double ArcLength(ReadOnlySpan<float> points)
    {
        int n = points.Length / 2;
        double perimeter = 0;
        float prevX = points[(n - 1) * 2], prevY = points[(n - 1) * 2 + 1];
        for (int i = 0; i < n; i++)
        {
            float x = points[i * 2], y = points[i * 2 + 1];
            float dx = x - prevX, dy = y - prevY;
            perimeter += MathF.Sqrt(dx * dx + dy * dy);
            prevX = x;
            prevY = y;
        }

        return perimeter;
    }

    /// <summary>
    /// <c>cv::convexHull(points, clockwise=false, returnPoints=true)</c> (Sklansky), returned as float points in
    /// OpenCV's output order. Integer inputs use 64-bit cross products, float inputs double precision.
    /// </summary>
    internal static float[] ConvexHull(double[] coordinates, bool isFloat)
    {
        int total = coordinates.Length / 2;
        if (total == 0)
        {
            return [];
        }

        var order = new int[total];
        for (int i = 0; i < total; i++)
        {
            order[i] = i;
        }

        double X(int i) => coordinates[order[i] * 2];
        double Y(int i) => coordinates[order[i] * 2 + 1];
        Array.Sort(order, (a, b) =>
        {
            double ax = coordinates[a * 2], bx = coordinates[b * 2], ay = coordinates[a * 2 + 1], by = coordinates[b * 2 + 1];
            int c = ax.CompareTo(bx);
            return c != 0 ? c : ay.CompareTo(by);
        });

        var hull = new List<int>();
        if (X(0) == X(total - 1) && Y(0) == Y(total - 1))
        {
            hull.Add(0);
        }
        else
        {
            int minY = 0, maxY = 0;
            for (int i = 1; i < total; i++)
            {
                double y = Y(i);
                if (Y(minY) > y)
                {
                    minY = i;
                }

                if (Y(maxY) < y)
                {
                    maxY = i;
                }
            }

            var stack = new int[total + 2];
            var stack2 = new int[total + 2];
            int tlCount = Sklansky(X, Y, 0, maxY, stack, -1, 1, isFloat);
            int trCount = Sklansky(X, Y, total - 1, maxY, stack2, -1, -1, isFloat);
            int[] tl = stack, tr = stack2;
            // clockwise == false: swap the halves.
            (tl, tr) = (tr, tl);
            (tlCount, trCount) = (trCount, tlCount);
            for (int i = 0; i < tlCount - 1; i++)
            {
                hull.Add(tl[i]);
            }

            for (int i = trCount - 1; i > 0; i--)
            {
                hull.Add(tr[i]);
            }

            int stopIndex = trCount > 2 ? tr[1] : tlCount > 2 ? tl[tlCount - 2] : -1;

            var bottom = new int[total + 2];
            var bottom2 = new int[total + 2];
            int blCount = Sklansky(X, Y, 0, minY, bottom, 1, -1, isFloat);
            int brCount = Sklansky(X, Y, total - 1, minY, bottom2, 1, 1, isFloat);
            int[] bl = bottom, br = bottom2;
            if (stopIndex >= 0)
            {
                int checkIndex = blCount > 2 ? bl[1] : blCount + brCount > 3 ? br[2] : -1;
                if (checkIndex == stopIndex || (checkIndex >= 0 && X(checkIndex) == X(stopIndex) && Y(checkIndex) == Y(stopIndex)))
                {
                    blCount = Math.Min(blCount, 2);
                    brCount = Math.Min(brCount, 2);
                }
            }

            for (int i = 0; i < blCount - 1; i++)
            {
                hull.Add(bl[i]);
            }

            for (int i = brCount - 1; i > 0; i--)
            {
                hull.Add(br[i]);
            }
        }

        // Original point indices; OpenCV then rotates the cycle so the indices ascend or descend when they can.
        var indices = hull.Select(position => order[position]).ToArray();
        RotateToMonotonic(indices);
        var result = new float[indices.Length * 2];
        for (int i = 0; i < indices.Length; i++)
        {
            result[i * 2] = (float)coordinates[indices[i] * 2];
            result[i * 2 + 1] = (float)coordinates[indices[i] * 2 + 1];
        }

        return result;
    }

    /// <summary>
    /// OpenCV <c>convexHull</c>'s last step: "try to make the convex hull indices form an ascending or descending
    /// sequence by the cyclic shift of the output sequence".
    /// </summary>
    private static void RotateToMonotonic(int[] hull)
    {
        int nout = hull.Length;
        if (nout < 3)
        {
            return;
        }

        int minIndex = 0, maxIndex = 0, lt = 0;
        for (int i = 1; i < nout; i++)
        {
            int index = hull[i];
            lt += hull[i - 1] < index ? 1 : 0;
            if (lt > 1 && lt <= i - 2)
            {
                break;
            }

            if (index < hull[minIndex])
            {
                minIndex = i;
            }

            if (index > hull[maxIndex])
            {
                maxIndex = i;
            }
        }

        int distance = Math.Abs(maxIndex - minIndex);
        if ((distance == 1 || distance == nout - 1) && (lt <= 1 || lt >= nout - 2))
        {
            bool ascending = (maxIndex + 1) % nout == minIndex;
            int i0 = ascending ? minIndex : maxIndex, j = i0;
            if (i0 > 0)
            {
                var stack = new int[nout];
                int i;
                for (i = 0; i < nout; i++)
                {
                    int current = stack[i] = hull[j];
                    int nextJ = j + 1 < nout ? j + 1 : 0;
                    int next = hull[nextJ];
                    if (i < nout - 1 && ascending != (current < next))
                    {
                        break;
                    }

                    j = nextJ;
                }

                if (i == nout)
                {
                    stack.CopyTo(hull, 0);
                }
            }
        }
    }

    private static int Sklansky(Func<int, double> x, Func<int, double> y, int start, int end, int[] stack, int nsign, int sign2, bool isFloat)
    {
        int increment = end > start ? 1 : -1;
        int pprev = start, pcur = pprev + increment, pnext = pcur + increment;
        int size = 3;
        if (start == end || (x(start) == x(end) && y(start) == y(end)))
        {
            stack[0] = start;
            return 1;
        }

        stack[0] = pprev;
        stack[1] = pcur;
        stack[2] = pnext;
        end += increment;
        while (pnext != end)
        {
            double cury = y(pcur), nexty = y(pnext);
            double by = nexty - cury;
            if (Math.Sign(by) != nsign)
            {
                double ax = x(pcur) - x(pprev);
                double bx = x(pnext) - x(pcur);
                double ay = cury - y(pprev);
                double convexity;
                if (isFloat)
                {
                    // OpenCV computes the float differences in float, then the cross product in double.
                    float fax = (float)x(pcur) - (float)x(pprev), fbx = (float)x(pnext) - (float)x(pcur);
                    float fay = (float)cury - (float)y(pprev), fby = (float)nexty - (float)cury;
                    convexity = (double)fay * fbx - (double)fax * fby;
                    ax = fax;
                    ay = fay;
                }
                else
                {
                    convexity = (double)((long)ay * (long)bx - (long)ax * (long)by);
                }

                if (Math.Sign(convexity) == sign2 && (ax != 0 || ay != 0))
                {
                    pprev = pcur;
                    pcur = pnext;
                    pnext += increment;
                    stack[size] = pnext;
                    size++;
                }
                else if (pprev == start)
                {
                    pcur = pnext;
                    stack[1] = pcur;
                    pnext += increment;
                    stack[2] = pnext;
                }
                else
                {
                    stack[size - 2] = pnext;
                    pcur = pprev;
                    pprev = stack[size - 4];
                    size--;
                }
            }
            else
            {
                pnext += increment;
                stack[size - 1] = pnext;
            }
        }

        return --size;
    }

    /// <summary>OpenCV <c>rotatingCalipers(..., CALIPERS_MINAREARECT, out)</c>, single precision as upstream.</summary>
    private static void RotatingCalipers(float[] points, int n, Span<float> output)
    {
        float minArea = float.MaxValue;
        var invLength = new float[n];
        var vx = new float[n];
        var vy = new float[n];
        int left = 0, bottom = 0, right = 0, top = 0;
        Span<int> seq = stackalloc int[4];
        float orientation = 0;
        float baseA, baseB = 0;
        float p0x = points[0], p0y = points[1];
        float leftX = p0x, rightX = p0x, topY = p0y, bottomY = p0y;
        for (int i = 0; i < n; i++)
        {
            if (p0x < leftX)
            {
                leftX = p0x;
                left = i;
            }

            if (p0x > rightX)
            {
                rightX = p0x;
                right = i;
            }

            if (p0y > topY)
            {
                topY = p0y;
                top = i;
            }

            if (p0y < bottomY)
            {
                bottomY = p0y;
                bottom = i;
            }

            int next = i + 1 < n ? i + 1 : 0;
            float px = points[next * 2], py = points[next * 2 + 1];
            double dx = px - p0x, dy = py - p0y;
            vx[i] = (float)dx;
            vy[i] = (float)dy;
            invLength[i] = (float)(1.0 / Math.Sqrt(dx * dx + dy * dy));
            p0x = px;
            p0y = py;
        }

        {
            double ax = vx[n - 1], ay = vy[n - 1];
            for (int i = 0; i < n; i++)
            {
                double bx = vx[i], by = vy[i];
                double convexity = ax * by - ay * bx;
                if (convexity != 0)
                {
                    orientation = convexity > 0 ? 1f : -1f;
                    break;
                }

                ax = bx;
                ay = by;
            }

            if (orientation == 0)
            {
                throw new InvalidOperationException("Degenerate hull in minAreaRect.");
            }
        }

        baseA = orientation;
        seq[0] = bottom;
        seq[1] = right;
        seq[2] = top;
        seq[3] = left;
        int bestLeft = 0, bestBottom = 0;
        float bestA = 0, bestB = 0, bestWidth = 0, bestHeight = 0;
        Span<float> rx = stackalloc float[4];
        Span<float> ry = stackalloc float[4];
        for (int k = 0; k < n; k++)
        {
            // The caliper side whose edge turns least: rotate each side's edge into the base frame and keep the one
            // furthest clockwise (OpenCV's firstVecIsRight, single precision).
            (rx[0], ry[0]) = (vx[seq[0]], vy[seq[0]]);
            (rx[1], ry[1]) = (vy[seq[1]], -vx[seq[1]]);
            (rx[2], ry[2]) = (-vx[seq[2]], -vy[seq[2]]);
            (rx[3], ry[3]) = (-vy[seq[3]], vx[seq[3]]);
            int main = 0;
            for (int i = 1; i < 4; i++)
            {
                // rotate90CW(v1) . v2 < 0
                if (ry[i] * rx[main] + -rx[i] * ry[main] < 0)
                {
                    main = i;
                }
            }

            int pindex = seq[main];
            float leadX = vx[pindex] * invLength[pindex];
            float leadY = vy[pindex] * invLength[pindex];
            switch (main)
            {
                case 0:
                    baseA = leadX;
                    baseB = leadY;
                    break;
                case 1:
                    baseA = leadY;
                    baseB = -leadX;
                    break;
                case 2:
                    baseA = -leadX;
                    baseB = -leadY;
                    break;
                default:
                    baseA = -leadY;
                    baseB = leadX;
                    break;
            }

            seq[main] += 1;
            seq[main] = seq[main] == n ? 0 : seq[main];

            float ddx = points[seq[1] * 2] - points[seq[3] * 2];
            float ddy = points[seq[1] * 2 + 1] - points[seq[3] * 2 + 1];
            float width = ddx * baseA + ddy * baseB;
            ddx = points[seq[2] * 2] - points[seq[0] * 2];
            ddy = points[seq[2] * 2 + 1] - points[seq[0] * 2 + 1];
            float height = -ddx * baseB + ddy * baseA;
            float area = width * height;
            if (area <= minArea)
            {
                minArea = area;
                bestLeft = seq[3];
                bestA = baseA;
                bestWidth = width;
                bestB = baseB;
                bestHeight = height;
                bestBottom = seq[0];
            }
        }

        float a1 = bestA, b1 = bestB, a2 = -bestB, b2 = bestA;
        float c1 = a1 * points[bestLeft * 2] + points[bestLeft * 2 + 1] * b1;
        float c2 = a2 * points[bestBottom * 2] + points[bestBottom * 2 + 1] * b2;
        float idet = 1f / (a1 * b2 - a2 * b1);
        output[0] = (c1 * b2 - c2 * b1) * idet;
        output[1] = (a1 * c2 - a2 * c1) * idet;
        output[2] = a1 * bestWidth;
        output[3] = b1 * bestWidth;
        output[4] = a2 * bestHeight;
        output[5] = b2 * bestHeight;
    }
}
