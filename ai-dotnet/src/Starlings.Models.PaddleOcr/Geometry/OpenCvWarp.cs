// Ported from OpenCV 4.10.0 (https://github.com/opencv/opencv, modules/imgproc and modules/core), Copyright (C) 2000-2024
// the OpenCV contributors, licensed under the Apache License, Version 2.0. Modified: translated to C# and reduced to the
// cases PP-OCRv5 uses. See ai-dotnet/THIRD-PARTY-NOTICES.md and ai-dotnet/PROVENANCE.md.
namespace Starlings.Models.PaddleOcr.Geometry;

/// <summary>
/// OpenCV 4.10 <c>cv::getPerspectiveTransform</c> (8x8 system solved by LU with partial pivoting) and
/// <c>cv::warpPerspective</c> with <c>INTER_CUBIC</c> and <c>BORDER_REPLICATE</c> on 8-bit images, restated bit for bit
/// (adapted from OpenCV, Apache-2.0; see THIRD-PARTY-NOTICES.md): the matrix is inverted with OpenCV's 3x3 cofactor
/// formula, source coordinates are computed per 32x32-element block and quantized to 1/32 pixel, and the 4x4 bicubic
/// kernel (A = -0.75) uses OpenCV's 15-bit fixed-point weight table.
/// </summary>
internal static class OpenCvWarp
{
    private const int TabBits = 5;
    private const int TabSize = 1 << TabBits;
    private const int CoefficientBits = 15;
    private const int CoefficientScale = 1 << CoefficientBits;
    private const int BlockSize = 32;

    private static readonly int[] CubicTable = BuildCubicTable();

    /// <summary><c>cv::getPerspectiveTransform(src, dst)</c>: the 3x3 matrix (row-major) mapping source to target points.</summary>
    public static double[] PerspectiveTransform(ReadOnlySpan<float> source, ReadOnlySpan<float> target)
    {
        if (source.Length != 8 || target.Length != 8)
        {
            throw new ArgumentException("Four (x, y) points are required.");
        }

        var a = new double[64];
        var b = new double[8];
        for (int i = 0; i < 4; i++)
        {
            float sx = source[i * 2], sy = source[i * 2 + 1], tx = target[i * 2], ty = target[i * 2 + 1];
            a[i * 8 + 0] = a[(i + 4) * 8 + 3] = sx;
            a[i * 8 + 1] = a[(i + 4) * 8 + 4] = sy;
            a[i * 8 + 2] = a[(i + 4) * 8 + 5] = 1;
            // OpenCV multiplies the float coordinates in float before storing the double.
            a[i * 8 + 6] = -sx * tx;
            a[i * 8 + 7] = -sy * tx;
            a[(i + 4) * 8 + 6] = -sx * ty;
            a[(i + 4) * 8 + 7] = -sy * ty;
            b[i] = tx;
            b[i + 4] = ty;
        }

        if (!SolveLu(a, b, 8))
        {
            throw new ArgumentException("The points are degenerate: no perspective transform maps them.");
        }

        return [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1.0];
    }

    /// <summary>OpenCV's <c>LUImpl</c> for one right-hand side: Gaussian elimination with partial pivoting.</summary>
    private static bool SolveLu(double[] a, double[] b, int m)
    {
        const double Epsilon = 2.220446049250313e-16 * 10; // DBL_EPSILON * 10, as cv::solve passes
        for (int i = 0; i < m; i++)
        {
            int k = i;
            for (int j = i + 1; j < m; j++)
            {
                if (Math.Abs(a[j * m + i]) > Math.Abs(a[k * m + i]))
                {
                    k = j;
                }
            }

            if (Math.Abs(a[k * m + i]) < Epsilon)
            {
                return false;
            }

            if (k != i)
            {
                for (int j = i; j < m; j++)
                {
                    (a[i * m + j], a[k * m + j]) = (a[k * m + j], a[i * m + j]);
                }

                (b[i], b[k]) = (b[k], b[i]);
            }

            double d = -1 / a[i * m + i];
            for (int j = i + 1; j < m; j++)
            {
                double alpha = a[j * m + i] * d;
                for (k = i + 1; k < m; k++)
                {
                    a[j * m + k] += alpha * a[i * m + k];
                }

                b[j] += alpha * b[i];
            }

            // OpenCV keeps the pivot's reciprocal on the diagonal and multiplies by it during back substitution.
            a[i * m + i] = -d;
        }

        for (int i = m - 1; i >= 0; i--)
        {
            double s = b[i];
            for (int k = i + 1; k < m; k++)
            {
                s -= a[i * m + k] * b[k];
            }

            b[i] = s * a[i * m + i];
        }

        return true;
    }

    /// <summary><c>cv::invert</c> of a 3x3 double matrix (DECOMP_LU takes OpenCV's cofactor path for n = 3).</summary>
    internal static double[] Invert3(ReadOnlySpan<double> m)
    {
        double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
        if (det == 0)
        {
            throw new ArgumentException("The perspective transform is singular.");
        }

        double d = 1.0 / det;
        return
        [
            (m[4] * m[8] - m[5] * m[7]) * d,
            (m[2] * m[7] - m[1] * m[8]) * d,
            (m[1] * m[5] - m[2] * m[4]) * d,
            (m[5] * m[6] - m[3] * m[8]) * d,
            (m[0] * m[8] - m[2] * m[6]) * d,
            (m[2] * m[3] - m[0] * m[5]) * d,
            (m[3] * m[7] - m[4] * m[6]) * d,
            (m[1] * m[6] - m[0] * m[7]) * d,
            (m[0] * m[4] - m[1] * m[3]) * d,
        ];
    }

    /// <summary>
    /// <c>cv::warpPerspective(src, M, (width, height), INTER_CUBIC, BORDER_REPLICATE)</c> for interleaved 8-bit pixels;
    /// <paramref name="matrix"/> maps source to target (it is inverted here, as OpenCV does without WARP_INVERSE_MAP).
    /// </summary>
    public static byte[] PerspectiveCubic(
        ReadOnlySpan<byte> source, int height, int width, int channels, ReadOnlySpan<double> matrix, int targetHeight, int targetWidth)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(targetHeight, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(targetWidth, 1);
        if (source.Length != checked(height * width * channels))
        {
            throw new ArgumentException("Pixel buffer size does not match the image size.", nameof(source));
        }

        var m = Invert3(matrix);
        var output = new byte[checked(targetHeight * targetWidth * channels)];
        var sum = new int[channels];

        // OpenCV runs the rows as parallel stripes (one per 65,536 output pixels), each split into blocks whose
        // origin enters the coordinate arithmetic. Neither depends on the thread count.
        foreach (var (start, end) in Stripes(targetHeight, (double)targetHeight * targetWidth / (1 << 16)))
        {
            int bh0 = Math.Min(BlockSize / 2, end - start);
            int bw0 = Math.Min(BlockSize * BlockSize / bh0, targetWidth);
            bh0 = Math.Min(BlockSize * BlockSize / bw0, end - start);
            for (int y = start; y < end; y += bh0)
            {
                for (int x = 0; x < targetWidth; x += bw0)
                {
                    int bw = Math.Min(bw0, targetWidth - x);
                    int bh = Math.Min(bh0, end - y);
                    for (int y1 = 0; y1 < bh; y1++)
                    {
                        double x0 = m[0] * x + m[1] * (y + y1) + m[2];
                        double yy0 = m[3] * x + m[4] * (y + y1) + m[5];
                        double w0 = m[6] * x + m[7] * (y + y1) + m[8];
                        for (int x1 = 0; x1 < bw; x1++)
                        {
                            double w = w0 + m[6] * x1;
                            w = w != 0 ? TabSize / w : 0;
                            double fx = Math.Max(int.MinValue, Math.Min(int.MaxValue, (x0 + m[0] * x1) * w));
                            double fy = Math.Max(int.MinValue, Math.Min(int.MaxValue, (yy0 + m[3] * x1) * w));
                            int ix = SaturateInt(fx), iy = SaturateInt(fy);
                            int sx = SaturateShortInt(ix >> TabBits) - 1;
                            int sy = SaturateShortInt(iy >> TabBits) - 1;
                            int alpha = (iy & (TabSize - 1)) * TabSize + (ix & (TabSize - 1));
                            Sample(source, height, width, channels, sx, sy, alpha, sum);
                            int o = ((y + y1) * targetWidth + x + x1) * channels;
                            for (int c = 0; c < channels; c++)
                            {
                                output[o + c] = (byte)Math.Clamp((sum[c] + (1 << (CoefficientBits - 1))) >> CoefficientBits, 0, 255);
                            }
                        }
                    }
                }
            }
        }

        return output;
    }

    private static void Sample(ReadOnlySpan<byte> source, int height, int width, int channels, int sx, int sy, int alpha, int[] sum)
    {
        Array.Clear(sum);
        int t = alpha * 16;
        for (int i = 0; i < 4; i++)
        {
            int row = Math.Clamp(sy + i, 0, height - 1) * width;
            for (int j = 0; j < 4; j++)
            {
                int weight = CubicTable[t + i * 4 + j];
                int p = (row + Math.Clamp(sx + j, 0, width - 1)) * channels;
                for (int c = 0; c < channels; c++)
                {
                    sum[c] += source[p + c] * weight;
                }
            }
        }
    }

    /// <summary>OpenCV <c>parallel_for_</c> stripes of a row range: <c>nstripes = clamp(round(n), 1, rows)</c>.</summary>
    private static IEnumerable<(int Start, int End)> Stripes(int rows, double requested)
    {
        int stripes = Math.Clamp((int)Math.Round(requested, MidpointRounding.ToEven), 1, rows);
        for (int s = 0; s < stripes; s++)
        {
            int start = (int)(((long)s * rows + stripes / 2) / stripes);
            int end = s + 1 >= stripes ? rows : (int)(((long)(s + 1) * rows + stripes / 2) / stripes);
            if (end > start)
            {
                yield return (start, end);
            }
        }
    }

    private static int SaturateInt(double value)
    {
        double rounded = Math.Round(value, MidpointRounding.ToEven);
        return rounded >= int.MaxValue ? int.MaxValue : rounded <= int.MinValue ? int.MinValue : (int)rounded;
    }

    private static int SaturateShortInt(int value) => Math.Clamp(value, short.MinValue, short.MaxValue);

    /// <summary>OpenCV <c>initInterTab2D(INTER_CUBIC, fixpt=true)</c>: 32x32 subpixel positions, 4x4 weights each.</summary>
    private static int[] BuildCubicTable()
    {
        var table1 = new float[TabSize * 4];
        float scale = 1f / TabSize;
        for (int i = 0; i < TabSize; i++)
        {
            Cubic(i * scale, table1.AsSpan(i * 4, 4));
        }

        var table = new int[TabSize * TabSize * 16];
        Span<int> values = stackalloc int[16];
        for (int i = 0; i < TabSize; i++)
        {
            for (int j = 0; j < TabSize; j++)
            {
                int sum = 0;
                for (int k1 = 0; k1 < 4; k1++)
                {
                    float vy = table1[i * 4 + k1];
                    for (int k2 = 0; k2 < 4; k2++)
                    {
                        float v = vy * table1[j * 4 + k2];
                        sum += values[k1 * 4 + k2] = OpenCvResize.SaturateShort(v * CoefficientScale);
                    }
                }

                if (sum != CoefficientScale)
                {
                    int diff = sum - CoefficientScale;
                    int ksize2 = 2;
                    int mk1 = ksize2, mk2 = ksize2, maxK1 = ksize2, maxK2 = ksize2;
                    for (int k1 = ksize2; k1 < ksize2 + 2; k1++)
                    {
                        for (int k2 = ksize2; k2 < ksize2 + 2; k2++)
                        {
                            if (values[k1 * 4 + k2] < values[mk1 * 4 + mk2])
                            {
                                (mk1, mk2) = (k1, k2);
                            }
                            else if (values[k1 * 4 + k2] > values[maxK1 * 4 + maxK2])
                            {
                                (maxK1, maxK2) = (k1, k2);
                            }
                        }
                    }

                    if (diff < 0)
                    {
                        values[maxK1 * 4 + maxK2] -= diff;
                    }
                    else
                    {
                        values[mk1 * 4 + mk2] -= diff;
                    }
                }

                values.CopyTo(table.AsSpan((i * TabSize + j) * 16, 16));
            }
        }

        return table;
    }

    /// <summary>OpenCV <c>interpolateCubic</c> (A = -0.75), in float.</summary>
    private static void Cubic(float x, Span<float> coefficients)
    {
        const float A = -0.75f;
        coefficients[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
        coefficients[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
        coefficients[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
        coefficients[3] = 1f - coefficients[0] - coefficients[1] - coefficients[2];
    }
}
