namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// Port of PyTorch's CPU uint8 antialiased bicubic resize (<c>aten/src/ATen/native/cpu/UpSampleKernel.cpp</c>,
/// <c>_compute_index_ranges_int16_weights</c>, and the separable horizontal/vertical uint8 convolutions of
/// <c>UpSampleKernelAVXAntialias.h</c>, PyTorch 2.14.1), for interleaved RGB24.
/// </summary>
/// <remarks>
/// Per pass (horizontal when the width changes, then vertical when the height changes):
/// <list type="number">
/// <item><c>scale = in / out</c>; <c>support = 2 * max(scale, 1)</c>; output pixel i covers input pixels
/// <c>[trunc(center - support + 0.5), trunc(center + support + 0.5))</c> clipped to the image, with
/// <c>center = scale * (i + 0.5)</c>.</item>
/// <item>Weights <c>filter((j + xmin - center + 0.5) / max(scale, 1))</c> with the Keys cubic (a = -0.5), normalized
/// to sum 1 in double.</item>
/// <item>Precision p: the largest p below 22 with <c>(int)(0.5 + max weight * 2^(p+1)) &lt; 2^15</c> over the pass;
/// weights become <c>round-half-away(w * 2^p)</c> (int16 range).</item>
/// <item>Each output byte is <c>clamp((2^(p-1) + sum(src * weight)) &gt;&gt; p, 0, 255)</c> in int32 arithmetic (an
/// arithmetic shift), stored as uint8 between the passes.</item>
/// </list>
/// The AVX2 and generic PyTorch kernels use the same int16 weights and integer arithmetic; the fixtures were produced
/// on AVX2. Integer arithmetic makes the result independent of vectorization and thread count here.
/// </remarks>
internal static class TorchAntialiasResampler
{
    private const int MaxPrecision = 22;

    public static byte[] Resize(ReadOnlySpan<byte> rgb, int width, int height, int targetWidth, int targetHeight)
    {
        if (targetWidth == width && targetHeight == height)
        {
            return rgb.ToArray();
        }

        byte[]? horizontal = null;
        ReadOnlySpan<byte> source = rgb;
        if (targetWidth != width)
        {
            var c = ComputeWeights(width, targetWidth);
            horizontal = new byte[checked(targetWidth * height * 3)];
            Horizontal(source, width, height, targetWidth, c, horizontal);
            source = horizontal;
        }

        if (targetHeight == height)
        {
            return horizontal!;
        }

        var v = ComputeWeights(height, targetHeight);
        var result = new byte[checked(targetWidth * targetHeight * 3)];
        Vertical(source, targetWidth, targetHeight, v, result);
        return result;
    }

    /// <summary>Keys cubic convolution kernel with a = -0.5 (PyTorch's antialias bicubic filter).</summary>
    internal static double Filter(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        if (x < 1.0)
        {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        }

        if (x < 2.0)
        {
            return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
        }

        return 0.0;
    }

    internal static Weights ComputeWeights(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize;
        double support = scale >= 1.0 ? 2.0 * scale : 2.0;
        double invScale = scale >= 1.0 ? 1.0 / scale : 1.0;
        int maxSize = (int)Math.Ceiling(support) * 2 + 1;
        var w = new double[outSize * maxSize];
        var min = new int[outSize];
        var size = new int[outSize];
        double wMax = 0.0;
        for (int i = 0; i < outSize; i++)
        {
            double center = scale * (i + 0.5);
            int xmin = Math.Max((int)(center - support + 0.5), 0);
            int xsize = Math.Min((int)(center + support + 0.5), inSize) - xmin;
            xsize = Math.Clamp(xsize, 0, maxSize);
            double total = 0.0;
            int row = i * maxSize;
            for (int j = 0; j < xsize; j++)
            {
                double value = Filter((j + xmin - center + 0.5) * invScale);
                w[row + j] = value;
                total += value;
            }

            if (total != 0.0)
            {
                for (int j = 0; j < xsize; j++)
                {
                    w[row + j] /= total;
                }
            }

            for (int j = 0; j < xsize; j++)
            {
                wMax = Math.Max(wMax, w[row + j]);
            }

            min[i] = xmin;
            size[i] = xsize;
        }

        int precision;
        for (precision = 0; precision < MaxPrecision; precision++)
        {
            int next = (int)(0.5 + wMax * (1 << (precision + 1)));
            if (next >= (1 << 15))
            {
                break;
            }
        }

        var q = new int[w.Length];
        double factor = 1 << precision;
        for (int k = 0; k < w.Length; k++)
        {
            double v = w[k] * factor;
            q[k] = v < 0 ? (int)(-0.5 + v) : (int)(0.5 + v);
        }

        return new Weights(min, size, q, maxSize, precision);
    }

    private static void Horizontal(ReadOnlySpan<byte> src, int width, int height, int outWidth, Weights c, Span<byte> dst)
    {
        int round = 1 << (c.Precision - 1);
        int shift = c.Precision;
        for (int y = 0; y < height; y++)
        {
            var srcRow = src.Slice(y * width * 3, width * 3);
            var dstRow = dst.Slice(y * outWidth * 3, outWidth * 3);
            for (int x = 0; x < outWidth; x++)
            {
                int s0 = round, s1 = round, s2 = round;
                int p = c.Min[x] * 3;
                var k = c.Coefficients.AsSpan(x * c.MaxSize, c.Size[x]);
                for (int j = 0; j < k.Length; j++, p += 3)
                {
                    int wj = k[j];
                    s0 += srcRow[p] * wj;
                    s1 += srcRow[p + 1] * wj;
                    s2 += srcRow[p + 2] * wj;
                }

                int o = x * 3;
                dstRow[o] = Clip(s0 >> shift);
                dstRow[o + 1] = Clip(s1 >> shift);
                dstRow[o + 2] = Clip(s2 >> shift);
            }
        }
    }

    private static void Vertical(ReadOnlySpan<byte> src, int width, int outHeight, Weights c, Span<byte> dst)
    {
        int round = 1 << (c.Precision - 1);
        int shift = c.Precision;
        int rowBytes = width * 3;
        var sums = new int[rowBytes];
        for (int y = 0; y < outHeight; y++)
        {
            Array.Fill(sums, round);
            var k = c.Coefficients.AsSpan(y * c.MaxSize, c.Size[y]);
            int ymin = c.Min[y];
            for (int j = 0; j < k.Length; j++)
            {
                int wj = k[j];
                var srcRow = src.Slice((ymin + j) * rowBytes, rowBytes);
                for (int b = 0; b < rowBytes; b++)
                {
                    sums[b] += srcRow[b] * wj;
                }
            }

            var dstRow = dst.Slice(y * rowBytes, rowBytes);
            for (int b = 0; b < rowBytes; b++)
            {
                dstRow[b] = Clip(sums[b] >> shift);
            }
        }
    }

    private static byte Clip(int v) => (byte)(v < 0 ? 0 : v > 255 ? 255 : v);

    /// <summary>Per output index: first input index, tap count, int16-range weights (row stride <c>MaxSize</c>), precision.</summary>
    internal sealed record Weights(int[] Min, int[] Size, int[] Coefficients, int MaxSize, int Precision);
}
