using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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
/// <para>
/// Execution: with AVX2 both passes multiply uint8 pixels by pairs of int16 weights (<c>vpmaddwd</c>, exact in int32)
/// and saturate on packing, which is the scalar clamp; rows (horizontal pass) and output rows (vertical pass) are
/// independent, so large images split them over the thread pool. Without AVX2 the scalar loops run. Every path gives
/// the same bytes (tests compare them on random sizes and contents).
/// </para>
/// </remarks>
internal static class TorchAntialiasResampler
{
    private const int MaxPrecision = 22;

    /// <summary>Multiply-adds below which a pass stays on the calling thread, and the least work per parallel chunk.</summary>
    private const long MinParallelWork = 1L << 19;

    /// <summary>Resizes with the fastest available path, single-threaded (the source is a span).</summary>
    public static byte[] Resize(ReadOnlySpan<byte> rgb, int width, int height, int targetWidth, int targetHeight) =>
        Resize(rgb, default, width, height, targetWidth, targetHeight, Avx2.IsSupported, 1);

    /// <summary>
    /// Resize with an explicit path: <paramref name="vectorized"/> (AVX2, ignored without it) and the largest thread
    /// count. <paramref name="memory"/> is <paramref name="rgb"/> as memory, or empty when only the span exists (then
    /// the horizontal pass runs on the calling thread).
    /// </summary>
    internal static byte[] Resize(
        ReadOnlySpan<byte> rgb, ReadOnlyMemory<byte> memory, int width, int height, int targetWidth, int targetHeight, bool vectorized, int maxThreads)
    {
        if (targetWidth == width && targetHeight == height)
        {
            return rgb.ToArray();
        }

        vectorized &= Avx2.IsSupported;
        byte[]? horizontal = null;
        bool rented = false;
        try
        {
            ReadOnlySpan<byte> source = rgb;
            ReadOnlyMemory<byte> sourceMemory = memory;
            if (targetWidth != width)
            {
                var c = ComputeWeights(width, targetWidth);
                int bytes = checked(targetWidth * height * 3);
                rented = targetHeight != height;
                horizontal = rented ? ArrayPool<byte>.Shared.Rent(bytes) : new byte[bytes];
                byte[] h = horizontal;
                long work = (long)targetWidth * c.MaxSize * 3;
                if (memory.Length == rgb.Length && !memory.IsEmpty)
                {
                    For(height, work, maxThreads, (y0, y1) => Horizontal(memory.Span, width, targetWidth, c, h, y0, y1, vectorized));
                }
                else
                {
                    Horizontal(rgb, width, targetWidth, c, h, 0, height, vectorized);
                }

                source = horizontal.AsSpan(0, bytes);
                sourceMemory = horizontal.AsMemory(0, bytes);
            }

            if (targetHeight == height)
            {
                return horizontal!;
            }

            var v = ComputeWeights(height, targetHeight);
            var result = new byte[checked(targetWidth * targetHeight * 3)];
            long rowWork = (long)targetWidth * 3 * v.MaxSize;
            if (!sourceMemory.IsEmpty && sourceMemory.Length == source.Length)
            {
                For(targetHeight, rowWork, maxThreads, (y0, y1) => Vertical(sourceMemory.Span, targetWidth, v, result, y0, y1, vectorized));
            }
            else
            {
                Vertical(source, targetWidth, v, result, 0, targetHeight, vectorized);
            }

            return result;
        }
        finally
        {
            if (rented)
            {
                ArrayPool<byte>.Shared.Return(horizontal!);
            }
        }
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

        // Pairs of consecutive weights as two int16 halves of one int32 (low: tap 2k, high: tap 2k + 1, 0 past the end).
        int maxPairs = (maxSize + 1) / 2;
        var pairs = new int[outSize * maxPairs];
        for (int i = 0; i < outSize; i++)
        {
            for (int k = 0; 2 * k < size[i]; k++)
            {
                int w0 = q[i * maxSize + 2 * k];
                int w1 = 2 * k + 1 < size[i] ? q[i * maxSize + 2 * k + 1] : 0;
                pairs[i * maxPairs + k] = (ushort)(short)w0 | (w1 << 16);
            }
        }

        return new Weights(min, size, q, maxSize, precision, pairs, maxPairs);
    }

    /// <summary>Runs <paramref name="body"/> over [0, count) in contiguous chunks, in parallel when the work is large.</summary>
    internal static void For(int count, long workPerItem, int maxThreads, Action<int, int> body)
    {
        long total = count * Math.Max(workPerItem, 1);
        int chunks = (int)Math.Min(Math.Min((long)maxThreads * 2, count), total / MinParallelWork);
        if (maxThreads <= 1 || chunks <= 1)
        {
            body(0, count);
            return;
        }

        Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = maxThreads }, i =>
            body((int)((long)i * count / chunks), (int)((long)(i + 1) * count / chunks)));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Horizontal(ReadOnlySpan<byte> src, int width, int outWidth, Weights c, Span<byte> dst, int y0, int y1, bool vectorized)
    {
        int y = y0;
        if (vectorized)
        {
            // Four rows per sweep; a short tail repeats its last row (the same bytes are written twice).
            for (; y < y1; y += 4)
            {
                int last = y1 - 1;
                HorizontalAvx2(src, width, outWidth, c, dst, y, Math.Min(y + 1, last), Math.Min(y + 2, last), Math.Min(y + 3, last));
            }

            return;
        }

        for (; y < y1; y++)
        {
            var srcRow = src.Slice(y * width * 3, width * 3);
            var dstRow = dst.Slice(y * outWidth * 3, outWidth * 3);
            for (int x = 0; x < outWidth; x++)
            {
                HorizontalPixel(srcRow, dstRow, x, c);
            }
        }
    }

    private static void HorizontalPixel(ReadOnlySpan<byte> srcRow, Span<byte> dstRow, int x, Weights c)
    {
        int s0 = 1 << (c.Precision - 1), s1 = s0, s2 = s0;
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
        dstRow[o] = Clip(s0 >> c.Precision);
        dstRow[o + 1] = Clip(s1 >> c.Precision);
        dstRow[o + 2] = Clip(s2 >> c.Precision);
    }

    /// <summary>
    /// Four rows of the horizontal pass (rows may repeat), two per 256-bit register, one in each 128-bit lane. A pair of
    /// taps is two source pixels shuffled to (R, R', G, G', B, B', 0, 0) as int16 and multiplied with (w, w') by
    /// <c>vpmaddwd</c> into int32 (R, G, B, 0) sums; one 16-byte load feeds two pairs.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void HorizontalAvx2(ReadOnlySpan<byte> src, int width, int outWidth, Weights c, Span<byte> dst, int ya, int yb, int yc, int yd)
    {
        int rowBytes = width * 3;
        int outBytes = outWidth * 3;
        var rowA = src.Slice(ya * rowBytes, rowBytes);
        var rowB = src.Slice(yb * rowBytes, rowBytes);
        var rowC = src.Slice(yc * rowBytes, rowBytes);
        var rowD = src.Slice(yd * rowBytes, rowBytes);
        var dstA = dst.Slice(ya * outBytes, outBytes);
        var dstB = dst.Slice(yb * outBytes, outBytes);
        var dstC = dst.Slice(yc * outBytes, outBytes);
        var dstD = dst.Slice(yd * outBytes, outBytes);
        ref byte a = ref MemoryMarshal.GetReference(rowA);

        // Rows b, c and d as offsets from row a (all four rows were sliced in bounds above).
        nuint ob = (nuint)((nint)(yb - ya) * rowBytes);
        nuint oc = (nuint)((nint)(yc - ya) * rowBytes);
        nuint od = (nuint)((nint)(yd - ya) * rowBytes);
        ref byte target = ref MemoryMarshal.GetReference(dstA);
        nuint tb = (nuint)((nint)(yb - ya) * outBytes);
        nuint tc = (nuint)((nint)(yc - ya) * outBytes);
        nuint td = (nuint)((nint)(yd - ya) * outBytes);
        ref int pairs = ref MemoryMarshal.GetArrayDataReference(c.Pairs);
        int round = 1 << (c.Precision - 1);
        var init = Vector256.Create(round, round, round, 0, round, round, round, 0);
        var first = Vector256.Create(
            (byte)0, 0x80, 3, 0x80, 1, 0x80, 4, 0x80, 2, 0x80, 5, 0x80, 0x80, 0x80, 0x80, 0x80,
            0, 0x80, 3, 0x80, 1, 0x80, 4, 0x80, 2, 0x80, 5, 0x80, 0x80, 0x80, 0x80, 0x80);
        var second = Vector256.Create(
            (byte)6, 0x80, 9, 0x80, 7, 0x80, 10, 0x80, 8, 0x80, 11, 0x80, 0x80, 0x80, 0x80, 0x80,
            6, 0x80, 9, 0x80, 7, 0x80, 10, 0x80, 8, 0x80, 11, 0x80, 0x80, 0x80, 0x80, 0x80);
        var shift = Vector128.CreateScalar(c.Precision);
        for (int x = 0; x < outWidth; x++)
        {
            int min = c.Min[x];
            int count = (c.Size[x] + 1) >> 1;

            // The last pair reads 8 bytes from pixel min + 2 * (count - 1); past the row end, the pixel runs scalar below
            // (no call in this loop, so the row references stay in registers).
            if ((min + 2 * count) * 3 + 2 > rowBytes)
            {
                continue;
            }

            var ab = init;
            var cd = init;
            int p = min * 3;
            ref int w = ref Unsafe.Add(ref pairs, x * c.MaxPairs);

            // One base reference and the other rows as offsets from it keeps the loop's references in registers.
            ref byte q = ref Unsafe.Add(ref a, p);
            int quads = p + 16 <= rowBytes ? Math.Min(count >> 1, ((rowBytes - 16 - p) / 12) + 1) : 0;
            for (int k = 0; k < quads; k++)
            {
                var w0 = Vector256.Create(Unsafe.Add(ref w, 2 * k)).AsInt16();
                var w1 = Vector256.Create(Unsafe.Add(ref w, 2 * k + 1)).AsInt16();
                var vab = Vector256.Create(Vector128.LoadUnsafe(ref q), Vector128.LoadUnsafe(ref q, ob));
                var vcd = Vector256.Create(Vector128.LoadUnsafe(ref q, oc), Vector128.LoadUnsafe(ref q, od));
                ab = Avx2.Add(ab, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(vab, first).AsInt16(), w0));
                cd = Avx2.Add(cd, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(vcd, first).AsInt16(), w0));
                ab = Avx2.Add(ab, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(vab, second).AsInt16(), w1));
                cd = Avx2.Add(cd, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(vcd, second).AsInt16(), w1));
                q = ref Unsafe.Add(ref q, 12);
            }

            for (int k = 2 * quads; k < count; k++)
            {
                var w0 = Vector256.Create(Unsafe.Add(ref w, k)).AsInt16();
                var vab = Vector256.Create(
                    Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<long>(ref q)),
                    Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<long>(ref Unsafe.Add(ref q, ob)))).AsByte();
                var vcd = Vector256.Create(
                    Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<long>(ref Unsafe.Add(ref q, oc))),
                    Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<long>(ref Unsafe.Add(ref q, od)))).AsByte();
                ab = Avx2.Add(ab, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(vab, first).AsInt16(), w0));
                cd = Avx2.Add(cd, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(vcd, first).AsInt16(), w0));
                q = ref Unsafe.Add(ref q, 6);
            }

            // Saturating packs are the clamp to [0, 255]; each lane's first three bytes are one row's (R, G, B).
            var words = Avx2.PackSignedSaturate(Avx2.ShiftRightArithmetic(ab, shift), Avx2.ShiftRightArithmetic(cd, shift));
            var bytes = Avx2.PackUnsignedSaturate(words, words).AsUInt32();
            int o = x * 3;
            if (o + 4 <= outBytes)
            {
                // Four-byte stores: the fourth byte is pixel x + 1's red, which a later iteration (or the scalar loop
                // below) overwrites; it never leaves the row.
                ref byte t = ref Unsafe.Add(ref target, o);
                Unsafe.WriteUnaligned(ref t, bytes.GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref t, tb), bytes.GetElement(4));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref t, tc), bytes.GetElement(1));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref t, td), bytes.GetElement(5));
            }
            else
            {
                Write3(dstA, o, bytes.GetElement(0));
                Write3(dstB, o, bytes.GetElement(4));
                Write3(dstC, o, bytes.GetElement(1));
                Write3(dstD, o, bytes.GetElement(5));
            }
        }

        for (int x = 0; x < outWidth; x++)
        {
            if ((c.Min[x] + 2 * ((c.Size[x] + 1) >> 1)) * 3 + 2 > rowBytes)
            {
                HorizontalPixel(rowA, dstA, x, c);
                HorizontalPixel(rowB, dstB, x, c);
                HorizontalPixel(rowC, dstC, x, c);
                HorizontalPixel(rowD, dstD, x, c);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Write3(Span<byte> row, int o, uint rgb)
    {
        row[o + 2] = (byte)(rgb >> 16);
        row[o] = (byte)rgb;
        row[o + 1] = (byte)(rgb >> 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Vertical(ReadOnlySpan<byte> src, int width, Weights c, Span<byte> dst, int y0, int y1, bool vectorized)
    {
        int rowBytes = width * 3;
        int round = 1 << (c.Precision - 1);
        int shift = c.Precision;
        for (int y = y0; y < y1; y++)
        {
            var dstRow = dst.Slice(y * rowBytes, rowBytes);
            int ymin = c.Min[y];
            int size = c.Size[y];
            int b = 0;
            if (vectorized)
            {
                b = VerticalAvx2(src, rowBytes, c, y, dstRow);
            }

            var k = c.Coefficients.AsSpan(y * c.MaxSize, size);
            for (; b < rowBytes; b++)
            {
                int sum = round;
                for (int j = 0; j < k.Length; j++)
                {
                    sum += src[(ymin + j) * rowBytes + b] * k[j];
                }

                dstRow[b] = Clip(sum >> shift);
            }
        }
    }

    /// <summary>
    /// The vertical pass of output row <paramref name="y"/> in blocks of 16 bytes: per pair of source rows, the bytes are
    /// widened to int16, interleaved (r, r') and multiplied with (w, w') by <c>vpmaddwd</c>. Returns the bytes done.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int VerticalAvx2(ReadOnlySpan<byte> src, int rowBytes, Weights c, int y, Span<byte> dstRow)
    {
        int ymin = c.Min[y];
        int size = c.Size[y];
        int count = (size + 1) >> 1;
        if (size == 0 || rowBytes < 16)
        {
            return 0;
        }

        // Bounds once: every row read below lies in this slice.
        var rows = src.Slice(ymin * rowBytes, size * rowBytes);
        ref byte s = ref MemoryMarshal.GetReference(rows);
        ref byte d = ref MemoryMarshal.GetReference(dstRow);
        ref int pairs = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(c.Pairs), y * c.MaxPairs);
        var init = Vector256.Create(1 << (c.Precision - 1));
        var shift = Vector128.CreateScalar(c.Precision);
        int b = 0;
        for (; b + 32 <= rowBytes; b += 32)
        {
            var lo = init;
            var hi = init;
            var lo2 = init;
            var hi2 = init;
            for (int k = 0; k < count; k++)
            {
                int r0 = 2 * k;
                int r1 = r0 + 1 < size ? r0 + 1 : r0;
                var w = Vector256.Create(Unsafe.Add(ref pairs, k)).AsInt16();
                var v0 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref s, (nuint)(r0 * rowBytes + b)));
                var v1 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref s, (nuint)(r1 * rowBytes + b)));
                var u0 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref s, (nuint)(r0 * rowBytes + b + 16)));
                var u1 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref s, (nuint)(r1 * rowBytes + b + 16)));
                lo = Avx2.Add(lo, Avx2.MultiplyAddAdjacent(Avx2.UnpackLow(v0, v1), w));
                hi = Avx2.Add(hi, Avx2.MultiplyAddAdjacent(Avx2.UnpackHigh(v0, v1), w));
                lo2 = Avx2.Add(lo2, Avx2.MultiplyAddAdjacent(Avx2.UnpackLow(u0, u1), w));
                hi2 = Avx2.Add(hi2, Avx2.MultiplyAddAdjacent(Avx2.UnpackHigh(u0, u1), w));
            }

            var words = Avx2.PackSignedSaturate(Avx2.ShiftRightArithmetic(lo, shift), Avx2.ShiftRightArithmetic(hi, shift));
            var words2 = Avx2.PackSignedSaturate(Avx2.ShiftRightArithmetic(lo2, shift), Avx2.ShiftRightArithmetic(hi2, shift));

            // Lane 0: bytes 0-7 then 16-23; lane 1: bytes 8-15 then 24-31.
            var bytes = Avx2.PackUnsignedSaturate(words, words2);
            Avx2.Permute4x64(bytes.AsUInt64(), 0b11_01_10_00).AsByte().StoreUnsafe(ref d, (nuint)b);
        }

        for (; b + 16 <= rowBytes; b += 16)
        {
            var lo = init;
            var hi = init;
            for (int k = 0; k < count; k++)
            {
                int r0 = 2 * k;
                int r1 = r0 + 1 < size ? r0 + 1 : r0;
                var v0 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref s, (nuint)(r0 * rowBytes + b)));
                var v1 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref s, (nuint)(r1 * rowBytes + b)));
                var w = Vector256.Create(Unsafe.Add(ref pairs, k)).AsInt16();
                lo = Avx2.Add(lo, Avx2.MultiplyAddAdjacent(Avx2.UnpackLow(v0, v1), w));
                hi = Avx2.Add(hi, Avx2.MultiplyAddAdjacent(Avx2.UnpackHigh(v0, v1), w));
            }

            // Per 128-bit lane, lo holds bytes 0-3 (8-11) and hi bytes 4-7 (12-15): packing restores the order.
            var words = Avx2.PackSignedSaturate(Avx2.ShiftRightArithmetic(lo, shift), Avx2.ShiftRightArithmetic(hi, shift));
            var bytes = Avx2.PackUnsignedSaturate(words, words);
            Avx2.Permute4x64(bytes.AsUInt64(), 0b10_00_10_00).GetLower().AsByte().StoreUnsafe(ref d, (nuint)b);
        }

        return b;
    }

    private static byte Clip(int v) => (byte)(v < 0 ? 0 : v > 255 ? 255 : v);

    /// <summary>
    /// Per output index: first input index, tap count, int16-range weights (row stride <c>MaxSize</c>), precision, and
    /// the weights as packed int16 pairs (row stride <c>MaxPairs</c>).
    /// </summary>
    internal sealed record Weights(int[] Min, int[] Size, int[] Coefficients, int MaxSize, int Precision, int[] Pairs, int MaxPairs);
}
