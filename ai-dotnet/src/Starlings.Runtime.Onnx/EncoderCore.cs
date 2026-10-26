using System.Buffers;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Runtime.Onnx;

/// <summary>How <see cref="EncoderCore.Pool"/> reduces a row's token states to one vector.</summary>
internal enum EncoderPooling
{
    /// <summary>Mean over the positions with attention mask 1 (sentence-transformers <c>Pooling(mean)</c>).</summary>
    Mean,

    /// <summary>The first position's state (sentence-transformers <c>Pooling(cls)</c>); the mask is not read.</summary>
    Cls,
}

/// <summary>
/// The token ids, attention mask and token type ids of one tokenized input, as <see cref="EncoderCore"/> packs them.
/// An empty <see cref="AttentionMask"/> means every position is attended; empty <see cref="TokenTypeIds"/> means all
/// zero. When present, both have the length of <see cref="InputIds"/>.
/// </summary>
internal readonly record struct EncoderRow(ReadOnlyMemory<long> InputIds, ReadOnlyMemory<long> AttentionMask, ReadOnlyMemory<long> TokenTypeIds);

/// <summary>
/// Receives one chunk's graph output (row-major, <c>rows x outputPerRow</c> floats) and the packed attention mask
/// (<c>rows x width</c>). Both spans are valid only during the call.
/// </summary>
internal delegate void EncoderChunkOutput(ReadOnlySpan<float> output, ReadOnlySpan<long> mask);

/// <summary>
/// The encoder steps the text embedders share (e5, EmbeddingGemma 2 text, the BERT route): right-padded packing of a
/// chunk of rows, one ONNX Runtime run over the chunk, and mean or CLS pooling with optional L2 normalization (double
/// accumulation, rounded to float once). Moved here from <c>E5Encoder</c>, <c>EmbeddingGemma2Encoder</c> and
/// <c>BertSession</c>, which computed the same numbers three times; the arithmetic and its order are unchanged, so
/// their outputs are bit for bit those of the copies it replaces. Chunk planning stays with the callers
/// (<c>Starlings.Batching.BatchPlan</c>).
/// </summary>
/// <remarks>
/// An instance holds a session and the graph's input and output names. It is safe to use concurrently: every call
/// rents its own buffers (<see cref="ArrayPool{T}.Shared"/>), and the <see cref="OrtValue"/>s that pin them are disposed
/// before the buffers return to the pool.
/// </remarks>
internal sealed class EncoderCore
{
    private readonly OnnxModelSession _session;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly bool _feedsTokenTypes;

    /// <summary>Binds a session and the graph's names.</summary>
    /// <param name="session">The session; not owned.</param>
    /// <param name="inputIdsName">The int64 <c>[batch, sequence]</c> token id input.</param>
    /// <param name="attentionMaskName">The int64 <c>[batch, sequence]</c> attention mask input.</param>
    /// <param name="tokenTypeIdsName">The int64 <c>[batch, sequence]</c> token type id input, or null when the graph has none.</param>
    /// <param name="outputName">The float output read back.</param>
    public EncoderCore(OnnxModelSession session, string inputIdsName, string attentionMaskName, string? tokenTypeIdsName, string outputName)
    {
        ArgumentNullException.ThrowIfNull(session);
        _session = session;
        _feedsTokenTypes = tokenTypeIdsName is not null;
        _inputNames = tokenTypeIdsName is null ? [inputIdsName, attentionMaskName] : [inputIdsName, attentionMaskName, tokenTypeIdsName];
        _outputNames = [outputName];
    }

    /// <summary>
    /// Packs the inputs at <paramref name="rows"/> right-padded to <paramref name="width"/>, runs them as one call
    /// whose output is <c>[rows, width, hiddenWidth]</c> token states, pools each row and writes row <c>rows[r]</c>'s
    /// <paramref name="outputDimension"/> floats at <c>destination[rows[r] * outputDimension]</c>.
    /// </summary>
    public void EmbedChunk<TInput>(
        IReadOnlyList<TInput> inputs,
        ReadOnlySpan<int> rows,
        int width,
        Func<TInput, EncoderRow> read,
        long padId,
        int hiddenWidth,
        EncoderPooling pooling,
        bool normalize,
        int outputDimension,
        Span<float> destination,
        CancellationToken cancellationToken)
    {
        int count = rows.Length;
        int cells = count * width;
        int hiddenCells = checked(cells * hiddenWidth);
        long[] ids = ArrayPool<long>.Shared.Rent(cells);
        long[] mask = ArrayPool<long>.Shared.Rent(cells);
        long[]? types = _feedsTokenTypes ? ArrayPool<long>.Shared.Rent(cells) : null;
        float[] hidden = ArrayPool<float>.Shared.Rent(hiddenCells);
        try
        {
            PackRightPadded(inputs, rows, width, padId, read, ids, mask, types is null ? default : types.AsSpan(0, cells));
            Run(ids, mask, types, hidden, count, width, [count, width, hiddenWidth], hiddenCells, cancellationToken);
            for (int r = 0; r < count; r++)
            {
                Pool(
                    hidden.AsSpan(r * width * hiddenWidth, width * hiddenWidth),
                    mask.AsSpan(r * width, width),
                    hiddenWidth,
                    outputDimension,
                    pooling,
                    normalize,
                    destination.Slice(rows[r] * outputDimension, outputDimension));
            }
        }
        finally
        {
            // The OrtValues pinning these arrays are disposed inside Run, before the arrays return to the pool.
            ArrayPool<float>.Shared.Return(hidden);
            if (types is not null)
            {
                ArrayPool<long>.Shared.Return(types);
            }

            ArrayPool<long>.Shared.Return(mask);
            ArrayPool<long>.Shared.Return(ids);
        }
    }

    /// <summary>
    /// Packs the inputs at <paramref name="rows"/> right-padded to <paramref name="width"/>, runs them as one call with
    /// output shape <paramref name="outputShape"/> (<c>outputCells</c> floats in all) and hands the output and the
    /// packed mask to <paramref name="consume"/>, for callers that read the output themselves (scores, or pooling with
    /// their own checks).
    /// </summary>
    public void RunChunk<TInput>(
        IReadOnlyList<TInput> inputs,
        ReadOnlySpan<int> rows,
        int width,
        Func<TInput, EncoderRow> read,
        long padId,
        long[] outputShape,
        EncoderChunkOutput consume,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(outputShape);
        ArgumentNullException.ThrowIfNull(consume);
        int count = rows.Length;
        int cells = count * width;
        long outputLength = 1;
        foreach (long extent in outputShape)
        {
            outputLength = checked(outputLength * extent);
        }

        int outputCells = checked((int)outputLength);
        long[] ids = ArrayPool<long>.Shared.Rent(cells);
        long[] mask = ArrayPool<long>.Shared.Rent(cells);
        long[]? types = _feedsTokenTypes ? ArrayPool<long>.Shared.Rent(cells) : null;
        float[] output = ArrayPool<float>.Shared.Rent(outputCells);
        try
        {
            PackRightPadded(inputs, rows, width, padId, read, ids, mask, types is null ? default : types.AsSpan(0, cells));
            Run(ids, mask, types, output, count, width, outputShape, outputCells, cancellationToken);
            consume(output.AsSpan(0, outputCells), mask.AsSpan(0, cells));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(output);
            if (types is not null)
            {
                ArrayPool<long>.Shared.Return(types);
            }

            ArrayPool<long>.Shared.Return(mask);
            ArrayPool<long>.Shared.Return(ids);
        }
    }

    /// <summary>
    /// Writes the inputs at <paramref name="rows"/>, in that order, into row-major <c>[rows.Length, width]</c> buffers:
    /// each row's ids then <paramref name="padId"/>; its mask (all 1 when the input has none) then 0; and, when
    /// <paramref name="tokenTypes"/> is not empty, its token type ids (all 0 when the input has none) then 0.
    /// </summary>
    internal static void PackRightPadded<TInput>(
        IReadOnlyList<TInput> inputs,
        ReadOnlySpan<int> rows,
        int width,
        long padId,
        Func<TInput, EncoderRow> read,
        Span<long> ids,
        Span<long> mask,
        Span<long> tokenTypes)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(read);
        for (int r = 0; r < rows.Length; r++)
        {
            var row = read(inputs[rows[r]]);
            int length = row.InputIds.Length;
            var idRow = ids.Slice(r * width, width);
            var maskRow = mask.Slice(r * width, width);
            row.InputIds.Span.CopyTo(idRow);
            idRow[length..].Fill(padId);
            if (row.AttentionMask.IsEmpty)
            {
                maskRow[..length].Fill(1);
            }
            else
            {
                row.AttentionMask.Span.CopyTo(maskRow);
            }

            maskRow[length..].Clear();
            if (!tokenTypes.IsEmpty)
            {
                var typeRow = tokenTypes.Slice(r * width, width);
                if (row.TokenTypeIds.IsEmpty)
                {
                    typeRow.Clear();
                }
                else
                {
                    row.TokenTypeIds.Span.CopyTo(typeRow);
                    typeRow[length..].Clear();
                }
            }
        }
    }

    /// <summary>
    /// Pools one row of <paramref name="width"/>-wide token states into <paramref name="destination"/>
    /// (<paramref name="outputDimension"/> floats, at most <paramref name="width"/>), accumulating in double and rounding
    /// to float once. <see cref="EncoderPooling.Mean"/>: the sum over positions with mask 1 divided by max(count, 1e-9).
    /// <see cref="EncoderPooling.Cls"/>: position 0. With <paramref name="normalize"/>, the first
    /// <paramref name="outputDimension"/> values are divided by their L2 norm (max(norm, 1e-12)), which for a smaller
    /// output dimension equals normalizing, truncating and normalizing again (Matryoshka; normalization is
    /// scale-invariant). Returns the number of attended positions (mean) or 1 (CLS).
    /// </summary>
    internal static int Pool(ReadOnlySpan<float> hidden, ReadOnlySpan<long> mask, int width, int outputDimension, EncoderPooling pooling, bool normalize, Span<float> destination)
    {
        Span<double> sum = width <= 1024 ? stackalloc double[width] : new double[width];
        sum.Clear();
        int attended;
        if (pooling == EncoderPooling.Cls)
        {
            for (int d = 0; d < width; d++)
            {
                sum[d] = hidden[d];
            }

            attended = 1;
        }
        else
        {
            double count = 0;
            for (int t = 0; t < mask.Length; t++)
            {
                if (mask[t] == 0)
                {
                    continue;
                }

                count++;
                var row = hidden.Slice(t * width, width);
                for (int d = 0; d < width; d++)
                {
                    sum[d] += row[d];
                }
            }

            attended = (int)count;
            count = Math.Max(count, 1e-9);
            for (int d = 0; d < outputDimension; d++)
            {
                sum[d] /= count;
            }
        }

        if (!normalize)
        {
            for (int d = 0; d < outputDimension; d++)
            {
                destination[d] = (float)sum[d];
            }

            return attended;
        }

        double norm = 0;
        for (int d = 0; d < outputDimension; d++)
        {
            norm += sum[d] * sum[d];
        }

        norm = Math.Max(Math.Sqrt(norm), 1e-12);
        for (int d = 0; d < outputDimension; d++)
        {
            destination[d] = (float)(sum[d] / norm);
        }

        return attended;
    }

    private void Run(long[] ids, long[] mask, long[]? types, float[] output, int count, int width, long[] outputShape, int outputCells, CancellationToken cancellationToken)
    {
        int cells = count * width;
        long[] shape = [count, width];
        var memoryInfo = OrtMemoryInfo.DefaultInstance;
        using var idsValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, ids.AsMemory(0, cells), shape);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, mask.AsMemory(0, cells), shape);
        using var typesValue = types is null ? null : OrtValue.CreateTensorValueFromMemory(memoryInfo, types.AsMemory(0, cells), shape);
        using var outputValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, output.AsMemory(0, outputCells), outputShape);
        OrtValue[] inputValues = typesValue is null ? [idsValue, maskValue] : [idsValue, maskValue, typesValue];
        _session.Run(_inputNames, inputValues, _outputNames, [outputValue], cancellationToken);
    }
}
