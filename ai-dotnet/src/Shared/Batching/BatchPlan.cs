// Compiled into Starlings.Models.E5, Starlings.Models.Bert, Starlings.Models.Gliner2 and
// Starlings.Models.EmbeddingGemma2 (linked source, internal to each).
namespace Starlings.Batching;

/// <summary>
/// One ONNX Runtime call of a <see cref="BatchPlan"/>: the rows at positions <see cref="Start"/> to
/// <c>Start + Count</c> of <see cref="BatchPlan.Order"/>, right-padded to <see cref="Width"/> (their longest row).
/// </summary>
internal readonly record struct BatchChunk(int Start, int Count, int Width)
{
    /// <summary>Padded cells of the call: rows x longest row.</summary>
    public long PaddedCells => (long)Count * Width;
}

/// <summary>
/// How an explicit batch is split into ONNX Runtime calls. Rows are taken in <see cref="Order"/> (input order, or
/// ascending token length when sorting, ties in input order) and grouped greedily: a chunk grows while it has fewer
/// than <c>maxRows</c> rows and, when a token budget is set, its padded size (rows x longest row) stays within it.
/// Every chunk has at least one row, so a row longer than the budget runs alone.
/// </summary>
/// <remarks>
/// Without sorting and without a budget the chunks are the consecutive runs of <c>maxRows</c> rows the encoders used
/// before the plan existed, so results are unchanged bit for bit. Sorting changes which rows share a call (and so the
/// padding each row sees); callers scatter each row's result back to its input index.
/// </remarks>
internal sealed class BatchPlan
{
    private readonly int[] _order;

    private BatchPlan(int[] order, bool isInputOrder, List<BatchChunk> chunks)
    {
        _order = order;
        IsInputOrder = isInputOrder;
        Chunks = chunks;
    }

    /// <summary>Input index of the row at each plan position.</summary>
    public IReadOnlyList<int> Order => _order;

    /// <summary>True when <see cref="Order"/> is the identity (no sorting, or the input was already sorted).</summary>
    public bool IsInputOrder { get; }

    /// <summary>The calls, in execution order.</summary>
    public IReadOnlyList<BatchChunk> Chunks { get; }

    /// <summary>Total padded cells over all calls.</summary>
    public long PaddedCells => Chunks.Sum(c => c.PaddedCells);

    /// <summary>Plans <paramref name="lengths"/> (token count per input, in input order).</summary>
    /// <param name="lengths">Token count of each input; each at least 0.</param>
    /// <param name="maxRows">Rows per call at most; at least 1.</param>
    /// <param name="maxCells">Padded cells per call at most, or null for no token budget.</param>
    /// <param name="sortByLength">Order rows by ascending length (stable) before chunking.</param>
    public static BatchPlan Create(ReadOnlySpan<int> lengths, int maxRows, int? maxCells, bool sortByLength)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxRows, 1);
        if (maxCells is { } cells)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(cells, 1, nameof(maxCells));
        }

        int n = lengths.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
        }

        bool isInputOrder = true;
        if (sortByLength && n > 1)
        {
            // Unique keys (length, index) make the unstable Array.Sort stable and deterministic.
            var keys = new long[n];
            for (int i = 0; i < n; i++)
            {
                keys[i] = ((long)lengths[i] << 32) | (uint)i;
            }

            Array.Sort(keys, order);
            for (int i = 0; i < n && isInputOrder; i++)
            {
                isInputOrder = order[i] == i;
            }
        }

        var chunks = new List<BatchChunk>();
        int start = 0, count = 0, width = 0;
        for (int p = 0; p < n; p++)
        {
            int length = lengths[order[p]];
            int newWidth = Math.Max(width, length);
            if (count > 0 && (count >= maxRows || (maxCells is { } budget && (long)(count + 1) * newWidth > budget)))
            {
                chunks.Add(new BatchChunk(start, count, width));
                start = p;
                count = 0;
                newWidth = length;
            }

            count++;
            width = newWidth;
        }

        if (count > 0)
        {
            chunks.Add(new BatchChunk(start, count, width));
        }

        return new BatchPlan(order, isInputOrder, chunks);
    }

    /// <summary>The input indices of <paramref name="chunk"/>'s rows, in call order.</summary>
    public ReadOnlySpan<int> Rows(BatchChunk chunk) => _order.AsSpan(chunk.Start, chunk.Count);
}
