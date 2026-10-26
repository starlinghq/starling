using Starlings.Models.E5.Browser;

namespace Starlings.Models.E5.Browser.Tests;

/// <summary>
/// A graph runner whose hidden state at (row, position, d) depends only on that position's token id, its index and d,
/// never on the rest of the batch, so pooled results are batch-independent and easy to compute by hand. Records every
/// call.
/// </summary>
internal sealed class FakeGraphRunner(int dimension = 8, bool hasTokenTypeIds = true) : IE5GraphRunner
{
    public int Dimension { get; } = dimension;

    public bool HasTokenTypeIds { get; } = hasTokenTypeIds;

    public List<(int Rows, int Width, long[] Ids, long[] Mask, long[]? Types)> Calls { get; } = [];

    public int DisposeCount { get; private set; }

    /// <summary>Called inside each run, before it writes the output (to cancel mid-batch, for example).</summary>
    public Action<int>? OnRun { get; set; }

    public static float Value(long id, int position, int d) => (float)(Math.Sin((id * 0.37) + (position * 0.11) + (d * 1.3)) + (d * 0.01));

    public Task RunAsync(long[] ids, long[] mask, long[]? types, int rows, int width, Memory<float> hidden, CancellationToken cancellationToken)
    {
        int cells = rows * width;
        Calls.Add((rows, width, ids[..cells], mask[..cells], types?[..cells]));
        OnRun?.Invoke(Calls.Count);
        Assert.Equal(cells * Dimension, hidden.Length);
        var span = hidden.Span;
        for (int c = 0; c < cells; c++)
        {
            for (int d = 0; d < Dimension; d++)
            {
                span[(c * Dimension) + d] = Value(ids[c], c % width, d);
            }
        }

        return Task.CompletedTask;
    }

    public void Dispose() => DisposeCount++;

    /// <summary>What the encoder must return for <paramref name="input"/>: the library's pooling on the unpadded hidden state.</summary>
    public float[] Expected(EncoderInput input)
    {
        var ids = input.InputIds.Span;
        var hidden = new float[ids.Length * Dimension];
        for (int t = 0; t < ids.Length; t++)
        {
            for (int d = 0; d < Dimension; d++)
            {
                hidden[(t * Dimension) + d] = Value(ids[t], t, d);
            }
        }

        long[] mask = input.AttentionMask.IsEmpty ? [.. Enumerable.Repeat(1L, ids.Length)] : input.AttentionMask.ToArray();
        var result = new float[Dimension];
        E5Encoder.MeanPoolAndNormalize(hidden, mask, Dimension, result);
        return result;
    }
}
