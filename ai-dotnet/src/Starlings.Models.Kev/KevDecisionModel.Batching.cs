using Microsoft.ML.OnnxRuntime;

namespace Starlings.Models.Kev;

// Experimental batched question rows (CPU spike). Internal: not on the default path, and capability 'batched_rows'
// stays Unsupported until the maintainer decides (docs/evidence/kev-batched-rows.md).
public sealed partial class KevDecisionModel
{
    /// <summary>
    /// ONNX Runtime's GroupQueryAttention (1.30.0, CPU EP; CUDA not checked) accepts a
    /// non-empty past with more than one new token only at batch 1 ("subsequent prompt"). Cached batched rows are
    /// exactly that shape, so the graph refuses them.
    /// </summary>
    internal const string GroupQueryAttentionBatchLimit = "batch_size must be 1 when sequence_length > 1 and past context is given";

    /// <summary>
    /// Uncached batched rows: every question as one full causal row (state then branch), several rows per graph run,
    /// right-padded with <c>&lt;|endoftext|&gt;</c> under a zero attention mask and position 0, as
    /// <c>model-pipeline/reference/kev/kev_ort.py</c> builds its padded batches. Pads only trail, so no real token
    /// attends to one and the recurrent DeltaNet state of a real token never sees one. Equal to
    /// <see cref="RunUncached"/> within <c>padded_batch_vs_single_probability_max_abs</c>.
    /// </summary>
    internal IReadOnlyList<float[]> RunUncachedBatched(KevEncoding encoding, KevBatchOptions? batch = null, CancellationToken cancellationToken = default)
    {
        CheckEncoding(encoding);
        batch ??= new KevBatchOptions();
        int stateLength = encoding.StateIds.Length;
        var widths = encoding.Rows.Select(r => stateLength + r.Ids.Length).ToArray();
        var result = new float[encoding.Rows.Count][];
        foreach (int[] group in KevBatchPlanner.Plan(widths, 0, batch))
        {
            int b = group.Length;
            int width = group.Max(i => widths[i]);
            var ids = new long[b * width];
            var mask = new long[b * width];
            var pos = new long[b * width];
            Array.Fill(ids, KevModelProfile.PadTokenId);
            for (int r = 0; r < b; r++)
            {
                var row = encoding.Rows[group[r]];
                int at = r * width;
                for (int t = 0; t < stateLength; t++, at++)
                {
                    (ids[at], mask[at], pos[at]) = (encoding.StateIds[t], 1, t);
                }

                for (int t = 0; t < row.Ids.Length; t++, at++)
                {
                    (ids[at], mask[at], pos[at]) = (row.Ids[t], 1, row.Positions[t]);
                }
            }

            var empty = b == 1 ? _emptyPast : TilePast(_emptyPast, b);
            try
            {
                using var outputs = RunBatchGraph(ids, mask, pos, b, width, empty, cancellationToken);
                for (int r = 0; r < b; r++)
                {
                    var row = encoding.Rows[group[r]];
                    result[group[r]] = Readout(outputs[0], row.Decide, row.Options, offset: r * width + stateLength);
                }
            }
            finally
            {
                DisposeTiled(empty);
            }
        }

        return result;
    }

    /// <summary>
    /// Cached batched rows: the state runs once, then several branches per graph run continue from the state's cache
    /// tiled along the batch dimension (ONNX Runtime does not broadcast a batch-1 cache). Branches are right-padded with
    /// <c>&lt;|endoftext|&gt;</c> at continuing positions, and the attention mask is all ones over past plus width:
    /// GroupQueryAttention derives each row's past length from the mask sum, so a zero-masked pad would shift the row's
    /// past. The pads therefore act as trailing tokens of their row, which no real token sees (causal attention,
    /// recurrent state carried forward only). Rows are never left-padded: a leading pad would enter the DeltaNet
    /// convolution and recurrent state before the real tokens. With one row per run this is <see cref="Run"/>.
    /// <para>
    /// <b>Not runnable on ONNX Runtime 1.30.0</b> for more than one row per run: GroupQueryAttention rejects the shape
    /// (<see cref="GroupQueryAttentionBatchLimit"/>), surfaced as <see cref="NotSupportedException"/>. Kept so the layout
    /// is ready, and tested, if the kernel lifts the restriction.
    /// </para>
    /// </summary>
    internal IReadOnlyList<float[]> RunBatched(KevEncoding encoding, KevBatchOptions? batch = null, CancellationToken cancellationToken = default)
    {
        CheckEncoding(encoding);
        batch ??= new KevBatchOptions();
        int stateLength = encoding.StateIds.Length;
        var groups = KevBatchPlanner.Plan([.. encoding.Rows.Select(r => r.Ids.Length)], stateLength, batch);
        using var cache = RunGraph([.. encoding.StateIds], Positions(0, stateLength), 0, _emptyPast, _presentNames, cancellationToken);
        var past = cache.ToArray();
        var result = new float[encoding.Rows.Count][];
        foreach (int[] group in groups)
        {
            int b = group.Length;
            int width = group.Max(i => encoding.Rows[i].Ids.Length);
            var ids = new long[b * width];
            var mask = new long[b * (stateLength + width)];
            var pos = new long[b * width];
            Array.Fill(ids, KevModelProfile.PadTokenId);
            Array.Fill(mask, 1L);
            for (int r = 0; r < b; r++)
            {
                var row = encoding.Rows[group[r]];
                for (int t = 0; t < width; t++)
                {
                    int at = r * width + t;
                    if (t < row.Ids.Length)
                    {
                        (ids[at], pos[at]) = (row.Ids[t], row.Positions[t]);
                    }
                    else
                    {
                        pos[at] = stateLength + t; // trailing pad: the next positions of its own row
                    }
                }
            }

            var tiled = b == 1 ? past : TilePast(past, b);
            try
            {
                using var outputs = RunBatchGraph(ids, mask, pos, b, width, tiled, cancellationToken);
                for (int r = 0; r < b; r++)
                {
                    var row = encoding.Rows[group[r]];
                    result[group[r]] = Readout(outputs[0], row.Decide, row.Options, offset: r * width);
                }
            }
            catch (OnnxRuntimeException e) when (b > 1 && e.Message.Contains(GroupQueryAttentionBatchLimit, StringComparison.Ordinal))
            {
                throw new NotSupportedException(
                    "Cached batched rows need GroupQueryAttention to continue several rows from a past at once, which ONNX Runtime's kernel "
                    + "refuses (\"" + GroupQueryAttentionBatchLimit + "\"). Use one row per run (MaxRows = 1, the sequential path) or the "
                    + "uncached batched path.", e);
            }
            finally
            {
                if (b != 1)
                {
                    DisposeTiled(tiled);
                }
            }
        }

        return result;
    }

    private void DisposeTiled(IReadOnlyList<OrtValue> values)
    {
        if (ReferenceEquals(values, _emptyPast))
        {
            return;
        }

        foreach (var value in values)
        {
            value.Dispose();
        }
    }

    /// <summary>Copies of batch-1 cache tensors repeated <paramref name="batch"/> times along dimension 0.</summary>
    private static OrtValue[] TilePast(OrtValue[] past, int batch)
    {
        var tiled = new OrtValue[past.Length];
        try
        {
            for (int i = 0; i < past.Length; i++)
            {
                long[] shape = [.. past[i].GetTensorTypeAndShape().Shape];
                if (shape[0] != 1)
                {
                    throw new InvalidOperationException($"Cache tensor {i} has batch {shape[0]}; expected 1.");
                }

                var source = past[i].GetTensorDataAsSpan<float>();
                var data = new float[checked(source.Length * batch)];
                for (int r = 0; r < batch; r++)
                {
                    source.CopyTo(data.AsSpan(r * source.Length));
                }

                shape[0] = batch;
                tiled[i] = OrtValue.CreateTensorValueFromMemory(data, shape);
            }

            return tiled;
        }
        catch
        {
            foreach (var value in tiled)
            {
                value?.Dispose();
            }

            throw;
        }
    }

    /// <summary>One graph run over a [batch, width] block. <paramref name="mask"/> is [batch, past + width], <paramref name="positions"/> [batch, width].</summary>
    private IDisposableReadOnlyCollection<OrtValue> RunBatchGraph(
        long[] ids, long[] mask, long[] positions, int batch, int width, IReadOnlyList<OrtValue> past, CancellationToken cancellationToken)
    {
        int cells = batch * width;
        long[] pos = new long[3 * cells];
        for (int axis = 0; axis < 3; axis++)
        {
            positions.CopyTo(pos, axis * cells);
        }

        using var idsValue = OrtValue.CreateTensorValueFromMemory(ids, [batch, width]);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, [batch, mask.Length / batch]);
        using var posValue = OrtValue.CreateTensorValueFromMemory(pos, [3, batch, width]);
        var names = new List<string>(3 + _pastNames.Length) { InputIdsName, AttentionMaskName, PositionIdsName };
        names.AddRange(_pastNames);
        var values = new List<OrtValue>(names.Count) { idsValue, maskValue, posValue };
        values.AddRange(past);
        return _session.Run(names, values, [HiddenStatesName], cancellationToken);
    }
}
