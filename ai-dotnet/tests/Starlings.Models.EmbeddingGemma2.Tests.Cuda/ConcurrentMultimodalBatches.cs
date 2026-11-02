using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests.Cuda;

/// <summary>
/// Capability <c>concurrent_calls</c> over CUDA for the image and video profiles (issue #257; the text profile's check is
/// <see cref="CudaParityTests.ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit"/>): threads released together
/// each run <see cref="EmbeddingGemma2MultimodalEmbedder.EmbedBatch(IReadOnlyList{EmbeddingGemma2MultimodalInput}, EmbeddingGemma2Prompt, CancellationToken)"/>
/// on one embedder (one vision and one multimodal text session) with a different batch. Every vector must equal the
/// sequential result of the same batch on the same embedder bit for bit.
/// </summary>
internal static class ConcurrentMultimodalBatches
{
    /// <summary>
    /// Builds <paramref name="threads"/> batches of <paramref name="batchRows"/> inputs, each a different rotation of
    /// <paramref name="inputs"/> (so their media sizes and token counts differ). The inputs are built (decoded) before any
    /// thread starts.
    /// </summary>
    public static List<List<(string Id, EmbeddingGemma2MultimodalInput Input)>> Rotations(IReadOnlyList<(string Id, EmbeddingGemma2MultimodalInput Input)> inputs, int threads, int batchRows) =>
        Enumerable.Range(0, threads)
            .Select(t => Enumerable.Range(0, batchRows).Select(i => inputs[((t * 5) + (i * 3)) % inputs.Count]).ToList())
            .ToList();

    /// <summary>Runs the check and returns the number of vectors compared and the largest absolute difference (0 when it passes).</summary>
    public static (int Vectors, double MaxAbs) Check(EmbeddingGemma2MultimodalEmbedder embedder, List<List<(string Id, EmbeddingGemma2MultimodalInput Input)>> batches, int rounds, CancellationToken ct)
    {
        int threads = batches.Count;
        var inputs = batches.Select(b => (IReadOnlyList<EmbeddingGemma2MultimodalInput>)b.Select(x => x.Input).ToList()).ToList();
        var sequential = inputs.Select(b => embedder.EmbedBatch(b, EmbeddingGemma2Prompt.None, ct)).ToList();
        int vectors = 0;
        double max = 0;
        for (int round = 0; round < rounds; round++)
        {
            var parallel = new float[threads][][];
            var errors = new Exception?[threads];
            using var start = new Barrier(threads);
            var workers = Enumerable.Range(0, threads).Select(t => new Thread(() =>
            {
                try
                {
                    start.SignalAndWait(ct);
                    parallel[t] = embedder.EmbedBatch(inputs[t], EmbeddingGemma2Prompt.None, ct);
                }
                catch (Exception ex)
                {
                    errors[t] = ex;
                }
            })
            { IsBackground = true, Name = $"embed-{t}" }).ToList();
            workers.ForEach(w => w.Start());
            workers.ForEach(w => w.Join());
            Assert.All(errors, e => Assert.Null(e));
            for (int t = 0; t < threads; t++)
            {
                for (int i = 0; i < batches[t].Count; i++)
                {
                    var stats = new EmbeddingGemma2ErrorStats();
                    stats.Add(batches[t][i].Id, parallel[t][i], sequential[t][i]);
                    max = Math.Max(max, stats.Max);
                    Assert.True(sequential[t][i].AsSpan().SequenceEqual(parallel[t][i]), $"round {round}, thread {t}, row {i} ({batches[t][i].Id}): max abs {stats.Max:E3} against the sequential run");
                    vectors++;
                }
            }
        }

        return (vectors, max);
    }
}
