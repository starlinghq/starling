using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// The multimodal embedder's reusable request buffers (<see cref="EmbeddingGemma2BufferPool"/>,
/// docs/evidence/eg2-image-loaded-soak/): the pool itself, and embeddings bit-identical with and without it for single,
/// batched and concurrent requests.
/// </summary>
public class BufferPoolTests(ITestOutputHelper output)
{
    [Fact]
    public void ARentedBufferIsReusedAfterItIsReturned()
    {
        using var pool = new EmbeddingGemma2BufferPool();
        var a = pool.Rent<float>(100_000);
        Assert.True(a.Length >= 100_000);
        pool.Return(a);
        Assert.Same(a, pool.Rent<float>(90_000));

        // Another element type never gets it.
        pool.Return(a);
        Assert.NotSame(a, (object)pool.Rent<int>(10));
    }

    [Fact]
    public void ConcurrentRentsNeverShareABuffer()
    {
        using var pool = new EmbeddingGemma2BufferPool();
        var owners = new System.Collections.Concurrent.ConcurrentDictionary<float[], int>(ReferenceEqualityComparer.Instance);
        Parallel.For(0, 2_000, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i =>
        {
            var b = pool.Rent<float>(1_000 + (i % 7 * 5_000));
            Assert.True(owners.TryAdd(b, i), "a buffer was rented twice");
            b.AsSpan().Fill(i);
            Thread.SpinWait(50);
            Assert.All(b, v => Assert.Equal(i, v));
            Assert.True(owners.TryRemove(b, out _));
            pool.Return(b);
        });
    }

    [Fact]
    public void TheFreeSetIsBoundedAndFollowsTheLargestRequests()
    {
        using var pool = new EmbeddingGemma2BufferPool();
        var many = Enumerable.Range(0, EmbeddingGemma2BufferPool.MaxFreePerType * 3).Select(_ => pool.Rent<float>(50_000)).ToList();
        many.ForEach(pool.Return);
        long bound = (long)EmbeddingGemma2BufferPool.MaxFreePerType * EmbeddingGemma2BufferPool.SizeClass(50_000) * sizeof(float);
        Assert.True(pool.FreeBytes <= bound, $"{pool.FreeBytes} bytes free, bound {bound}");

        // A larger request replaces a buffer that no longer fits instead of adding to the set.
        long before = pool.FreeBytes;
        var big = pool.Rent<float>(200_000);
        pool.Return(big);
        Assert.True(pool.FreeBytes <= before - (EmbeddingGemma2BufferPool.SizeClass(50_000) * sizeof(float)) + (big.Length * sizeof(float)));
    }

    [Fact]
    public void SizeClassesAreAtMostAQuarterAboveTheRequest()
    {
        foreach (int n in new[] { 0, 1, 1_024, 1_025, 4_097, 100_000, 2_359_296, 6_000_001, Array.MaxLength })
        {
            int c = EmbeddingGemma2BufferPool.SizeClass(n);
            Assert.InRange(c, n, Math.Max(n, (long)(n * 1.25) + 1));
        }
    }

    [Fact]
    public void DisposeDropsTheBuffersAndADisabledPoolKeepsNone()
    {
        var pool = new EmbeddingGemma2BufferPool();
        pool.Return(pool.Rent<byte>(1 << 20));
        Assert.True(pool.FreeBytes > 0);
        pool.Dispose();
        Assert.Equal(0, pool.FreeBytes);
        pool.Return(new byte[1 << 20]);
        Assert.Equal(0, pool.FreeBytes);

        using var off = new EmbeddingGemma2BufferPool(enabled: false);
        var a = off.Rent<float>(10_000);
        Assert.Equal(10_000, a.Length);
        off.Return(a);
        Assert.Equal(0, off.FreeBytes);
    }

    [Fact]
    public void ImageAndVideoEmbeddingsAreBitIdenticalWithAndWithoutTheBufferPool()
    {
        var (files, reason) = EmbeddingGemma2ImageTestAssets.FindImageModel();
        Assert.SkipWhen(files is null, reason);
        var ct = TestContext.Current.CancellationToken;
        var inputs = MeasurementInputs.Create();
        var batch = new[] { inputs[0], inputs[3], inputs[1] };

        float[][] Run(bool pooled, out float[][] batched, out float[][] concurrent)
        {
            using var embedder = Create(files!, pooled);

            // Twice over, so the second pass runs on buffers the first one returned.
            var single = inputs.Select(i => embedder.Embed(i, ct)).ToArray();
            var again = inputs.Select(i => embedder.Embed(i, ct)).ToArray();
            for (int i = 0; i < single.Length; i++)
            {
                Assert.Equal(single[i], again[i]);
            }

            batched = embedder.EmbedBatch(batch, EmbeddingGemma2Prompt.None, ct);
            var results = new float[inputs.Length * 2][];
            Parallel.For(0, results.Length, new ParallelOptions { MaxDegreeOfParallelism = 4 }, k =>
                results[k] = embedder.Embed(inputs[k % inputs.Length], ct));
            concurrent = results;
            return single;
        }

        var on = Run(pooled: true, out var onBatch, out var onConcurrent);
        var off = Run(pooled: false, out var offBatch, out var offConcurrent);
        for (int i = 0; i < on.Length; i++)
        {
            Assert.Equal(off[i], on[i]);
        }

        for (int i = 0; i < onBatch.Length; i++)
        {
            Assert.Equal(offBatch[i], onBatch[i]);
        }

        for (int k = 0; k < onConcurrent.Length; k++)
        {
            Assert.Equal(off[k % inputs.Length], onConcurrent[k]);
            Assert.Equal(off[k % inputs.Length], offConcurrent[k]);
        }

        output.WriteLine($"{on.Length} single, {onBatch.Length} batched and {onConcurrent.Length} concurrent embeddings bit-identical with and without the buffer pool");
    }

    private static EmbeddingGemma2MultimodalEmbedder Create(EmbeddingGemma2ImageModelFiles files, bool pooled) =>
        EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions
        {
            Session = new OnnxSessionOptions { ComputeModelSha256 = false },
            VerifyModelSha256 = false,
            PoolBuffers = pooled,
        });
}
