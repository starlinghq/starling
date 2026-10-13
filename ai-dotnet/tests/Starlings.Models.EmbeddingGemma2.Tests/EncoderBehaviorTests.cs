using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// Batching, padding, pooling, Matryoshka truncation, validation and ownership on a weight-free graph with the
/// EmbeddingGemma 2 text-path signature whose token embeddings are <c>id * (1 + d / 768) + mask * cos(d)</c>, so pooled
/// results have a closed form. Runs in CI.
/// </summary>
public class EncoderBehaviorTests
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };

    internal static EmbeddingGemma2Encoder FakeEncoder(EmbeddingGemma2EncoderOptions? options = null, byte[]? model = null) =>
        new(OnnxModelSession.Create(model ?? TestModels.FakeEmbeddingGemma2, Fast), options, ownsSession: true);

    /// <summary>The first <paramref name="dimension"/> values of the masked mean of the closed-form token embeddings, normalized.</summary>
    internal static float[] Expected(long[] ids, long[]? mask = null, int dimension = 768)
    {
        // Padding added by the encoder has mask 0 and is excluded, so only the given positions matter.
        var sum = new double[768];
        int count = 0;
        for (int t = 0; t < ids.Length; t++)
        {
            if (mask is not null && mask[t] == 0)
            {
                continue;
            }

            count++;
            for (int d = 0; d < 768; d++)
            {
                sum[d] += (float)((float)ids[t] * (float)(1.0 + (d / 768.0))) + (float)Math.Cos(d);
            }
        }

        double norm = 0;
        for (int d = 0; d < dimension; d++)
        {
            sum[d] /= count;
            norm += sum[d] * sum[d];
        }

        norm = Math.Sqrt(norm);
        return sum.Take(dimension).Select(x => (float)(x / norm)).ToArray();
    }

    private static void Near(float[] expected, float[] actual, double bound = 2e-6)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= bound, $"[{i}] {actual[i]} vs {expected[i]}");
        }
    }

    [Fact]
    public void SignatureAndDefaults()
    {
        using var encoder = FakeEncoder();
        Assert.Equal(768, encoder.Dimension);
        Assert.Equal(8192, encoder.Options.MaxSequenceLength);
        Assert.Equal(["input_ids", "attention_mask"], encoder.Session.Inputs.Select(i => i.Name));
        Assert.Equal([768, 512, 256, 128], EmbeddingGemma2Encoder.SupportedDimensions);
    }

    [Fact]
    public void SingleInputIsMeanPooledOverEveryTokenAndNormalized()
    {
        using var encoder = FakeEncoder();
        long[] ids = [2, 10, 20, 1];
        var embedding = encoder.Embed(new EmbeddingGemma2Input(ids), TestContext.Current.CancellationToken);
        Near(Expected(ids), embedding);
        Assert.Equal(1.0, EmbeddingGemma2ParityChecks.Norm(embedding), 6);
    }

    [Fact]
    public void BatchRowsArePaddedAndPaddingIsExcludedFromPooling()
    {
        using var encoder = FakeEncoder();
        long[] shortRow = [2, 10, 1];
        long[] longRow = [2, 100, 200, 300, 400, 1];
        var ct = TestContext.Current.CancellationToken;
        var batch = encoder.EmbedBatch([new EmbeddingGemma2Input(shortRow), new EmbeddingGemma2Input(longRow)], ct);
        Near(Expected(shortRow), batch[0]);
        Near(Expected(longRow), batch[1]);
        Assert.Equal(encoder.Embed(new EmbeddingGemma2Input(shortRow), ct), batch[0]);
    }

    [Fact]
    public void PackingRightPadsWithPadIdZeroAndZeroMask()
    {
        EmbeddingGemma2Input[] inputs =
        [
            new(new long[] { 2, 5, 1 }),
            new(new long[] { 2, 6, 7, 8, 1 }),
            new(new long[] { 2, 9, 0, 0 }, new long[] { 1, 1, 0, 0 }),
        ];
        var ids = new long[15];
        var mask = new long[15];
        Array.Fill(ids, -1);
        Array.Fill(mask, -1);
        EmbeddingGemma2Encoder.PackRightPadded(inputs, 0, 3, 5, ids, mask);
        Assert.Equal([2L, 5, 1, 0, 0, 2, 6, 7, 8, 1, 2, 9, 0, 0, 0], ids);
        Assert.Equal([1L, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], mask);
    }

    [Fact]
    public void ExplicitMaskExcludesMaskedPositions()
    {
        using var encoder = FakeEncoder();
        long[] ids = [2, 10, 999, 1];
        long[] mask = [1, 1, 0, 1];
        Near(Expected(ids, mask), encoder.Embed(new EmbeddingGemma2Input(ids, mask), TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData(512)]
    [InlineData(256)]
    [InlineData(128)]
    public void MatryoshkaKeepsTheLeadingDimensionsAndRenormalizes(int dimension)
    {
        using var full = FakeEncoder();
        using var truncated = FakeEncoder(new EmbeddingGemma2EncoderOptions { Dimension = dimension });
        long[] ids = [2, 17, 33, 4000, 1];
        var ct = TestContext.Current.CancellationToken;
        var t = truncated.Embed(new EmbeddingGemma2Input(ids), ct);
        Assert.Equal(dimension, truncated.Dimension);
        Assert.Equal(dimension, t.Length);
        Near(Expected(ids, dimension: dimension), t);
        Near(EmbeddingGemma2Encoder.Truncate(full.Embed(new EmbeddingGemma2Input(ids), ct), dimension), t, 1e-7);
        Assert.Equal(1.0, EmbeddingGemma2ParityChecks.Norm(t), 6);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(64)]
    [InlineData(300)]
    [InlineData(384)]
    [InlineData(767)]
    [InlineData(769)]
    [InlineData(1024)]
    public void OtherDimensionsAreRejected(int dimension)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => FakeEncoder(new EmbeddingGemma2EncoderOptions { Dimension = dimension }));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2Encoder.Truncate(new float[768], dimension));
    }

    [Fact]
    public void TruncateCannotWiden()
    {
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2Encoder.Truncate(new float[256], 512));
    }

    [Fact]
    public void InputsOver8192TokensAreRejectedWithATypedExceptionBeforeAnyInference()
    {
        using var encoder = FakeEncoder();
        var tooLong = new long[8193];
        tooLong[0] = 2;
        tooLong[^1] = 1;
        var ex = Assert.Throws<EmbeddingGemma2InputTooLongException>(() =>
            encoder.EmbedBatch([new EmbeddingGemma2Input(new long[] { 2, 1 }), new EmbeddingGemma2Input(tooLong)], TestContext.Current.CancellationToken));
        Assert.Equal(1, ex.InputIndex);
        Assert.Equal(8193, ex.TokenCount);
        Assert.Equal(8192, ex.MaximumTokens);
        Assert.IsAssignableFrom<ArgumentException>(ex);
    }

    [Fact]
    public void ExactlyTheMaximumIsAccepted()
    {
        using var encoder = FakeEncoder(new EmbeddingGemma2EncoderOptions { MaxSequenceLength = 64 });
        var ids = Enumerable.Repeat(7L, 64).ToArray();
        Assert.Equal(768, encoder.Embed(new EmbeddingGemma2Input(ids), TestContext.Current.CancellationToken).Length);
        Assert.Throws<EmbeddingGemma2InputTooLongException>(() => encoder.Embed(new EmbeddingGemma2Input(Enumerable.Repeat(7L, 65).ToArray()), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void TheLimitCannotBeRaisedAbove8192()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => FakeEncoder(new EmbeddingGemma2EncoderOptions { MaxSequenceLength = 8193 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => FakeEncoder(new EmbeddingGemma2EncoderOptions { MaxSequenceLength = 0 }));
    }

    [Theory]
    [InlineData(255_999L)]
    [InlineData(256_000L)]
    [InlineData(258_880L)]
    [InlineData(258_881L)]
    [InlineData(258_882L)]
    [InlineData(258_883L)]
    [InlineData(258_884L)]
    public void MediaPlaceholderAndBoundaryTokensAreRejected(long id)
    {
        using var encoder = FakeEncoder();
        var ex = Assert.Throws<ArgumentException>(() => encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 5, id, 1 }), TestContext.Current.CancellationToken));
        Assert.Contains("text only", ex.Message, StringComparison.Ordinal);
        Assert.Contains(id, EmbeddingGemma2Encoder.MediaTokenIds);
    }

    [Fact]
    public void NeighboursOfMediaTokensAreAccepted()
    {
        using var encoder = FakeEncoder();
        var ct = TestContext.Current.CancellationToken;
        foreach (long id in new long[] { 255_998, 256_001, 258_879, 258_885, 262_143 })
        {
            Assert.Equal(768, encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, id, 1 }), ct).Length);
        }
    }

    [Fact]
    public void InvalidInputsAreRejectedBeforeAnyInference()
    {
        using var encoder = FakeEncoder();
        var ct = TestContext.Current.CancellationToken;
        Assert.Throws<ArgumentException>(() => encoder.Embed(new EmbeddingGemma2Input(Array.Empty<long>()), ct));
        Assert.Throws<ArgumentException>(() => encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 262_144, 1 }), ct));
        Assert.Throws<ArgumentException>(() => encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, -1, 1 }), ct));
        Assert.Throws<ArgumentException>(() => encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 5, 1 }, new long[] { 1, 1 }), ct));
        Assert.Throws<ArgumentException>(() => encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 5, 1 }, new long[] { 1, 2, 1 }), ct));
        Assert.Throws<ArgumentException>(() => encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 5, 1 }, new long[] { 0, 0, 0 }), ct));
    }

    [Fact]
    public void ChunkPlanRespectsRowsAndPaddedCells()
    {
        static EmbeddingGemma2Input Of(int length) => new(new long[length]);
        var inputs = Enumerable.Repeat(10, 5).Select(Of).ToList();
        Assert.Equal([2, 2, 1], EmbeddingGemma2Encoder.PlanChunks(inputs, maxRows: 2, maxCells: 1000));
        Assert.Equal([3, 2], EmbeddingGemma2Encoder.PlanChunks(inputs, maxRows: 16, maxCells: 30));

        // A long row closes the chunk when rows x width would exceed the budget; a row over the budget runs alone.
        int[] lengths = [5, 5, 100, 5, 5];
        var mixed = lengths.Select(Of).ToList();
        Assert.Equal([2, 1, 2], EmbeddingGemma2Encoder.PlanChunks(mixed, maxRows: 16, maxCells: 64));
        Assert.Equal([5], EmbeddingGemma2Encoder.PlanChunks(mixed, maxRows: 16, maxCells: 500));
        Assert.Empty(EmbeddingGemma2Encoder.PlanChunks([], 16, 64));
    }

    [Fact]
    public void ChunkedBatchesGiveTheSameResults()
    {
        var inputs = Enumerable.Range(0, 13)
            .Select(i => new EmbeddingGemma2Input(Enumerable.Range(0, 2 + (i % 5)).Select(t => (long)(t * (i + 1))).ToArray()))
            .ToList();
        using var chunked = FakeEncoder(new EmbeddingGemma2EncoderOptions { MaxBatchSize = 3, MaxBatchTokens = 12 });
        using var whole = FakeEncoder();
        var ct = TestContext.Current.CancellationToken;
        var a = chunked.EmbedBatch(inputs, ct);
        var b = whole.EmbedBatch(inputs, ct);
        Assert.Equal(b, a);
        for (int i = 0; i < inputs.Count; i++)
        {
            Near(Expected(inputs[i].InputIds.ToArray()), a[i]);
        }
    }

    [Fact]
    public void SpanOverloadWritesRowMajorAndChecksTheSize()
    {
        using var encoder = FakeEncoder(new EmbeddingGemma2EncoderOptions { Dimension = 128 });
        EmbeddingGemma2Input[] inputs = [new(new long[] { 2, 4, 1 }), new(new long[] { 2, 40, 1 })];
        var destination = new float[256];
        encoder.EmbedBatch(inputs, destination, TestContext.Current.CancellationToken);
        Near(Expected([2, 4, 1], dimension: 128), destination[..128]);
        Near(Expected([2, 40, 1], dimension: 128), destination[128..]);
        Assert.Throws<ArgumentException>(() => encoder.EmbedBatch(inputs, new float[255], TestContext.Current.CancellationToken));
    }

    [Fact]
    public void AnotherGraphSignatureIsRejected()
    {
        var session = OnnxModelSession.Create(TestModels.FakeE5, Fast);
        try
        {
            Assert.Throws<NotSupportedException>(() => new EmbeddingGemma2Encoder(session));
        }
        finally
        {
            session.Dispose();
        }
    }

    [Fact]
    public void CancelledTokenThrowsBeforeInference()
    {
        using var encoder = FakeEncoder();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 1 }), cts.Token));
        Assert.Equal(768, encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 1 }), TestContext.Current.CancellationToken).Length);
    }

    [Fact]
    public void ConcurrentCallsGiveTheSequentialResults()
    {
        using var encoder = FakeEncoder();
        var inputs = Enumerable.Range(1, 32).Select(i => new EmbeddingGemma2Input(Enumerable.Range(0, i).Select(t => (long)(t + i)).ToArray())).ToList();
        var ct = TestContext.Current.CancellationToken;
        var sequential = inputs.Select(x => encoder.Embed(x, ct)).ToList();
        var parallel = new float[inputs.Count][];
        Parallel.For(0, inputs.Count, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = ct }, i => parallel[i] = encoder.Embed(inputs[i], ct));
        Assert.Equal(sequential, parallel);
    }

    [Fact]
    public void DisposeIsIdempotentAndLaterCallsThrow()
    {
        var encoder = FakeEncoder();
        encoder.Dispose();
        encoder.Dispose();
        Assert.True(encoder.Session.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 1 }), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void AWrappedSessionIsNotDisposed()
    {
        using var session = OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2, Fast);
        var encoder = new EmbeddingGemma2Encoder(session);
        encoder.Dispose();
        Assert.False(session.IsDisposed);
    }
}
