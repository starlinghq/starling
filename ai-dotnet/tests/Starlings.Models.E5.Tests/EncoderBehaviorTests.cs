using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// Batching, padding, pooling, validation and ownership on a weight-free graph with the e5 signature whose hidden
/// state is <c>[id, 1000 * token_type_id, 1, 0.5 * mask]</c>, so pooled results have a closed form. Runs in CI.
/// </summary>
public class EncoderBehaviorTests
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };

    private static E5Encoder FakeEncoder(E5EncoderOptions? options = null, byte[]? model = null) =>
        new(OnnxModelSession.Create(model ?? TestModels.FakeE5, Fast), options, ownsSession: true);

    /// <summary>normalize([mean id over mask, 0, 1, 0.5]).</summary>
    private static float[] Expected(long[] ids, long[]? mask = null)
    {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < ids.Length; i++)
        {
            if (mask is null || mask[i] == 1)
            {
                sum += ids[i];
                count++;
            }
        }

        double[] v = [sum / count, 0, 1, 0.5];
        double norm = Math.Sqrt(v.Sum(x => x * x));
        return v.Select(x => (float)(x / norm)).ToArray();
    }

    [Fact]
    public void DimensionAndTokenTypeInputComeFromTheGraph()
    {
        using var encoder = FakeEncoder();
        Assert.Equal(4, encoder.Dimension);
        Assert.Equal(["input_ids", "attention_mask", "token_type_ids"], encoder.Session.Inputs.Select(i => i.Name));
    }

    [Fact]
    public void SingleInputIsMeanPooledAndNormalized()
    {
        using var encoder = FakeEncoder();
        long[] ids = [0, 10, 2];

        var embedding = encoder.Embed(new EncoderInput(ids), TestContext.Current.CancellationToken);

        Assert.Equal(Expected(ids), embedding);
        Assert.Equal(1.0, Math.Sqrt(embedding.Sum(x => (double)x * x)), 6);
    }

    [Fact]
    public void TokenTypeIdsAreFedAsZeros()
    {
        using var encoder = FakeEncoder();
        var embedding = encoder.Embed(new EncoderInput(new long[] { 0, 7, 2 }), TestContext.Current.CancellationToken);
        Assert.Equal(0f, embedding[1]);
    }

    [Fact]
    public void GraphWithoutTokenTypeIdsIsSupported()
    {
        using var encoder = FakeEncoder(model: TestModels.FakeE5NoTokenTypes);
        Assert.Equal(["input_ids", "attention_mask"], encoder.Session.Inputs.Select(i => i.Name));
        long[] ids = [0, 10, 2];
        Assert.Equal(Expected(ids), encoder.Embed(new EncoderInput(ids), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void BatchRowsArePaddedAndPaddingIsExcludedFromPooling()
    {
        using var encoder = FakeEncoder();
        long[] shortRow = [0, 10, 2];
        long[] longRow = [0, 100, 200, 300, 400, 2];

        var batch = encoder.EmbedBatch([new EncoderInput(shortRow), new EncoderInput(longRow)], TestContext.Current.CancellationToken);

        Assert.Equal(Expected(shortRow), batch[0]);
        Assert.Equal(Expected(longRow), batch[1]);
        Assert.Equal(encoder.Embed(new EncoderInput(shortRow), TestContext.Current.CancellationToken), batch[0]);
    }

    [Fact]
    public void PackingRightPadsWithPadIdAndZeroMask()
    {
        EncoderInput[] inputs =
        [
            new(new long[] { 0, 5, 2 }),
            new(new long[] { 0, 6, 7, 8, 2 }),
            new(new long[] { 0, 9, 1, 1 }, new long[] { 1, 1, 0, 0 }),
        ];
        var ids = new long[15];
        var mask = new long[15];

        E5Encoder.PackRightPadded(inputs, 0, 3, 5, 1, ids, mask);

        Assert.Equal([0L, 5, 2, 1, 1, 0, 6, 7, 8, 2, 0, 9, 1, 1, 1], ids);
        Assert.Equal([1L, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], mask);
    }

    [Fact]
    public void ExplicitMaskExcludesMaskedPositions()
    {
        using var encoder = FakeEncoder();
        long[] ids = [0, 10, 999, 2];
        long[] mask = [1, 1, 0, 1];

        var embedding = encoder.Embed(new EncoderInput(ids, mask), TestContext.Current.CancellationToken);

        Assert.Equal(Expected(ids, mask), embedding);
    }

    [Fact]
    public void LargeBatchesRunInChunksWithIdenticalResults()
    {
        var inputs = Enumerable.Range(0, 11)
            .Select(i => new EncoderInput(Enumerable.Range(0, 2 + (i % 5)).Select(t => (long)(t * (i + 1))).ToArray()))
            .ToList();
        using var chunked = FakeEncoder(new E5EncoderOptions { MaxBatchSize = 3 });
        using var whole = FakeEncoder();

        var a = chunked.EmbedBatch(inputs, TestContext.Current.CancellationToken);
        var b = whole.EmbedBatch(inputs, TestContext.Current.CancellationToken);

        Assert.Equal(b, a);
        for (int i = 0; i < inputs.Count; i++)
        {
            Assert.Equal(Expected(inputs[i].InputIds.ToArray()), a[i]);
        }
    }

    [Fact]
    public void SpanOverloadWritesRowMajor()
    {
        using var encoder = FakeEncoder();
        EncoderInput[] inputs = [new(new long[] { 0, 4, 2 }), new(new long[] { 0, 40, 2 })];
        var destination = new float[8];

        encoder.EmbedBatch(inputs, destination, TestContext.Current.CancellationToken);

        Assert.Equal(Expected([0, 4, 2]), destination[..4]);
        Assert.Equal(Expected([0, 40, 2]), destination[4..]);
        Assert.Throws<ArgumentException>(() => encoder.EmbedBatch(inputs, new float[7], TestContext.Current.CancellationToken));
    }

    [Fact]
    public void EmptyBatchReturnsEmpty()
    {
        using var encoder = FakeEncoder();
        Assert.Empty(encoder.EmbedBatch([], TestContext.Current.CancellationToken));
    }

    public static TheoryData<long[], long[]?> InvalidInputs => new()
    {
        { [], null },
        { Enumerable.Repeat(5L, 513).ToArray(), null },
        { [0, 5, 2], [1, 1] },
        { [0, 5, 2], [1, 2, 1] },
        { [0, 5, 2], [0, 0, 0] },
        { [0, -1, 2], null },
        { [0, 250_037, 2], null },
    };

    [Theory]
    [MemberData(nameof(InvalidInputs))]
    public void InvalidInputsAreRejectedBeforeAnyInference(long[] ids, long[]? mask)
    {
        using var encoder = FakeEncoder(new E5EncoderOptions { MaxBatchSize = 1 });
        var bad = mask is null ? new EncoderInput(ids) : new EncoderInput(ids, mask);
        var destination = new float[8];

        var batch = Assert.Throws<ArgumentException>(() => encoder.EmbedBatch([new EncoderInput(new long[] { 0, 3, 2 }), bad], destination, TestContext.Current.CancellationToken));

        // The valid first row would have been written by the first chunk if validation ran lazily.
        Assert.All(destination, v => Assert.Equal(0f, v));

        // ParamName names the caller's parameter: the batch for EmbedBatch, the input for Validate.
        Assert.Equal("inputs", batch.ParamName);
        Assert.StartsWith("Input 1 ", batch.Message, StringComparison.Ordinal);
        var single = Assert.Throws<ArgumentException>(() => encoder.Validate(bad));
        Assert.Equal("input", single.ParamName);
        Assert.StartsWith("Input 0 ", single.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ProfileConstantsAreTheDefaults()
    {
        Assert.Equal(512, E5Encoder.MaxSupportedSequenceLength);
        Assert.Equal(384, E5Encoder.NativeDimension);
        Assert.Equal(E5Encoder.MaxSupportedSequenceLength, new E5EncoderOptions().MaxSequenceLength);
    }

    [Fact]
    public void MaximumLengthIsAccepted()
    {
        using var encoder = FakeEncoder();
        var ids = Enumerable.Repeat(5L, 512).ToArray();
        Assert.Equal(Expected(ids), encoder.Embed(new EncoderInput(ids), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void GraphWithoutE5SignatureIsRejected()
    {
        using var session = OnnxModelSession.Create(TestModels.Add, Fast);
        Assert.Throws<NotSupportedException>(() => new E5Encoder(session));
    }

    [Fact]
    public void InvalidOptionsAreRejected()
    {
        using var session = OnnxModelSession.Create(TestModels.FakeE5, Fast);
        Assert.Throws<ArgumentOutOfRangeException>(() => new E5Encoder(session, new E5EncoderOptions { MaxBatchSize = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new E5Encoder(session, new E5EncoderOptions { MaxSequenceLength = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new E5Encoder(session, new E5EncoderOptions { PadTokenId = -1 }));
    }

    [Fact]
    public void OwnedSessionIsDisposedWithTheEncoder()
    {
        var encoder = FakeEncoder();
        encoder.Dispose();
        encoder.Dispose();
        Assert.True(encoder.Session.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => encoder.Embed(new EncoderInput(new long[] { 0, 2 }), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void BorrowedSessionOutlivesTheEncoder()
    {
        using var session = OnnxModelSession.Create(TestModels.FakeE5, Fast);
        var first = new E5Encoder(session);
        first.Dispose();
        Assert.False(session.IsDisposed);

        using var second = new E5Encoder(session);
        Assert.Equal(Expected([0, 3, 2]), second.Embed(new EncoderInput(new long[] { 0, 3, 2 }), TestContext.Current.CancellationToken));

        session.Dispose();
        Assert.Throws<ObjectDisposedException>(() => second.Embed(new EncoderInput(new long[] { 0, 2 }), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void CancelledTokenThrowsBeforeRunning()
    {
        using var encoder = FakeEncoder();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => encoder.Embed(new EncoderInput(new long[] { 0, 2 }), cts.Token));
    }

    [Fact]
    public void ConcurrentCallsOnSharedEncoderAreIsolated()
    {
        using var encoder = FakeEncoder();
        int failures = 0;
        Parallel.For(0, 8, t =>
        {
            for (int i = 0; i < 100; i++)
            {
                long[] a = [0, (t * 1000) + i, 2];
                long[] b = [0, i, i, i, 2];
                var batch = encoder.EmbedBatch([new EncoderInput(a), new EncoderInput(b)]);
                if (!batch[0].SequenceEqual(Expected(a)) || !batch[1].SequenceEqual(Expected(b)))
                {
                    Interlocked.Increment(ref failures);
                }
            }
        });
        Assert.Equal(0, failures);
    }

    [Fact]
    public void ZeroHiddenStateNormalizesToZeroWithoutNaN()
    {
        var destination = new float[3];
        E5Encoder.MeanPoolAndNormalize(new float[6], new long[] { 1, 1 }, 3, destination);
        Assert.Equal([0f, 0, 0], destination);
    }

    [Fact]
    public void EveryCapabilityHasAnExplicitStatus()
    {
        var all = E5Capabilities.All;
        Assert.All(all.Values, s => Assert.True(Enum.IsDefined(s)));
        Assert.Equal(CapabilityStatus.Supported, all["embedding.text"]);
        Assert.Equal(CapabilityStatus.Unsupported, all["tf32"]);
        Assert.Equal(CapabilityStatus.Unsupported, all["streaming"]);
        Assert.Equal(CapabilityStatus.Unsupported, all["truncation"]);
        Assert.Equal(CapabilityStatus.Supported, all["extensions_ai.embedding_generator"]);
        Assert.Equal(CapabilityStatus.Unsupported, all["extensions_ai.dimensions_truncation"]);
    }
}
