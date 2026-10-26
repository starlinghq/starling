using System.Collections;
using Starlings.Batching;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// <see cref="E5EncoderOptions.MaxBatchTokens"/> and <see cref="E5EncoderOptions.SortByLength"/> on the weight-free
/// e5 graph (each row's hidden state depends only on its own tokens, so results must equal single calls exactly)
/// and the shared chunk planner. Runs in CI.
/// </summary>
public class BatchingOptionsTests
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static E5Encoder FakeEncoder(E5EncoderOptions? options = null) =>
        new(OnnxModelSession.Create(TestModels.FakeE5, Fast), options, ownsSession: true);

    /// <summary>Rows of random lengths 2..maxLength with ids that identify the row.</summary>
    private static List<EncoderInput> MixedRows(int count, int maxLength, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count)
            .Select(i => new EncoderInput(Enumerable.Range(0, random.Next(2, maxLength + 1)).Select(t => (long)((i * 37) + t + 3)).ToArray()))
            .ToList();
    }

    /// <summary>The plan's invariants: a permutation, sorted when asked, chunks within both limits unless alone.</summary>
    internal static void AssertPlanInvariants(BatchPlan plan, int[] lengths, int maxRows, int? maxCells, bool sorted)
    {
        Assert.Equal(Enumerable.Range(0, lengths.Length), plan.Order.Order());
        if (sorted)
        {
            for (int p = 1; p < plan.Order.Count; p++)
            {
                int a = plan.Order[p - 1], b = plan.Order[p];
                Assert.True(lengths[a] < lengths[b] || (lengths[a] == lengths[b] && a < b), $"position {p}: not stably sorted");
            }
        }
        else
        {
            Assert.Equal(Enumerable.Range(0, lengths.Length), plan.Order);
        }

        int next = 0;
        foreach (var chunk in plan.Chunks)
        {
            Assert.Equal(next, chunk.Start);
            Assert.InRange(chunk.Count, 1, maxRows);
            Assert.Equal(plan.Order.Skip(chunk.Start).Take(chunk.Count).Max(i => lengths[i]), chunk.Width);
            if (maxCells is { } budget && chunk.Count > 1)
            {
                Assert.True(chunk.PaddedCells <= budget, $"chunk at {chunk.Start}: {chunk.Count} x {chunk.Width} > {budget}");
            }

            next += chunk.Count;
        }

        Assert.Equal(lengths.Length, next);
    }

    [Fact]
    public void DefaultPlanIsConsecutiveChunksOfMaxBatchSize()
    {
        int[] lengths = [5, 1, 9, 3, 3, 7, 2];
        var plan = BatchPlan.Create(lengths, maxRows: 3, maxCells: null, sortByLength: false);
        Assert.True(plan.IsInputOrder);
        Assert.Equal([(0, 3, 9), (3, 3, 7), (6, 1, 2)], plan.Chunks.Select(c => (c.Start, c.Count, c.Width)));
    }

    [Fact]
    public void SortedPlanIsStableAscendingAndChunksTheSortedOrder()
    {
        int[] lengths = [5, 1, 9, 3, 3, 7, 2];
        var plan = BatchPlan.Create(lengths, maxRows: 3, maxCells: null, sortByLength: true);
        Assert.False(plan.IsInputOrder);
        Assert.Equal([1, 6, 3, 4, 0, 5, 2], plan.Order);
        Assert.Equal([(0, 3, 3), (3, 3, 7), (6, 1, 9)], plan.Chunks.Select(c => (c.Start, c.Count, c.Width)));
        Assert.True(plan.PaddedCells < BatchPlan.Create(lengths, 3, null, false).PaddedCells);

        // Already sorted input keeps the identity order.
        Assert.True(BatchPlan.Create([1, 2, 2, 5], 3, null, sortByLength: true).IsInputOrder);
    }

    [Fact]
    public void TokenBudgetClosesChunksAndARowLongerThanTheBudgetRunsAlone()
    {
        int[] lengths = [4, 4, 4, 20, 4, 4];
        var plan = BatchPlan.Create(lengths, maxRows: 8, maxCells: 12, sortByLength: false);
        Assert.Equal([(0, 3, 4), (3, 1, 20), (4, 2, 4)], plan.Chunks.Select(c => (c.Start, c.Count, c.Width)));
        AssertPlanInvariants(plan, lengths, 8, 12, sorted: false);

        // A budget of one cell: every row alone.
        Assert.All(BatchPlan.Create(lengths, 8, 1, false).Chunks, c => Assert.Equal(1, c.Count));
        Assert.Empty(BatchPlan.Create([], 8, 12, true).Chunks);
    }

    [Theory]
    [InlineData(1, 4, null, false)]
    [InlineData(2, 4, null, true)]
    [InlineData(3, 32, 64, false)]
    [InlineData(4, 32, 64, true)]
    [InlineData(5, 3, 40, true)]
    [InlineData(6, 128, 512, true)]
    public void RandomPlansKeepTheirInvariants(int seed, int maxRows, int? maxCells, bool sort)
    {
        var random = new Random(seed);
        for (int trial = 0; trial < 50; trial++)
        {
            var lengths = Enumerable.Range(0, random.Next(0, 80)).Select(_ => random.Next(1, 100)).ToArray();
            AssertPlanInvariants(BatchPlan.Create(lengths, maxRows, maxCells, sort), lengths, maxRows, maxCells, sort);
        }
    }

    [Fact]
    public void EncoderPlanUsesTheOptions()
    {
        var inputs = MixedRows(40, 30, seed: 11);
        var lengths = inputs.Select(i => i.Length).ToArray();
        var options = new E5EncoderOptions { MaxBatchSize = 6, MaxBatchTokens = 60, SortByLength = true };
        AssertPlanInvariants(E5Encoder.Plan(inputs, options), lengths, 6, 60, sorted: true);
        AssertPlanInvariants(E5Encoder.Plan(inputs, new E5EncoderOptions()), lengths, 32, null, sorted: false);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void SortedAndBudgetedBatchesReturnRowsInInputOrder(int seed)
    {
        // Mixed neighbours: random lengths 2..40 in random order, so every chunk mixes rows from all over the input.
        var inputs = MixedRows(57, 40, seed);
        using var reference = FakeEncoder(new E5EncoderOptions { MaxBatchSize = 1 });
        using var sorted = FakeEncoder(new E5EncoderOptions { MaxBatchSize = 7, MaxBatchTokens = 120, SortByLength = true });
        using var budgeted = FakeEncoder(new E5EncoderOptions { MaxBatchSize = 32, MaxBatchTokens = 50 });
        using var defaults = FakeEncoder();

        var expected = reference.EmbedBatch(inputs, Ct);
        Assert.Equal(expected, sorted.EmbedBatch(inputs, Ct));
        Assert.Equal(expected, budgeted.EmbedBatch(inputs, Ct));
        Assert.Equal(expected, defaults.EmbedBatch(inputs, Ct));

        // A random permutation of the same rows comes back permuted the same way.
        var permutation = Enumerable.Range(0, inputs.Count).ToArray();
        new Random(seed * 7).Shuffle(permutation);
        var permuted = sorted.EmbedBatch(permutation.Select(i => inputs[i]).ToList(), Ct);
        for (int k = 0; k < permutation.Length; k++)
        {
            Assert.Equal(expected[permutation[k]], permuted[k]);
        }
    }

    [Fact]
    public void SortedSpanOverloadWritesEachRowAtItsInputPosition()
    {
        var inputs = MixedRows(9, 12, seed: 5);
        using var sorted = FakeEncoder(new E5EncoderOptions { MaxBatchSize = 2, SortByLength = true });
        using var reference = FakeEncoder();
        var destination = new float[inputs.Count * sorted.Dimension];
        sorted.EmbedBatch(inputs, destination, Ct);
        var expected = reference.EmbedBatch(inputs, Ct);
        for (int i = 0; i < inputs.Count; i++)
        {
            Assert.Equal(expected[i], destination.AsSpan(i * 4, 4).ToArray());
        }
    }

    [Fact]
    public void CancellationAfterTheFirstSortedChunkLeavesOnlyItsRowsWritten()
    {
        // Lengths 6, 2, 5, 3: sorted order 1, 3, 2, 0; chunks of two: {1, 3} then {2, 0}.
        long[][] ids = [[0, 9, 9, 9, 9, 2], [0, 2], [0, 8, 8, 8, 2], [0, 7, 2]];
        var inputs = ids.Select(i => new EncoderInput(i)).ToList();
        using var encoder = FakeEncoder(new E5EncoderOptions { MaxBatchSize = 2, SortByLength = true });
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(Ct);

        // Input 2 is read once to validate, once to plan, and a third time when the second chunk is packed: cancel
        // then, after the first chunk has run and before the second chunk's run.
        var list = new CancelOnRead(inputs, index: 2, reads: 3, cts);
        var destination = new float[inputs.Count * 4];
        Assert.ThrowsAny<OperationCanceledException>(() => encoder.EmbedBatch(list, destination, cts.Token));

        using var reference = FakeEncoder();
        Assert.Equal(reference.Embed(inputs[1], Ct), destination.AsSpan(4, 4).ToArray());
        Assert.Equal(reference.Embed(inputs[3], Ct), destination.AsSpan(12, 4).ToArray());
        Assert.All(destination.AsSpan(0, 4).ToArray(), v => Assert.Equal(0f, v));
        Assert.All(destination.AsSpan(8, 4).ToArray(), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void InvalidTokenBudgetIsRejected()
    {
        using var session = OnnxModelSession.Create(TestModels.FakeE5, Fast);
        Assert.Throws<ArgumentOutOfRangeException>(() => new E5Encoder(session, new E5EncoderOptions { MaxBatchTokens = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new E5Encoder(session, new E5EncoderOptions { MaxBatchTokens = -5 }));
    }

    [Fact]
    public void DefaultsKeepTheQualifiedBehaviour()
    {
        var options = new E5EncoderOptions();
        Assert.Null(options.MaxBatchTokens);
        Assert.False(options.SortByLength);
        Assert.Equal(32, options.MaxBatchSize);
        Assert.Equal(Status.Supported, E5Capabilities.Statuses["batch"]);
        Assert.Equal(Status.Experimental, E5Capabilities.Statuses["batch.token_budget"]);
        Assert.Equal(Status.Experimental, E5Capabilities.Statuses["batch.sort_by_length"]);
    }

    /// <summary>A list that cancels <paramref name="cts"/> on the n-th read of one index.</summary>
    private sealed class CancelOnRead(IReadOnlyList<EncoderInput> inner, int index, int reads, CancellationTokenSource cts) : IReadOnlyList<EncoderInput>
    {
        private int _reads;

        public int Count => inner.Count;

        public EncoderInput this[int i]
        {
            get
            {
                if (i == index && ++_reads == reads)
                {
                    cts.Cancel();
                }

                return inner[i];
            }
        }

        public IEnumerator<EncoderInput> GetEnumerator() => inner.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
