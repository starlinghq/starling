namespace Starlings.Models.Bert.Tests;

/// <summary>
/// The optional token budget and length-sorted chunking of the BERT session path, without a model: the plan, and the
/// scatter of chunk results back to input order through <see cref="BertSession.RunPlanned"/> with a recording runner.
/// </summary>
public sealed class BatchingOptionsTests
{
    /// <summary>A valid unpadded row of <paramref name="length"/> tokens whose second id encodes <paramref name="tag"/>.</summary>
    private static BertInput Row(int length, int tag)
    {
        var ids = new long[length];
        ids[0] = 101;
        for (int i = 1; i < length - 1; i++)
        {
            ids[i] = 1000 + tag;
        }

        ids[^1] = 102;
        return new BertInput(ids, Enumerable.Repeat(1L, length).ToArray(), new long[length]);
    }

    private static List<BertInput> MixedRows(int count, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(i => Row(random.Next(3, 60), i)).ToList();
    }

    private static int Tag(BertInput input) => (int)(input.InputIds.Span[1] - 1000);

    [Theory]
    [InlineData(1, 32, null, false)]
    [InlineData(2, 8, null, true)]
    [InlineData(3, 32, 200, false)]
    [InlineData(4, 16, 300, true)]
    [InlineData(5, 128, 59, true)]
    public void ResultsReturnInInputOrderAndChunksStayWithinTheLimits(int seed, int maxRows, int? maxCells, bool sort)
    {
        // Mixed neighbours: random lengths 3..59 in random order.
        var inputs = MixedRows(45, seed);
        var plan = BertSession.Plan(inputs, maxRows, maxCells, sort);
        var calls = new List<(int Count, int Width)>();
        var results = BertSession.RunPlanned(inputs, plan, (chunk, width) =>
        {
            Assert.Equal(chunk.Max(i => i.Length), width);
            calls.Add((chunk.Length, width));
            return chunk.Select(Tag).ToArray();
        }, TestContext.Current.CancellationToken);

        Assert.Equal(Enumerable.Range(0, inputs.Count), results);
        Assert.Equal(plan.Chunks.Count, calls.Count);
        Assert.All(calls, c => Assert.True(c.Count <= maxRows && (maxCells is null || c.Count == 1 || (long)c.Count * c.Width <= maxCells), $"{c}"));
        if (sort)
        {
            Assert.True(calls.Select(c => c.Width).SequenceEqual(calls.Select(c => c.Width).Order()), "sorted chunks are in ascending width");
        }
        else
        {
            Assert.True(plan.IsInputOrder);
        }
    }

    [Fact]
    public void ARowLongerThanTheBudgetRunsAlone()
    {
        BertInput[] inputs = [Row(5, 0), Row(5, 1), Row(40, 2), Row(5, 3)];
        var plan = BertSession.Plan(inputs, 32, 20, sortByLength: false);
        Assert.Equal([(0, 2, 5), (2, 1, 40), (3, 1, 5)], plan.Chunks.Select(c => (c.Start, c.Count, c.Width)));
    }

    [Fact]
    public void CancellationIsCheckedBeforeEachChunk()
    {
        var inputs = MixedRows(10, 7);
        using var cts = new CancellationTokenSource();
        int calls = 0;
        Assert.ThrowsAny<OperationCanceledException>(() => BertSession.RunPlanned(inputs, BertSession.Plan(inputs, 3, null, true), (chunk, _) =>
        {
            calls++;
            cts.Cancel();
            return new int[chunk.Length];
        }, cts.Token));
        Assert.Equal(1, calls);
    }

    [Fact]
    public void InvalidBatchingIsRejectedAndCapabilitiesAreExplicit()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => BertSession.ValidateBatching(0, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => BertSession.ValidateBatching(129, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => BertSession.ValidateBatching(8, 0));
        BertSession.ValidateBatching(8, null);
        BertSession.ValidateBatching(8, 1);
        Assert.Equal("experimental", BertCapabilities.BatchTokenBudget);
        Assert.Equal("experimental", BertCapabilities.BatchSortByLength);
    }
}
