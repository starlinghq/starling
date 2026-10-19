namespace Starlings.Benchmarks.Tests;

public sealed class BatchSweepTests
{
    private static ArgMap Args(params string[] args) => ArgMap.Parse(args, "usage");

    [Fact]
    public void PaddingCountsRowsTimesLongestRowPerRun()
    {
        int[] lengths = [10, 3, 7, 2, 9];
        var (cells, runs) = SweepMath.Padding(lengths, SweepMath.FixedChunks(lengths.Length, 2));
        Assert.Equal(3, runs);
        Assert.Equal((2 * 10) + (2 * 7) + (1 * 9), cells);

        var (one, oneRun) = SweepMath.Padding(lengths, SweepMath.FixedChunks(lengths.Length, 64));
        Assert.Equal(1, oneRun);
        Assert.Equal(5 * 10, one);
    }

    [Fact]
    public void FixedChunksCoverEveryRowInOrder()
    {
        Assert.Equal([32, 32, 1], SweepMath.FixedChunks(65, 32));
        Assert.Equal([1], SweepMath.FixedChunks(1, 8));
    }

    [Fact]
    public void DefaultsAreTheFullGridWithOneRunPerCall()
    {
        var o = SweepOptions.Parse(Args("--out", "x.jsonl"));
        Assert.Equal([1, 4, 8, 16, 32, 64], o.Batches);
        Assert.Equal([1, 4, 16], o.Concurrency);
        Assert.Equal(["short", "long", "mixed"], o.Sets);
        Assert.Equal(64, o.MaxBatchSize);
        Assert.Null(o.Pool);
    }

    [Fact]
    public void QuickAndMaxBatchSizeModes()
    {
        var quick = SweepOptions.Parse(Args("--out", "x", "--quick", "true", "--max-batch-size", "default"));
        Assert.Equal([1, 8, 32], quick.Batches);
        Assert.Equal([1, 4], quick.Concurrency);
        Assert.Null(quick.MaxBatchSize);
        Assert.Equal(32, quick.Pool);
        Assert.Equal(8, SweepOptions.Parse(Args("--out", "x", "--max-batch-size", "8")).MaxBatchSize);
        Assert.Throws<ArgumentException>(() => SweepOptions.Parse(Args("--out", "x", "--max-batch-size", "0")));
        Assert.Throws<ArgumentException>(() => SweepOptions.Parse(Args("--batches", "1")));
    }

    [Theory]
    [InlineData("sort-by-length", "true")]
    [InlineData("max-batch-tokens", "4096")]
    [InlineData("microbatch-wait-ms", "2")]
    [InlineData("microbatch-max-rows", "32")]
    [InlineData("microbatch-max-tokens", "4096")]
    public void ContractHooksAreRejectedWhereNotWired(string flag, string value)
    {
        var ex = Assert.Throws<ArgumentException>(() => SweepOptions.Parse(Args("--out", "x", "--" + flag, value)));
        Assert.Contains("not wired", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWiredHookIsAccepted()
    {
        var o = SweepOptions.Parse(Args("--out", "x", "--max-batch-tokens", "4096"), ["max-batch-tokens"]);
        Assert.Equal(4096, o.MaxBatchTokens);
    }

    [Fact]
    public void PlanWithoutSortOrBudgetIsTheFixedChunks()
    {
        int[] lengths = [10, 3, 7, 2, 9];
        Assert.Equal(SweepMath.Padding(lengths, SweepMath.FixedChunks(lengths.Length, 2)), SweepMath.Plan(lengths, 2, null, false));
        Assert.Equal(SweepMath.Padding(lengths, SweepMath.FixedChunks(lengths.Length, 64)), SweepMath.Plan(lengths, 64, null, false));
    }

    [Fact]
    public void PlanSortsAndAppliesTheTokenBudget()
    {
        int[] lengths = [10, 3, 7, 2, 9];
        // Sorted 2, 3, 7, 9, 10 in pairs: 2x3 + 2x9 + 1x10.
        Assert.Equal((2L * 3 + 2 * 9 + 10, 3), SweepMath.Plan(lengths, 2, null, true));
        // Budget 20 cells, input order: [10, 3] = 20; [7, 2] = 14 (adding 9 would make 27); [9] = 9.
        Assert.Equal((20L + 14 + 9, 3), SweepMath.Plan(lengths, 8, 20, false));
        // A row longer than the budget runs alone.
        Assert.Equal((10L + 3 + 7 + 2 + 9, 5), SweepMath.Plan(lengths, 8, 1, false));
        // Sorted with a budget of 24: [2, 3, 7] = 21; [9, 10] = 20.
        Assert.Equal((21L + 20, 2), SweepMath.Plan(lengths, 8, 24, true));
    }

    [Fact]
    public void BatchingHooksMapToOptionsAndAConfigLabel()
    {
        var all = SweepOptions.AllHooks;
        Assert.Equal("baseline", SweepOptions.Parse(Args("--out", "x"), all).Config);
        var sorted = SweepOptions.Parse(Args("--out", "x", "--sort-by-length", "true", "--max-batch-tokens", "2048"), all);
        Assert.True(sorted.SortByLength);
        Assert.False(sorted.MicroBatching);
        Assert.Equal("sorted+tokens2048", sorted.Config);
        Assert.Throws<ArgumentException>(() => SweepOptions.Parse(Args("--out", "x", "--sort-by-length", "yes"), all));

        var micro = SweepOptions.Parse(Args("--out", "x", "--batches", "1", "--microbatch-wait-ms", "2"), all);
        Assert.True(micro.MicroBatching);
        Assert.Equal(2, micro.MicroBatchWaitMs);
        Assert.Equal(SweepOptions.DefaultMicroBatchMaxRows, micro.MicroBatchRows);
        Assert.Equal("micro-w2-r8", micro.Config);
        // "call" rows per ORT run cover a whole dispatched batch, not the caller's single item.
        Assert.Equal(8, micro.MaxBatchSize);
        Assert.Equal("call (8)", micro.MaxBatchSizeLabel);

        var rows = SweepOptions.Parse(Args("--out", "x", "--batches", "1", "--microbatch-max-rows", "16", "--microbatch-max-tokens", "4096"), all);
        Assert.Equal("micro-w0-r16-t4096", rows.Config);
        Assert.Equal(16, rows.MaxBatchSize);
        Assert.Throws<ArgumentException>(() => SweepOptions.Parse(Args("--out", "x", "--microbatch-wait-ms", "-1"), all));
        Assert.Throws<ArgumentException>(() => SweepOptions.Parse(Args("--out", "x", "--microbatch-max-rows", "0"), all));
    }

    [Fact]
    public void BatcherCountersSubtract()
    {
        var a = new SweepBatcherStats(10, 80, 4000, 3000, 50, 900);
        var b = new SweepBatcherStats(4, 30, 1000, 800, 20, 300);
        Assert.Equal(new SweepBatcherStats(6, 50, 3000, 2200, 30, 600), a.Since(b));
    }

    [Fact]
    public void ChecksMergeToTheWorstItem()
    {
        var a = new SweepCheck(2, 1e-7, true, 0, "a");
        var b = new SweepCheck(3, 4e-7, true, 1, "b");
        var m = a.Merge(b).Merge(SweepCheck.Empty);
        Assert.Equal(5, m.Items);
        Assert.Equal(4e-7, m.MaxDiff);
        Assert.Equal(1, m.Mismatches);
        Assert.Equal("b", m.Worst);
        Assert.False(m.Merge(new SweepCheck(1, 0, false, 0, "c")).Finite);
    }

    [Fact]
    public void EmbeddingCheckFindsTheWorstRowAndNonFiniteValues()
    {
        SweepItem[] items = [new("a", "x", "query"), new("b", "y", "passage")];
        var singles = new Dictionary<string, object> { ["a"] = new[] { 0.5f, 0.5f }, ["b"] = new[] { 1f, 0f } };
        var same = SweepCheck.Embeddings(items, [[0.5f, 0.5f], [1f, 0f]], singles);
        Assert.Equal(0, same.MaxDiff);
        Assert.Equal(2, same.Items);

        var off = SweepCheck.Embeddings(items, [[0.5f, 0.5f], [1f, 3e-5f]], singles);
        Assert.Equal(3e-5, off.MaxDiff, 1e-9);
        Assert.Equal("b", off.Worst);
        Assert.False(SweepCheck.Embeddings(items, [[float.NaN, 0.5f], [1f, 0f]], singles).Finite);
    }

    /// <summary>A stand-in token count for pool tests: one token per four UTF-16 units, plus two.</summary>
    private static int FakeTokens(SweepItem item) => (item.Text.Length / 4) + 2;

    [Theory]
    [InlineData("multilingual-e5-small")]
    [InlineData("gliner2-base-v1")]
    public void PoolsAreDeterministicSizedAndWithinLimits(string model)
    {
        var file = SweepWorkloadFile.Load(null);
        var a = file.Build(model, FakeTokens, 512, 24);
        var b = file.Build(model, FakeTokens, 512, 24);
        foreach (string set in new[] { "short", "medium", "long", "mixed" })
        {
            Assert.Equal(a.Pools[set].Select(i => i.Id), b.Pools[set].Select(i => i.Id));
            Assert.Equal(24, a.Pools[set].Length);
            Assert.All(a.Pools[set], i => Assert.InRange(a.Tokens(i), 1, 512));
        }

        double target = model == "gliner2-base-v1" ? 480 : 512;
        Assert.All(a.Pools["long"], i => Assert.InRange(a.Tokens(i), 0.75 * target, target));
        Assert.True(a.Pools["short"].Average(a.Tokens) < a.Pools["medium"].Average(a.Tokens));

        // Mixed draws from the short and long categories (and medium).
        var kinds = a.Pools["mixed"].Select(i => i.Id[0]).ToHashSet();
        Assert.Contains(model == "gliner2-base-v1" ? 's' : 'q', kinds);
        Assert.Contains('l', kinds);
    }
}
