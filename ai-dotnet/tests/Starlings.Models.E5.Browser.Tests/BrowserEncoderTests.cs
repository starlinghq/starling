using Starlings.Models.E5.Browser;

namespace Starlings.Models.E5.Browser.Tests;

public sealed class BrowserEncoderTests
{
    private static EncoderInput Input(params long[] ids) => new(ids);

    private static readonly EncoderInput[] Five =
    [
        Input(0, 10, 11, 2),
        Input(0, 20, 2),
        Input(0, 30, 31, 32, 33, 34, 2),
        Input(0, 2),
        Input(0, 50, 51, 52, 2),
    ];

    [Fact]
    public async Task EmbedsEachInputWithTheLibraryPoolingAndInInputOrder()
    {
        var runner = new FakeGraphRunner();
        using var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions());
        float[][] result = await encoder.EmbedBatchAsync(Five, TestContext.Current.CancellationToken);

        Assert.Equal(5, result.Length);
        for (int i = 0; i < Five.Length; i++)
        {
            Assert.Equal(runner.Expected(Five[i]), result[i]);
        }

        var call = Assert.Single(runner.Calls);
        Assert.Equal((5, 7), (call.Rows, call.Width));
    }

    [Fact]
    public async Task PadsRightWithThePadIdAndMaskZeroAndFeedsZeroTokenTypes()
    {
        var runner = new FakeGraphRunner();
        using var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions());
        await encoder.EmbedBatchAsync([Input(0, 7, 2), Input(0, 2)], TestContext.Current.CancellationToken);

        var call = Assert.Single(runner.Calls);
        Assert.Equal([0, 7, 2, 0, 2, 1], call.Ids);
        Assert.Equal([1, 1, 1, 1, 1, 0], call.Mask);
        Assert.Equal(new long[6], call.Types);
    }

    [Fact]
    public async Task OmitsTokenTypesWhenTheGraphHasNone()
    {
        var runner = new FakeGraphRunner(hasTokenTypeIds: false);
        using var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions());
        await encoder.EmbedAsync(Input(0, 7, 2), TestContext.Current.CancellationToken);
        Assert.Null(Assert.Single(runner.Calls).Types);
    }

    [Fact]
    public async Task ChunksByMaxBatchSizeInInputOrder()
    {
        var runner = new FakeGraphRunner();
        using var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions { MaxBatchSize = 2 });
        float[][] result = await encoder.EmbedBatchAsync(Five, TestContext.Current.CancellationToken);

        Assert.Equal([(2, 4), (2, 7), (1, 5)], runner.Calls.Select(c => (c.Rows, c.Width)));
        for (int i = 0; i < Five.Length; i++)
        {
            Assert.Equal(runner.Expected(Five[i]), result[i]);
        }
    }

    [Fact]
    public async Task SortByLengthScattersResultsBackToInputPositions()
    {
        var runner = new FakeGraphRunner();
        using var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions { MaxBatchSize = 2, SortByLength = true });
        float[][] result = await encoder.EmbedBatchAsync(Five, TestContext.Current.CancellationToken);

        // Lengths 4, 3, 7, 2, 5 sorted: (2, 3), (4, 5), (7).
        Assert.Equal([(2, 3), (2, 5), (1, 7)], runner.Calls.Select(c => (c.Rows, c.Width)));
        for (int i = 0; i < Five.Length; i++)
        {
            Assert.Equal(runner.Expected(Five[i]), result[i]);
        }
    }

    [Fact]
    public async Task ExplicitAttentionMaskExcludesPositionsFromPooling()
    {
        var runner = new FakeGraphRunner();
        using var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions());
        var masked = new EncoderInput(new long[] { 0, 5, 6, 2 }, new long[] { 1, 1, 0, 1 });
        float[] result = await encoder.EmbedAsync(masked, TestContext.Current.CancellationToken);
        Assert.Equal(runner.Expected(masked), result);
        Assert.NotEqual(runner.Expected(Input(0, 5, 6, 2)), result);
    }

    [Theory]
    [InlineData("empty")]
    [InlineData("too-long")]
    [InlineData("id-out-of-range")]
    [InlineData("mask-length")]
    [InlineData("mask-zero")]
    public async Task RejectsAnInvalidInputBeforeAnyRun(string kind)
    {
        var runner = new FakeGraphRunner();
        using var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions { MaxSequenceLength = 8 });
        EncoderInput bad = kind switch
        {
            "empty" => Input(),
            "too-long" => Input(0, 1, 1, 1, 1, 1, 1, 1, 2),
            "id-out-of-range" => Input(0, 250_037, 2),
            "mask-length" => new EncoderInput(new long[] { 0, 2 }, new long[] { 1 }),
            _ => new EncoderInput(new long[] { 0, 2 }, new long[] { 0, 0 }),
        };

        await Assert.ThrowsAnyAsync<ArgumentException>(() => encoder.EmbedBatchAsync([Input(0, 2), bad], TestContext.Current.CancellationToken));
        Assert.ThrowsAny<ArgumentException>(() => encoder.Validate(bad));
        Assert.Empty(runner.Calls);
    }

    [Fact]
    public async Task RejectsADestinationOfTheWrongSize()
    {
        using var encoder = new E5BrowserEncoder(new FakeGraphRunner(), new E5EncoderOptions());
        await Assert.ThrowsAsync<ArgumentException>(() => encoder.EmbedBatchAsync([Input(0, 2)], new float[7], TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ACancelledTokenRunsNothing()
    {
        var runner = new FakeGraphRunner();
        using var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions());
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => encoder.EmbedBatchAsync(Five, cts.Token));
        Assert.Empty(runner.Calls);
    }

    [Fact]
    public async Task CancellationIsCheckedBeforeEachChunk()
    {
        using var cts = new CancellationTokenSource();
        var runner = new FakeGraphRunner { OnRun = n => cts.Cancel() };
        using var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions { MaxBatchSize = 2 });
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => encoder.EmbedBatchAsync(Five, cts.Token));
        Assert.Single(runner.Calls);
    }

    [Fact]
    public async Task DisposeDisposesTheRunnerOnceAndLaterCallsFail()
    {
        var runner = new FakeGraphRunner();
        var encoder = new E5BrowserEncoder(runner, new E5EncoderOptions());
        encoder.Dispose();
        encoder.Dispose();
        Assert.Equal(1, runner.DisposeCount);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => encoder.EmbedAsync(Input(0, 2), TestContext.Current.CancellationToken));
    }

    [Fact]
    public void DimensionComesFromTheRunner()
    {
        using var encoder = new E5BrowserEncoder(new FakeGraphRunner(dimension: 384), null);
        Assert.Equal(384, encoder.Dimension);
    }
}
