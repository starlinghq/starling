using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// <see cref="EmbeddingGemma2EncoderOptions.SortByLength"/> with the token budget on the weight-free text graph (each
/// row's token embeddings depend only on its own tokens, so batches must equal single calls). Runs in CI.
/// </summary>
public class BatchingOptionsTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static List<EmbeddingGemma2Input> MixedRows(int count, int maxLength, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count)
            .Select(i => new EmbeddingGemma2Input(Enumerable.Range(0, random.Next(2, maxLength + 1)).Select(t => (long)(((i * 13) + t) % 50) + 3).ToArray()))
            .ToList();
    }

    [Fact]
    public void DefaultsKeepInputOrderChunking()
    {
        Assert.False(new EmbeddingGemma2EncoderOptions().SortByLength);
        Assert.False(new EmbeddingGemma2MultimodalOptions().SortByLength);
        var inputs = MixedRows(40, 30, seed: 3);
        var plan = EmbeddingGemma2Encoder.Plan(inputs, 16, 8_192, sortByLength: false);
        Assert.True(plan.IsInputOrder);
        Assert.Equal(EmbeddingGemma2Encoder.PlanChunks(inputs, 16, 8_192), plan.Chunks.Select(c => c.Count));
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["batch.sort_by_length"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, EmbeddingGemma2Capabilities.All["batch"]);
    }

    [Fact]
    public void SortedPlanIsStableAndWithinTheBudget()
    {
        var inputs = MixedRows(60, 40, seed: 9);
        var plan = EmbeddingGemma2Encoder.Plan(inputs, maxRows: 6, maxCells: 100, sortByLength: true);
        Assert.Equal(Enumerable.Range(0, inputs.Count), plan.Order.Order());
        for (int p = 1; p < plan.Order.Count; p++)
        {
            int a = plan.Order[p - 1], b = plan.Order[p];
            Assert.True(inputs[a].Length < inputs[b].Length || (inputs[a].Length == inputs[b].Length && a < b));
        }

        Assert.All(plan.Chunks, c => Assert.True(c.Count <= 6 && (c.Count == 1 || c.PaddedCells <= 100)));
        Assert.True(plan.PaddedCells <= EmbeddingGemma2Encoder.Plan(inputs, 6, 100, sortByLength: false).PaddedCells);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void SortedBatchesReturnRowsInInputOrder(int seed)
    {
        // Mixed neighbours: random lengths in random order; a 1-row budget case (a row longer than the budget) included.
        var inputs = MixedRows(37, 48, seed);
        using var reference = EncoderBehaviorTests.FakeEncoder(new EmbeddingGemma2EncoderOptions { MaxBatchSize = 1 });
        using var sorted = EncoderBehaviorTests.FakeEncoder(new EmbeddingGemma2EncoderOptions { MaxBatchSize = 5, MaxBatchTokens = 40, SortByLength = true });
        using var defaults = EncoderBehaviorTests.FakeEncoder();

        var expected = reference.EmbedBatch(inputs, Ct);
        Assert.Equal(expected, sorted.EmbedBatch(inputs, Ct));
        Assert.Equal(expected, defaults.EmbedBatch(inputs, Ct));

        var permutation = Enumerable.Range(0, inputs.Count).ToArray();
        new Random(seed * 31).Shuffle(permutation);
        var permuted = sorted.EmbedBatch(permutation.Select(i => inputs[i]).ToList(), Ct);
        for (int k = 0; k < permutation.Length; k++)
        {
            Assert.Equal(expected[permutation[k]], permuted[k]);
        }

        var destination = new float[inputs.Count * sorted.Dimension];
        sorted.EmbedBatch(inputs, destination, Ct);
        for (int i = 0; i < inputs.Count; i++)
        {
            Assert.Equal(expected[i], destination.AsSpan(i * sorted.Dimension, sorted.Dimension).ToArray());
        }
    }

    [Fact]
    public void MultimodalOptionsPassSortByLengthToTheTextEncoder()
    {
        var options = new EmbeddingGemma2MultimodalOptions { SortByLength = true, MaxBatchSize = 4, MaxBatchTokens = 512 };
        var text = options.ToTextEncoderOptions(null);
        Assert.True(text.SortByLength);
        Assert.Equal(4, text.MaxBatchSize);
        Assert.Equal(512, text.MaxBatchTokens);
        Assert.False(new EmbeddingGemma2MultimodalOptions().ToTextEncoderOptions(null).SortByLength);
    }
}
