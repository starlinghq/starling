using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// <see cref="Gliner2ExtractorOptions.MaxBatchTokens"/> and <see cref="Gliner2ExtractorOptions.SortByLength"/> on the
/// weight-free route A graph (each row's logits depend only on its own word positions, so batches must equal single
/// calls exactly). Needs the pinned tokenizer.json, not the model.
/// </summary>
public class BatchingOptionsTests
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };

    private static readonly string[] Labels = ["person", "location"];

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static Gliner2EntityExtractor Fake(Gliner2ExtractorOptions? options = null)
    {
        var (tokenizer, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizer is null, reason);
        return Gliner2EntityExtractor.Create(OnnxModelSession.Create(TestModels.FakeGliner2, Fast), tokenizer!, options, ownsSession: true);
    }

    /// <summary>Texts of 0 to <paramref name="maxWords"/> words, in random length order.</summary>
    private static List<string> MixedTexts(int count, int maxWords, int seed)
    {
        string[] vocabulary = ["Ann", "met", "Bob", "in", "Paris", "and", "Zoë", "went", "to", "Malmö", "yesterday", "with", "friends"];
        var random = new Random(seed);
        return Enumerable.Range(0, count)
            .Select(_ => string.Join(' ', Enumerable.Range(0, random.Next(0, maxWords + 1)).Select(_ => vocabulary[random.Next(vocabulary.Length)])) + ".")
            .ToList();
    }

    private static void AssertSameOutput(Gliner2ModelOutput expected, Gliner2ModelOutput actual)
    {
        Assert.Equal((expected.LabelCount, expected.WordCount, expected.MaxWidth), (actual.LabelCount, actual.WordCount, actual.MaxWidth));
        Assert.Equal(expected.CountLogits.ToArray(), actual.CountLogits.ToArray());
        Assert.Equal(expected.SpanLogits.ToArray(), actual.SpanLogits.ToArray());
    }

    [Fact]
    public void DefaultsKeepTheQualifiedBehaviour()
    {
        var options = new Gliner2ExtractorOptions();
        Assert.Null(options.MaxBatchTokens);
        Assert.False(options.SortByLength);
        Assert.Equal(8, options.MaxBatchSize);
        Assert.Equal(Status.Supported, Gliner2Capabilities.Statuses["batch"]);
        Assert.Equal(Status.Experimental, Gliner2Capabilities.Statuses["batch.token_budget"]);
        Assert.Equal(Status.Experimental, Gliner2Capabilities.Statuses["batch.sort_by_length"]);
        Assert.Throws<ArgumentOutOfRangeException>(() => Fake(new Gliner2ExtractorOptions { MaxBatchTokens = 0 }));
    }

    [Fact]
    public void PlanSortsByTokenCountAndRespectsTheBudget()
    {
        using var extractor = Fake();
        var inputs = MixedTexts(30, 25, seed: 4).Select(t => extractor.Prepare(t, Labels)).ToList();
        int longest = inputs.Max(i => i.TokenCount);
        int budget = longest - 1; // below the longest text, which must then run alone
        var options = new Gliner2ExtractorOptions { MaxBatchSize = 5, MaxBatchTokens = budget, SortByLength = true };
        var plan = Gliner2EntityExtractor.Plan(inputs, options);
        Assert.Equal(Enumerable.Range(0, inputs.Count), plan.Order.Order());
        for (int p = 1; p < plan.Order.Count; p++)
        {
            int a = plan.Order[p - 1], b = plan.Order[p];
            Assert.True(inputs[a].TokenCount < inputs[b].TokenCount || (inputs[a].TokenCount == inputs[b].TokenCount && a < b));
        }

        Assert.All(plan.Chunks, c => Assert.True(c.Count <= 5 && (c.Count == 1 || c.PaddedCells <= budget), $"{c}"));
        Assert.Equal((1, longest), (plan.Chunks[^1].Count, plan.Chunks[^1].Width));

        var defaults = Gliner2EntityExtractor.Plan(inputs, new Gliner2ExtractorOptions());
        Assert.True(defaults.IsInputOrder);
        Assert.Equal([8, 8, 8, 6], defaults.Chunks.Select(c => c.Count));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void SortedAndBudgetedBatchesReturnResultsInInputOrder(int seed)
    {
        // Mixed neighbours: 0 to 40 words in random order; a budget below the longest text makes it run alone.
        var texts = MixedTexts(23, 40, seed);
        using var single = Fake(new Gliner2ExtractorOptions { MaxBatchSize = 1 });
        var inputs = texts.Select(t => single.Prepare(t, Labels)).ToList();
        int budget = inputs.Max(i => i.TokenCount) * 2 - 1; // the longest text cannot share a call
        using var sorted = Fake(new Gliner2ExtractorOptions { MaxBatchSize = 4, MaxBatchTokens = budget, SortByLength = true });

        var expected = single.RunModel(inputs, Ct);
        var actual = sorted.RunModel(inputs, Ct);
        for (int i = 0; i < inputs.Count; i++)
        {
            AssertSameOutput(expected[i], actual[i]);
        }

        var permutation = Enumerable.Range(0, texts.Count).ToArray();
        new Random(seed * 17).Shuffle(permutation);
        var permuted = sorted.ExtractBatch(permutation.Select(i => texts[i]).ToList(), Labels, cancellationToken: Ct);
        for (int k = 0; k < permutation.Length; k++)
        {
            var reference = single.Extract(texts[permutation[k]], Labels, cancellationToken: Ct);
            Assert.Equal(texts[permutation[k]], permuted[k].Text);
            Assert.Equal(reference.Entities, permuted[k].Entities);
        }
    }
}
