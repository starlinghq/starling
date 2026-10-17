using Starlings.Models.Gliner2;
using Starlings.Runtime.Onnx;

namespace Starlings.TestSupport;

/// <summary>
/// One extractor over the real model per test class, with the graph's SHA-256 enforced, plus the protected development fixtures.
/// When the model, the tokenizer or the fixtures are absent the fixture holds a skip reason and the tests skip.
/// </summary>
public abstract class Gliner2ModelFixture : IDisposable
{
    private readonly Gliner2EntityExtractor? _extractor;
    private readonly IReadOnlyList<Gliner2Case>? _cases;

    protected Gliner2ModelFixture(OnnxSessionOptions sessionOptions, Func<string?>? gate = null)
    {
        string? gateReason = gate?.Invoke();
        if (gateReason is not null)
        {
            SkipReason = gateReason;
            return;
        }

        var (files, reason) = Gliner2TestAssets.FindModel();
        if (files is null)
        {
            SkipReason = reason;
            return;
        }

        var (fixtures, fixtureReason) = Gliner2TestAssets.FindExtractionFixtures();
        FixtureSkipReason = fixtures is null ? fixtureReason : null;
        _cases = fixtures is null ? null : Gliner2Fixtures.LoadCases(fixtures);
        _extractor = Gliner2EntityExtractor.Create(files, new Gliner2ExtractorOptions
        {
            Session = sessionOptions with { ExpectedModelSha256 = Gliner2ModelAssets.ModelSha256, CapturePlacement = true },
        });
    }

    /// <summary>Why the model tests skip, or null when the model is loaded.</summary>
    public string? SkipReason { get; }

    /// <summary>Why the fixture-based tests skip, or null.</summary>
    public string? FixtureSkipReason { get; }

    /// <summary>The extractor; skips the calling test when the model is unavailable.</summary>
    public Gliner2EntityExtractor Extractor
    {
        get
        {
            Assert.SkipWhen(_extractor is null, SkipReason ?? "model unavailable");
            return _extractor!;
        }
    }

    /// <summary>The protected fixture cases; skips the calling test when they or the model are unavailable.</summary>
    public IReadOnlyList<Gliner2Case> Cases
    {
        get
        {
            _ = Extractor;
            Assert.SkipWhen(_cases is null, FixtureSkipReason ?? "fixtures unavailable");
            return _cases!;
        }
    }

    public void Dispose()
    {
        _extractor?.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>Summary of a parity run over the in-contract fixture cases.</summary>
internal sealed record Gliner2ParitySummary(
    int Cases,
    int InputsExact,
    int EntitiesExact,
    int EntitiesWithinRules,
    int AllowedSwaps,
    int AllowedFlips,
    Gliner2Errors CountLogits,
    Gliner2Errors SpanLogits,
    Gliner2Errors Scores,
    Gliner2Errors Confidence,
    Dictionary<string, Gliner2ModelOutput> Outputs);

internal static class Gliner2ModelParity
{
    /// <summary>
    /// Every in-contract case, one by one: discrete inputs exact; count logits, gate, span logits, scores, entities and
    /// confidences against the reference under the protected tolerances; public offsets inside the input and equal to the
    /// reference mapped onto the input.
    /// </summary>
    public static Gliner2ParitySummary SingleCases(Gliner2EntityExtractor extractor, IReadOnlyList<Gliner2Case> cases, Gliner2Tolerances tolerances, ITestOutputHelper output)
    {
        var count = new Gliner2Errors();
        var span = new Gliner2Errors();
        var scores = new Gliner2Errors();
        var confidence = new Gliner2Errors();
        var outputs = new Dictionary<string, Gliner2ModelOutput>(StringComparer.Ordinal);
        int n = 0, inputsExact = 0, exact = 0, within = 0, swaps = 0, flips = 0;
        var failures = new List<string>();
        foreach (var c in cases)
        {
            if (c.OutOfContract is not null)
            {
                continue;
            }

            n++;
            var input = extractor.Prepare(c.Text, c.LabelsRequested, c.Descriptions);
            bool inputOk = input.NormalizedText == c.NormalizedText
                && input.Words.SequenceEqual(c.Words) && input.WordStarts.SequenceEqual(c.StartMap) && input.WordEnds.SequenceEqual(c.EndMap)
                && input.SchemaTokens.SequenceEqual(c.SchemaTokens) && input.Labels.SequenceEqual(c.Labels)
                && input.InputIds.SequenceEqual(c.InputIds) && input.WordIndex.SequenceEqual(c.WordIndex) && input.SchemaIndex.SequenceEqual(c.SchemaIndex);
            if (inputOk)
            {
                inputsExact++;
            }
            else
            {
                failures.Add($"{c.Id}: inputs differ");
            }

            var o = extractor.RunModel([input], TestContext.Current.CancellationToken)[0];
            outputs[c.Id] = o;
            count.Add(o.CountLogits, c.CountLogits, c.Id);
            if ((o.PredictedCount > 0) != c.GateOpen)
            {
                failures.Add($"{c.Id}: gate {(o.PredictedCount > 0)} vs reference {c.GateOpen}");
            }

            var valid = o.GetValidSpanLogits();
            span.Add(valid, c.SpanLogitsValid, c.Id);
            double scoreDiff = 0;
            for (int i = 0; i < valid.Length; i++)
            {
                scoreDiff = Math.Max(scoreDiff, Math.Abs((double)Sigmoid(valid[i]) - Sigmoid(c.SpanLogitsValid[i])));
            }

            scores.Add(scoreDiff, c.Id);

            var upstream = Gliner2ParityChecks.UpstreamEntities(input, o, c.Threshold);
            var comparison = Gliner2ParityChecks.CompareEntities(c, upstream, tolerances);
            exact += comparison.Exact ? 1 : 0;
            within += comparison.WithinRules ? 1 : 0;
            swaps += comparison.AllowedSwaps;
            flips += comparison.AllowedFlips;
            confidence.Add(comparison.ConfidenceMaxAbs, c.Id);
            if (!comparison.WithinRules)
            {
                failures.Add($"{c.Id}: {comparison.Detail}");
            }

            // Public result: offsets inside the input, Text == input[Start..End], equal to the reference mapped onto the input.
            var result = extractor.Decode(input, o, c.Threshold);
            foreach (var e in result.Entities)
            {
                Assert.InRange(e.Start, 0, c.Text.Length);
                Assert.InRange(e.End, e.Start + 1, c.Text.Length);
                Assert.Equal(c.Text[e.Start..e.End], e.Text);
            }

            if (comparison.Exact)
            {
                var mapped = Gliner2ParityChecks.MapReferenceToOriginal(c);
                var actual = result.Entities.Select(e => (e.Label, e.Text, e.ScalarStart, e.ScalarEnd)).ToList();
                if (!mapped.SequenceEqual(actual))
                {
                    failures.Add($"{c.Id}: public entities differ from the reference mapped onto the input");
                }
            }
        }

        output.WriteLine($"in-contract cases {n}; inputs exact {inputsExact}; entities exact {exact}, within rules {within} (allowed swaps {swaps}, allowed flips {flips})");
        output.WriteLine($"count_logits {count}; span_logits(valid) {span}; scores {scores}; confidence {confidence}");
        output.WriteLine($"bounds: {tolerances.Source}");
        foreach (var f in failures)
        {
            output.WriteLine("FAIL " + f);
        }

        Assert.Empty(failures);
        Assert.False(count.NonFinite || span.NonFinite, "NaN or Inf in logits");
        Assert.True(count.MaxAbs <= tolerances.CountLogitsMaxAbs, $"count_logits max abs {count.MaxAbs:E2}");
        Assert.True(span.MaxRelative <= tolerances.SpanLogitsRelative, $"span_logits relative {span.MaxRelative:E2}");
        Assert.True(scores.MaxAbs <= tolerances.ScoresMaxAbs, $"scores max abs {scores.MaxAbs:E2}");
        Assert.True(confidence.MaxAbs <= tolerances.ConfidenceMaxAbs, $"confidence max abs {confidence.MaxAbs:E2}");
        return new Gliner2ParitySummary(n, inputsExact, exact, within, swaps, flips, count, span, scores, confidence, outputs);
    }

    /// <summary>
    /// Padded batches (grouped by label count, the graph's input contract; up to 8 rows) against the same cases run
    /// alone: span logits within the protected relative bound, entities identical.
    /// </summary>
    public static Gliner2Errors PaddedBatches(Gliner2EntityExtractor extractor, IReadOnlyList<Gliner2Case> cases, IReadOnlyDictionary<string, Gliner2ModelOutput> singles, Gliner2Tolerances tolerances, ITestOutputHelper output)
    {
        var errors = new Gliner2Errors();
        int rows = 0;
        foreach (var group in cases.Where(c => c.OutOfContract is null).GroupBy(c => c.Labels.Count))
        {
            var list = group.ToList();
            var inputs = list.Select(c => extractor.Prepare(c.Text, c.LabelsRequested, c.Descriptions)).ToList();
            var batched = extractor.RunModel(inputs, TestContext.Current.CancellationToken);
            for (int i = 0; i < list.Count; i++)
            {
                errors.Add(batched[i].GetValidSpanLogits(), singles[list[i].Id].GetValidSpanLogits(), list[i].Id);
                var a = Gliner2ParityChecks.UpstreamEntities(inputs[i], batched[i], list[i].Threshold).Select(e => e.Key);
                var b = Gliner2ParityChecks.UpstreamEntities(inputs[i], singles[list[i].Id], list[i].Threshold).Select(e => e.Key);
                Assert.Equal(b, a);
                rows++;
            }
        }

        output.WriteLine($"padded batches (max {extractor.Options.MaxBatchSize} rows, grouped by label count), {rows} rows vs single runs: span logits {errors}; entities identical");
        Assert.True(errors.MaxRelative <= tolerances.PaddedBatchVsSingleRelative, $"batch vs single relative {errors.MaxRelative:E2}");
        return errors;
    }

    /// <summary>
    /// The in-contract cases of each label count in random orders (mixed neighbours), run in one call per order by
    /// <paramref name="extractor"/> (typically with <see cref="Gliner2ExtractorOptions.SortByLength"/> or
    /// <see cref="Gliner2ExtractorOptions.MaxBatchTokens"/>): every output comes back at its input position, span logits
    /// within the protected padded-batch-vs-single bound of the same case run alone, entities identical.
    /// </summary>
    public static Gliner2Errors ShuffledBatches(Gliner2EntityExtractor extractor, IReadOnlyList<Gliner2Case> cases, int[] seeds, Gliner2Tolerances tolerances, ITestOutputHelper output)
    {
        var ct = TestContext.Current.CancellationToken;
        var errors = new Gliner2Errors();
        int rows = 0;
        foreach (var group in cases.Where(c => c.OutOfContract is null).GroupBy(c => c.Labels.Count))
        {
            var list = group.ToList();
            var inputs = list.Select(c => extractor.Prepare(c.Text, c.LabelsRequested, c.Descriptions)).ToList();
            var singles = inputs.Select(i => extractor.RunModel([i], ct)[0]).ToList();
            foreach (int seed in seeds)
            {
                var order = Enumerable.Range(0, list.Count).ToArray();
                new Random(seed).Shuffle(order);
                var batched = extractor.RunModel(order.Select(i => inputs[i]).ToList(), ct);
                for (int k = 0; k < order.Length; k++)
                {
                    int i = order[k];
                    errors.Add(batched[k].GetValidSpanLogits(), singles[i].GetValidSpanLogits(), list[i].Id);
                    var a = Gliner2ParityChecks.UpstreamEntities(inputs[i], batched[k], list[i].Threshold).Select(e => e.Key);
                    var b = Gliner2ParityChecks.UpstreamEntities(inputs[i], singles[i], list[i].Threshold).Select(e => e.Key);
                    Assert.Equal(b, a);
                    rows++;
                }
            }
        }

        var o = extractor.Options;
        output.WriteLine($"{rows} rows in random orders (MaxBatchSize {o.MaxBatchSize}, MaxBatchTokens {o.MaxBatchTokens?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "none"}, SortByLength {o.SortByLength}) vs single runs: span logits {errors}; entities identical");
        Assert.False(errors.NonFinite);
        Assert.True(errors.MaxRelative <= tolerances.PaddedBatchVsSingleRelative, $"shuffled batches vs single relative {errors.MaxRelative:E2}");
        return errors;
    }

    /// <summary>Texts above 512 tokens are rejected with the typed exception, before any inference.</summary>
    public static void LongInputsAreRejected(Gliner2EntityExtractor extractor, IReadOnlyList<Gliner2Case> cases, ITestOutputHelper output)
    {
        var longCases = cases.Where(c => c.MaxLen is null && c.InputIds.Length > 512).ToList();
        Assert.Equal(2, longCases.Count);
        foreach (var c in longCases)
        {
            var ex = Assert.Throws<Gliner2InputTooLongException>(() => extractor.Extract(c.Text, c.LabelsRequested, c.Threshold, TestContext.Current.CancellationToken));
            Assert.Equal(c.InputIds.Length, ex.TokenCount);
            Assert.Equal(512, ex.MaxTokens);
            output.WriteLine($"{c.Id}: rejected, {ex.TokenCount} tokens > {ex.MaxTokens}");
        }
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
