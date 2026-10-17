using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Starlings.Models.Kev;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// The experimental batched-row paths (internal; capability <c>batched_rows</c> stays Unsupported) against single rows:
/// uncached batched rows within <c>padded_batch_vs_single_probability_max_abs</c> 1e-5 of <see cref="KevDecisionModel.RunUncached"/>
/// and within <c>cached_vs_uncached_probability_max_abs</c> 1e-5 of the default cached path
/// (protected/kev-0.8b/onnx-fp32/tolerances.yaml). Cached batched rows are refused by ONNX Runtime 1.30.0's
/// GroupQueryAttention; a test pins that, so a runtime that lifts it shows up as a failure here.
/// Model tests skip without a bundle (see <see cref="KevTestAssets"/>).
/// </summary>
public sealed class BatchedRowsTests
{
    private const double PaddedBatchBound = 1.0e-5;
    private const double CachedBound = 1.0e-5;

    /// <summary>Set to a file path to run <see cref="Latency"/> and write its JSON there.</summary>
    public const string BenchmarkEnvironmentVariable = "STARLING_KEV_BATCH_BENCH";

    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static void Log(string line) => TestContext.Current.TestOutputHelper?.WriteLine(line);

    // ---- planner (no model) ----

    [Fact]
    public void PlannerKeepsQuestionOrderByDefault()
    {
        var groups = KevBatchPlanner.Plan([5, 1, 9, 3, 7], 10, new KevBatchOptions { MaxRows = 2, MaxTokens = null });
        Assert.Equal([[0, 1], [2, 3], [4]], groups);
    }

    [Fact]
    public void PlannerCountsTheTiledPastInTheTokenBudget()
    {
        // (rows) * (past 100 + width 10): two rows cost 220, three 330.
        var groups = KevBatchPlanner.Plan([10, 10, 10, 10], 100, new KevBatchOptions { MaxRows = 8, MaxTokens = 250 });
        Assert.Equal([[0, 1], [2, 3]], groups);
    }

    [Fact]
    public void PlannerUsesTheWidestRowOfAGroup()
    {
        // 2 * (0 + 50) = 100 fits; adding the 60-wide row makes 3 * 60 = 180 > 150.
        var groups = KevBatchPlanner.Plan([50, 10, 60, 5], 0, new KevBatchOptions { MaxRows = 8, MaxTokens = 150 });
        Assert.Equal([[0, 1], [2, 3]], groups);
    }

    [Fact]
    public void PlannerRunsARowOverTheBudgetAlone()
    {
        var groups = KevBatchPlanner.Plan([5, 500, 5], 10, new KevBatchOptions { MaxRows = 8, MaxTokens = 100 });
        Assert.Equal([[0], [1], [2]], groups);
    }

    [Fact]
    public void PlannerSortsLongestFirstAndCoversEveryRowOnce()
    {
        int[] widths = [3, 40, 7, 40, 1, 12, 9];
        var groups = KevBatchPlanner.Plan(widths, 0, new KevBatchOptions { MaxRows = 3, MaxTokens = null, SortByLength = true });
        Assert.Equal([[1, 3, 5], [6, 2, 0], [4]], groups);
        Assert.Equal(Enumerable.Range(0, widths.Length), groups.SelectMany(g => g).Order());
    }

    [Theory]
    [InlineData(0, null)]
    [InlineData(1, 0)]
    public void PlannerRejectsAnEmptyBudget(int maxRows, int? maxTokens) =>
        Assert.ThrowsAny<ArgumentOutOfRangeException>(() =>
            KevBatchPlanner.Plan([1], 0, new KevBatchOptions { MaxRows = maxRows, MaxTokens = maxTokens }));

    // ---- model ----

    /// <summary>The request goldens and the onboarding corpus as records.</summary>
    private static IEnumerable<(string Id, KevRecord Record)> Records()
    {
        foreach (var c in KevTestAssets.ReadData("requests.json").EnumerateArray())
        {
            yield return (c.GetProperty("id").GetString()!, KevRequest.ParseSystemOne(c.GetProperty("request").GetString()!).ToRecord());
        }

        string corpus = Path.Combine(KevTestAssets.RepositoryRoot, "model-pipeline", "reference", "corpus", "kev-decisions-v1.jsonl");
        foreach (string line in File.ReadLines(corpus).Where(l => l.Length > 0))
        {
            using var doc = JsonDocument.Parse(line);
            var r = doc.RootElement;
            yield return ("corpus:" + r.GetProperty("id").GetString(), new KevRecord(
                r.GetProperty("state").GetString()!,
                [.. r.GetProperty("questions").EnumerateArray().Select(q => new KevRecordQuestion(
                    q.GetProperty("id").GetString()!, q.GetProperty("instr").GetString()!,
                    [.. q.GetProperty("options").EnumerateArray().Select(o => o.GetString()!)]))]));
        }
    }

    /// <summary>One short question next to a long one with many long options: the widest padding a request makes.</summary>
    private static KevRecord MixedNeighbours()
    {
        string longOption(int i) => string.Concat(Enumerable.Repeat($"option {i} covers refunds, exchanges and store credit for damaged goods; ", 6));
        return new KevRecord(
            "Order 4471 arrived on 3 March with a cracked screen. The customer asked for a refund on 5 March.",
            [
                new("short", "Refund?", ["no", "yes"]),
                new("long", "Which of these policies applies to the order described above, given the dates and the damage?",
                    [.. Enumerable.Range(0, 8).Select(longOption)]),
                new("medium", "Is the request within the 30-day window?", ["no", "yes"]),
            ]);
    }

    private static double MaxAbs(IReadOnlyList<float[]> a, IReadOnlyList<float[]> b)
    {
        Assert.Equal(a.Count, b.Count);
        double max = 0;
        for (int q = 0; q < a.Count; q++)
        {
            Assert.Equal(a[q].Length, b[q].Length);
            for (int i = 0; i < a[q].Length; i++)
            {
                max = Math.Max(max, Math.Abs(a[q][i] - b[q][i]));
            }
        }

        return max;
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void BatchedRowsMatchSingleRows(KevModel model)
    {
        var (kev, reason) = ModelParityTests.Load(model);
        Assert.SkipWhen(kev is null, reason);
        var budgets = new (string Name, KevBatchOptions Options)[]
        {
            ("all rows, question order", new KevBatchOptions { MaxRows = 64, MaxTokens = null }),
            ("4 rows, sorted", new KevBatchOptions { MaxRows = 4, MaxTokens = null, SortByLength = true }),
            ("token cap 1024, sorted", new KevBatchOptions { MaxTokens = 1024, SortByLength = true }),
        };
        double uncachedVsSingle = 0, cachedVsSequential = 0;
        int records = 0, questions = 0, multiRow = 0;
        foreach (var (id, record) in Records().Append(("mixed-neighbours", MixedNeighbours())))
        {
            var encoding = kev!.Encode(record);
            var sequential = kev.Run(encoding, Ct);
            var uncached = kev.RunUncached(encoding, Ct);
            foreach (var (name, options) in budgets)
            {
                var batched = kev.RunUncachedBatched(encoding, options, Ct);
                double u = MaxAbs(batched, uncached), c = MaxAbs(batched, sequential);
                Assert.True(u <= PaddedBatchBound, $"{id} ({name}): uncached batched vs RunUncached {u:G3} > {PaddedBatchBound:G3}");
                Assert.True(c <= CachedBound, $"{id} ({name}): uncached batched vs Run (cached) {c:G3} > {CachedBound:G3}");
                (uncachedVsSingle, cachedVsSequential) = (Math.Max(uncachedVsSingle, u), Math.Max(cachedVsSequential, c));
            }

            records++;
            questions += encoding.Rows.Count;
            multiRow += encoding.Rows.Count > 1 ? 1 : 0;
        }

        Log(string.Create(CultureInfo.InvariantCulture,
            $"{model}: {records} records ({multiRow} with several rows), {questions} questions; max abs probability of uncached batched rows: vs RunUncached {uncachedVsSingle:G3}, vs Run {cachedVsSequential:G3}"));
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void OneRowPerRunIsTheSequentialPathBitForBit(KevModel model)
    {
        var (kev, reason) = ModelParityTests.Load(model);
        Assert.SkipWhen(kev is null, reason);
        var encoding = kev!.Encode(MixedNeighbours());
        var single = new KevBatchOptions { MaxRows = 1 };
        Assert.Equal(kev.Run(encoding, Ct).SelectMany(p => p), kev.RunBatched(encoding, single, Ct).SelectMany(p => p));
        Assert.Equal(kev.RunUncached(encoding, Ct).SelectMany(p => p), kev.RunUncachedBatched(encoding, single, Ct).SelectMany(p => p));
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void ARowsResultDoesNotDependOnItsNeighboursBeyondTheBound(KevModel model)
    {
        var (kev, reason) = ModelParityTests.Load(model);
        Assert.SkipWhen(kev is null, reason);
        var record = MixedNeighbours();
        var all = new KevBatchOptions { MaxRows = 8, MaxTokens = null };
        var withLong = kev!.RunUncachedBatched(kev.Encode(record), all, Ct);
        var shortOnly = kev.RunUncachedBatched(kev.Encode(record with { Questions = [record.Questions[0], record.Questions[2]] }), all, Ct);
        var alone = kev.RunUncached(kev.Encode(record with { Questions = [record.Questions[0]] }), Ct);
        double neighbours = MaxAbs([withLong[0], withLong[2]], shortOnly);
        double vsAlone = MaxAbs([withLong[0]], alone);
        Log(string.Create(CultureInfo.InvariantCulture, $"{model}: short row next to the long row vs without it {neighbours:G3}, vs alone {vsAlone:G3}"));
        Assert.True(neighbours <= PaddedBatchBound, $"neighbours changed a row by {neighbours:G3}");
        Assert.True(vsAlone <= PaddedBatchBound, $"batched short row vs alone {vsAlone:G3}");
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void CancelledBatchedCallThrowsAndModelStaysUsable(KevModel model)
    {
        var (kev, reason) = ModelParityTests.Load(model);
        Assert.SkipWhen(kev is null, reason);
        var encoding = kev!.Encode(MixedNeighbours());
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => kev.RunBatched(encoding, null, cts.Token));
        Assert.ThrowsAny<OperationCanceledException>(() => kev.RunUncachedBatched(encoding, null, cts.Token));
        Assert.Equal(3, kev.RunUncachedBatched(encoding, null, Ct).Count);
    }

    /// <summary>
    /// Pins the runtime limit that blocks the cached batched layout. If this fails because the graph ran, the runtime
    /// accepts batched continuation from a past: add the cached batched path to <see cref="BatchedRowsMatchSingleRows"/>.
    /// </summary>
    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void CachedBatchedRowsAreRefusedByGroupQueryAttention(KevModel model)
    {
        var (kev, reason) = ModelParityTests.Load(model);
        Assert.SkipWhen(kev is null, reason);
        var encoding = kev!.Encode(MixedNeighbours());
        var e = Assert.Throws<NotSupportedException>(() => kev.RunBatched(encoding, new KevBatchOptions { MaxRows = 2 }, Ct));
        Assert.Contains(KevDecisionModel.GroupQueryAttentionBatchLimit, e.InnerException!.Message, StringComparison.Ordinal);
        Assert.Equal(3, kev.Run(encoding, Ct).Count); // the model stays usable
    }

    /// <summary>
    /// The default cached sequential path against uncached rows, one by one and batched, on CPU (opt-in: set <see cref="BenchmarkEnvironmentVariable"/>
    /// to an output path). States of 32, 128 and 512 tokens, 8 questions of mixed widths (noul, choice, score, long
    /// options); the session is warmed by <see cref="KevDecisionModel.Run"/> on each record first, then each variant runs twice.
    /// </summary>
    [Theory]
    [InlineData(KevModel.Kev08B)]
    public void Latency(KevModel model)
    {
        string? output = Environment.GetEnvironmentVariable(BenchmarkEnvironmentVariable);
        Assert.SkipWhen(string.IsNullOrEmpty(output), $"{BenchmarkEnvironmentVariable} is not set");
        var (kev, reason) = ModelParityTests.Load(model);
        Assert.SkipWhen(kev is null, reason);
        const int reps = 2;
        var results = new List<Dictionary<string, object>>();
        foreach (int stateTokens in new[] { 32, 128, 512 })
        {
            var record = BenchRecord(kev!, stateTokens, 8);
            var encoding = kev!.Encode(record);
            var widths = encoding.Rows.Select(r => r.Ids.Length).ToArray();
            var reference = kev.Run(encoding, Ct);
            var variants = new (string Name, Func<IReadOnlyList<float[]>> Run)[]
            {
                ("cached sequential (Run, default)", () => kev.Run(encoding, Ct)),
                ("uncached sequential (RunUncached)", () => kev.RunUncached(encoding, Ct)),
                ("uncached batched b4 sorted", () => kev.RunUncachedBatched(encoding, new KevBatchOptions { MaxRows = 4, MaxTokens = null, SortByLength = true }, Ct)),
                ("uncached batched b8 sorted", () => kev.RunUncachedBatched(encoding, new KevBatchOptions { MaxRows = 8, MaxTokens = null, SortByLength = true }, Ct)),
            };
            foreach (var (name, run) in variants)
            {
                double maxAbs = 0;
                var ms = new List<double>();
                for (int i = 0; i < reps; i++)
                {
                    var sw = Stopwatch.StartNew();
                    var probs = run();
                    ms.Add(sw.Elapsed.TotalMilliseconds);
                    maxAbs = Math.Max(maxAbs, MaxAbs(probs, reference));
                }

                ms.Sort();
                results.Add(new()
                {
                    ["state_tokens"] = encoding.StateIds.Length, ["questions"] = widths.Length,
                    ["branch_tokens_min"] = widths.Min(), ["branch_tokens_max"] = widths.Max(), ["branch_tokens_sum"] = widths.Sum(),
                    ["variant"] = name, ["min_ms"] = Math.Round(ms[0], 1), ["max_ms"] = Math.Round(ms[^1], 1),
                    ["max_abs_vs_sequential"] = maxAbs,
                });
                Log(string.Create(CultureInfo.InvariantCulture, $"state {encoding.StateIds.Length} {name}: min {ms[0]:F0} ms, max abs {maxAbs:G3}"));
            }
        }

        File.WriteAllText(output!, JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["model"] = model.ToString(), ["intra_op_threads"] = kev!.Options.Session.IntraOpNumThreads?.ToString(CultureInfo.InvariantCulture) ?? "default", ["processors"] = Environment.ProcessorCount,
            ["reps"] = reps, ["results"] = results,
        }, Indented));
    }

    private static KevRecord BenchRecord(KevDecisionModel kev, int stateTokens, int questions)
    {
        const string sentence = "The customer reported that order 4471 arrived late with a cracked screen and asked for a refund. ";
        var state = new System.Text.StringBuilder();
        while (kev.Encode(new KevRecord(state.ToString(), [new("q", "x", ["no", "yes"])])).StateIds.Length < stateTokens)
        {
            state.Append(sentence);
        }

        var list = new List<KevRecordQuestion>();
        for (int i = 0; i < questions; i++)
        {
            list.Add((i % 4) switch
            {
                0 => new($"n{i}", "Is the customer eligible for a refund?", ["no", "yes"]),
                1 => new($"c{i}", "Which team should handle this ticket?", ["billing", "shipping", "returns", "support"]),
                2 => new($"s{i}", "How severe is the problem?", ["minor", "moderate", "serious", "critical", "blocking"]),
                _ => new($"l{i}", "Which policy applies to this order, given the dates and the damage described?",
                    [.. Enumerable.Range(0, 6).Select(k => $"policy {k}: refunds, exchanges or store credit for goods damaged in transit, within {k * 15 + 15} days")]),
            });
        }

        return new KevRecord(state.ToString(), list);
    }
}
