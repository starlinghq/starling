using Starlings.Models.Kev;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// Checks shared by the CPU and CUDA test projects: the request goldens against Kev's PyTorch fp32 probabilities
/// (Data/probs-*.json, make_dotnet_cases.py probs) and concurrent calls against sequential ones.
/// </summary>
internal static class KevParity
{
    /// <summary>Probability bound of the profiles (protected/kev-0.8b/onnx-fp32/tolerances.yaml; proposed for 4B).</summary>
    public const double ProbabilityBound = 2.0e-4;

    /// <summary>Cached against uncached rows (same file).</summary>
    public const double CachedBound = 1.0e-5;

    /// <summary>Reference margin below which an argmax swap is a near tie, not a flip.</summary>
    public const double TieMargin = 1.0e-4;

    public static IEnumerable<string> CaseIds() =>
        KevTestAssets.ReadData("requests.json").EnumerateArray().Select(c => c.GetProperty("id").GetString()!);

    /// <summary>The reference file of a model, or null with the reason.</summary>
    public static string? MissingReference(KevModel model)
    {
        string file = ReferenceFile(model);
        return File.Exists(Path.Combine(KevTestAssets.DataDirectory, file)) ? null : $"{file} not generated";
    }

    public static KevRequest Request(string id) => KevRequest.ParseSystemOne(
        KevTestAssets.ReadData("requests.json").EnumerateArray().Single(c => c.GetProperty("id").GetString() == id)
            .GetProperty("request").GetString()!);

    /// <summary>
    /// Decides golden <paramref name="id"/> and asserts every question within <see cref="ProbabilityBound"/> and with the
    /// reference's argmax (outside near ties). Returns the largest absolute difference.
    /// </summary>
    public static double AssertMatchesReference(KevDecisionModel kev, string id, CancellationToken cancellationToken)
    {
        var expected = KevTestAssets.ReadData(ReferenceFile(kev.Profile.Model)).GetProperty("cases").EnumerateArray()
            .Single(c => c.GetProperty("id").GetString() == id).GetProperty("probs").EnumerateArray()
            .Select(q => q.EnumerateArray().Select(p => p.GetDouble()).ToArray()).ToArray();

        var decision = kev.Decide(Request(id), cancellationToken);
        Assert.Equal(expected.Length, decision.Answers.Count);
        double worst = 0;
        for (int q = 0; q < expected.Length; q++)
        {
            var actual = decision.Answers[q].ProbabilityList;
            Assert.Equal(expected[q].Length, actual.Count);
            double max = expected[q].Zip(actual, (a, b) => Math.Abs(a - b)).Max();
            worst = Math.Max(worst, max);
            Assert.True(max <= ProbabilityBound, $"{id} question {q}: max abs {max:G3} > {ProbabilityBound:G3}");
            int refArg = Array.IndexOf(expected[q], expected[q].Max());
            var sorted = expected[q].OrderDescending().ToArray();
            if (sorted.Length < 2 || sorted[0] - sorted[1] >= TieMargin)
            {
                Assert.Equal(refArg, actual.ToList().IndexOf(actual.Max()));
            }
        }

        return worst;
    }

    /// <summary>Cached rows against full uncached rows on the many-questions golden; returns the largest difference.</summary>
    public static double AssertCachedEqualsUncached(KevDecisionModel kev, CancellationToken cancellationToken)
    {
        var encoding = kev.Encode(Request("many-questions").ToRecord());
        var cached = kev.Run(encoding, cancellationToken);
        var uncached = kev.RunUncached(encoding, cancellationToken);
        double max = cached.Zip(uncached, (a, b) => a.Zip(b, (x, y) => Math.Abs(x - y)).Max()).Max();
        Assert.True(max <= CachedBound, $"cached vs uncached {max:G3} > {CachedBound:G3}");
        // The cache is read, never written: running again from the same encoding gives the same bits.
        Assert.Equal(cached.SelectMany(p => p), kev.Run(encoding, cancellationToken).SelectMany(p => p));
        return max;
    }

    /// <summary>
    /// Capability <c>concurrent_calls</c>: <paramref name="callers"/> threads decide every golden
    /// <paramref name="rounds"/> times on one model, and every answer must equal the sequential answer bit for bit.
    /// Returns the number of concurrent decisions checked.
    /// </summary>
    public static int AssertConcurrentEqualsSequential(KevDecisionModel kev, int callers, int rounds, CancellationToken cancellationToken)
    {
        var ids = CaseIds().ToArray();
        var requests = ids.ToDictionary(id => id, Request, StringComparer.Ordinal);
        var sequential = ids.ToDictionary(id => id, id => Flatten(kev.Decide(requests[id], cancellationToken)), StringComparer.Ordinal);
        var mismatches = new System.Collections.Concurrent.ConcurrentBag<string>();
        int checkedCount = 0;
        using var start = new Barrier(callers);
        var threads = Enumerable.Range(0, callers).Select(c => new Thread(() =>
        {
            start.SignalAndWait(cancellationToken);
            for (int r = 0; r < rounds; r++)
            {
                // Each caller walks the goldens from a different offset so different requests overlap.
                for (int i = 0; i < ids.Length; i++)
                {
                    string id = ids[(i + c) % ids.Length];
                    if (!Flatten(kev.Decide(requests[id], cancellationToken)).SequenceEqual(sequential[id]))
                    {
                        mismatches.Add($"caller {c} round {r} {id}");
                    }

                    Interlocked.Increment(ref checkedCount);
                }
            }
        })).ToList();
        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());
        Assert.True(mismatches.IsEmpty, $"{mismatches.Count} concurrent answers differ from sequential: {string.Join("; ", mismatches.Take(5))}");
        return checkedCount;
    }

    private static string ReferenceFile(KevModel model) => $"probs-{KevModelProfile.For(model).Id.Split('/')[0]}.json";

    private static float[] Flatten(KevDecision decision) => [.. decision.Answers.SelectMany(a => a.ProbabilityList)];
}
