using System.Globalization;
using System.Text;
using Starlings.Models.Gliner2;

namespace Starlings.TestSupport;

/// <summary>Max-abs and relative error accumulators.</summary>
internal sealed class Gliner2Errors
{
    public double MaxAbs { get; private set; }

    public double MaxRelative { get; private set; }

    public bool NonFinite { get; private set; }

    public string Worst { get; private set; } = string.Empty;

    public void Add(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, string id)
    {
        if (actual.Length != expected.Length)
        {
            throw new InvalidOperationException($"{id}: {actual.Length} values, expected {expected.Length}.");
        }

        for (int i = 0; i < actual.Length; i++)
        {
            if (!float.IsFinite(actual[i]))
            {
                NonFinite = true;
                continue;
            }

            double d = Math.Abs((double)actual[i] - expected[i]);
            double rel = d / Math.Max(1.0, Math.Abs((double)expected[i]));
            if (d > MaxAbs)
            {
                MaxAbs = d;
            }

            if (rel > MaxRelative)
            {
                MaxRelative = rel;
                Worst = id;
            }
        }
    }

    public void Add(double d, string id)
    {
        if (d > MaxAbs)
        {
            MaxAbs = d;
            Worst = id;
        }
    }

    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"max abs {MaxAbs:E2}, max rel {MaxRelative:E2}{(Worst.Length > 0 ? $" (worst {Worst})" : string.Empty)}");
}

/// <summary>Result of comparing one case's entities with the reference under the order and flip rules of the protected tolerances.</summary>
internal sealed record Gliner2EntityComparison(bool Exact, bool WithinRules, int AllowedSwaps, int AllowedFlips, double ConfidenceMaxAbs, string Detail);

/// <summary>
/// Parity checks of the C# GLiNER2 path against the protected development fixtures (accepted 2026-10-10, PR #19), shared by the CPU and CUDA test projects.
/// Bounds: the profile's protected tolerances.yaml (<see cref="Gliner2Tolerances"/>).
/// </summary>
internal static class Gliner2ParityChecks
{
    /// <summary>Upstream-format entities (normalized-text offsets) of the C# decoder for one case.</summary>
    public static List<Gliner2RefEntity> UpstreamEntities(Gliner2ModelInput input, Gliner2ModelOutput output, float threshold)
    {
        var list = new List<Gliner2RefEntity>();
        var spans = Gliner2Decoder.DecodeUpstream(input, output, threshold);
        if (spans is null)
        {
            return list;
        }

        for (int label = 0; label < input.Labels.Count; label++)
        {
            list.AddRange(spans[label].Select(s => new Gliner2RefEntity(input.Labels[label], s.Text, s.Start, s.End, s.Confidence)));
        }

        return list;
    }

    /// <summary>
    /// Entities equal, including order, except: a span may flip only when its reference score is within
    /// <see cref="Gliner2Tolerances.ThresholdFlipWindow"/> of the threshold, and two items may swap only when their
    /// reference confidences differ by at most <see cref="Gliner2Tolerances.OrderTieWindow"/> (maintainer decision 2,
    /// 2026-10-10).
    /// </summary>
    public static Gliner2EntityComparison CompareEntities(Gliner2Case c, IReadOnlyList<Gliner2RefEntity> actual, Gliner2Tolerances tolerances)
    {
        var expected = c.Entities;
        bool exact = expected.Count == actual.Count && expected.Zip(actual).All(p => p.First.Key == p.Second.Key);
        var expectedKeys = expected.Select(e => e.Key).ToHashSet(StringComparer.Ordinal);
        var actualKeys = actual.Select(e => e.Key).ToHashSet(StringComparer.Ordinal);
        int flips = 0;
        var problems = new List<string>();
        foreach (var e in expected.Where(e => !actualKeys.Contains(e.Key)))
        {
            if (Math.Abs(e.Confidence - c.Threshold) <= tolerances.ThresholdFlipWindow)
            {
                flips++;
            }
            else
            {
                problems.Add($"missing {e.Label}:'{e.Text}'@{e.Start}-{e.End} (ref {e.Confidence:R})");
            }
        }

        foreach (var a in actual.Where(a => !expectedKeys.Contains(a.Key)))
        {
            double reference = ReferenceScoreOf(c, a);
            if (Math.Abs(reference - c.Threshold) <= tolerances.ThresholdFlipWindow)
            {
                flips++;
            }
            else
            {
                problems.Add($"extra {a.Label}:'{a.Text}'@{a.Start}-{a.End} (C# {a.Confidence:R}, ref {reference:R})");
            }
        }

        // Relative order of the common items, per label.
        int swaps = 0;
        double confidence = 0;
        foreach (var label in c.Labels)
        {
            var exp = expected.Where(e => e.Label == label && actualKeys.Contains(e.Key)).ToList();
            var act = actual.Where(a => a.Label == label && expectedKeys.Contains(a.Key)).ToList();
            var position = act.Select((a, i) => (a.Key, i)).ToDictionary(p => p.Key, p => p.i, StringComparer.Ordinal);
            for (int i = 0; i < exp.Count; i++)
            {
                confidence = Math.Max(confidence, Math.Abs(exp[i].Confidence - act[position[exp[i].Key]].Confidence));
                for (int j = i + 1; j < exp.Count; j++)
                {
                    if (position[exp[i].Key] > position[exp[j].Key])
                    {
                        if (Math.Abs(exp[i].Confidence - exp[j].Confidence) <= tolerances.OrderTieWindow)
                        {
                            swaps++;
                        }
                        else
                        {
                            problems.Add($"order {label}: '{exp[i].Text}' ({exp[i].Confidence:R}) after '{exp[j].Text}' ({exp[j].Confidence:R})");
                        }
                    }
                }
            }
        }

        return new Gliner2EntityComparison(exact, problems.Count == 0, swaps, flips, confidence, string.Join("; ", problems));
    }

    /// <summary>The reference entity list mapped onto the original text with the documented rule, written independently of the library.</summary>
    public static List<(string Label, string Text, int ScalarStart, int ScalarEnd)> MapReferenceToOriginal(Gliner2Case c)
    {
        var scalars = c.Text.EnumerateRunes().Select(r => r.Value).ToArray();
        var result = new List<(string, string, int, int)>();
        foreach (var e in c.Entities)
        {
            int start = e.Start;
            int end = Math.Min(e.End, scalars.Length);
            while (end > start && IsPythonSpace(scalars[end - 1]))
            {
                end--;
            }

            if (end <= start)
            {
                continue;
            }

            string text = string.Concat(scalars[start..end].Select(v => new Rune(v).ToString()));
            result.Add((e.Label, text, start, end));
        }

        return result;
    }

    private static bool IsPythonSpace(int c) => c is (>= 0x9 and <= 0xD) or (>= 0x1C and <= 0x20) or 0x85 or 0xA0 or 0x1680
        or (>= 0x2000 and <= 0x200A) or 0x2028 or 0x2029 or 0x202F or 0x205F or 0x3000;

    private static double ReferenceScoreOf(Gliner2Case c, Gliner2RefEntity a)
    {
        int labelIndex = c.Labels.ToList().IndexOf(a.Label);
        int startWord = Array.IndexOf(c.StartMap, a.Start);
        int endWord = Array.IndexOf(c.EndMap, a.End);
        return labelIndex < 0 || startWord < 0 || endWord < startWord || endWord - startWord >= 8
            ? double.NaN
            : c.ReferenceScore(labelIndex, startWord, endWord - startWord);
    }
}
