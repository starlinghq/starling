using System.Text.Json.Nodes;

namespace Starlings.Benchmarks;

/// <summary>
/// The quality decision of one run: the preliminary sweep over the protected cases and every timed series' own check
/// (each workload's outputs against the reference) must pass. A failure in any single series fails the run, so its
/// timings are never presented as a valid result.
/// </summary>
internal static class QualityGate
{
    /// <summary>
    /// Embedding harnesses (e5): a series passes when its <c>quality.max_abs</c> is within <paramref name="tolerance"/>
    /// and its outputs are finite. See <see cref="Apply(JsonObject, IReadOnlyList{JsonObject}, Func{JsonObject, bool})"/>.
    /// </summary>
    public static bool Apply(JsonObject runQuality, IReadOnlyList<JsonObject> series, double tolerance) =>
        Apply(runQuality, series, q => q["max_abs"]!.GetValue<double>() <= tolerance && q["finite"]!.GetValue<bool>());

    /// <summary>
    /// Harnesses whose series carry their own verdict: a series passes when its <c>quality.pass</c> is true (a missing
    /// verdict fails).
    /// </summary>
    public static bool ApplyRecorded(JsonObject runQuality, IReadOnlyList<JsonObject> series) =>
        Apply(runQuality, series, q => q["pass"]?.GetValue<bool>() ?? false);

    /// <summary>
    /// Marks each series' <c>quality.pass</c> with <paramref name="seriesPasses"/>, records <c>series_pass</c> and
    /// <c>failed_series</c> in <paramref name="runQuality"/>, and sets <c>runQuality["pass"]</c> to the conjunction of
    /// the preliminary sweep and every series. Returns the final decision.
    /// </summary>
    public static bool Apply(JsonObject runQuality, IReadOnlyList<JsonObject> series, Func<JsonObject, bool> seriesPasses)
    {
        bool preliminary = runQuality["pass"]?.GetValue<bool>() ?? false;
        var failed = new JsonArray();
        foreach (var s in series)
        {
            var q = s["quality"]!.AsObject();
            bool pass = seriesPasses(q);
            q["pass"] = pass;
            if (!pass)
            {
                failed.Add((JsonNode)$"{s["mode"]}/{s["path"]}/{s["workload"]}/c{s["concurrency"]}");
            }
        }

        bool seriesPass = failed.Count == 0;
        runQuality["preliminary_pass"] = preliminary;
        runQuality["series_pass"] = seriesPass;
        runQuality["failed_series"] = failed;
        runQuality["pass"] = preliminary && seriesPass;
        return preliminary && seriesPass;
    }
}
