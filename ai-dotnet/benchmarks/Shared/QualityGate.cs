using System.Text.Json.Nodes;

namespace Starlings.Benchmarks;

/// <summary>
/// The quality decision of one run: the preliminary sweep over the 46 protected cases and every timed series' own
/// check (each workload's outputs against the protected embeddings) must be within the profile tolerance and finite.
/// A failure in any single series fails the run, so its timings are never presented as a valid result.
/// </summary>
internal static class QualityGate
{
    /// <summary>
    /// Marks each series' <c>quality.pass</c>, records <c>series_pass</c> and <c>failed_series</c> in
    /// <paramref name="runQuality"/>, and sets <c>runQuality["pass"]</c> to the conjunction of the preliminary sweep
    /// and every series. Returns the final decision.
    /// </summary>
    public static bool Apply(JsonObject runQuality, IReadOnlyList<JsonObject> series, double tolerance)
    {
        bool preliminary = runQuality["pass"]?.GetValue<bool>() ?? false;
        var failed = new JsonArray();
        foreach (var s in series)
        {
            var q = s["quality"]!.AsObject();
            bool pass = q["max_abs"]!.GetValue<double>() <= tolerance && q["finite"]!.GetValue<bool>();
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
