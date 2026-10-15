using System.Text.Json;
using System.Text.Json.Nodes;

/// <summary>Recomputes the verdicts and the markdown report of existing raw results (<c>reanalyze</c> command).</summary>
internal static class Reanalysis
{
    public static int Run(string[] paths)
    {
        if (paths.Length == 0)
        {
            Console.Error.WriteLine("usage: Starlings.Soak.Tool reanalyze <raw.json>...");
            return 2;
        }

        int worst = 0;
        foreach (string path in paths)
        {
            var raw = JsonNode.Parse(File.ReadAllText(path))!.AsObject();
            var o = raw["options"]!;
            var cycles = raw["cycles"]!.AsArray().Select(c => new CycleRecord(
                (int)c!["index"]!, (double)c["start_s"]!, (double)c["end_s"]!, (double)c["load_ms"]!, (double)c["unload_ms"]!,
                (int)c["requests"]!, (int)c["deadline_requests"]!, double.NaN, double.NaN, double.NaN,
                Sample(c["loaded"]!), Sample(c["unloaded"]!), c["trimmed"] is { } t ? Sample(t) : null)).ToList();
            var verdicts = SoakRunner.Analyze(cycles, (double)o["elapsed_s"]!, (double)o["warmup_fraction"]!, (double)o["growth_limit_floor_mb"]!);
            string verdict = SoakRunner.OverallVerdict(cycles.Count, verdicts, (int)raw["wrong_output_count"]!, (int)raw["error_count"]!);
            if (((string?)raw["verdict"])?.StartsWith("INCOMPLETE", StringComparison.Ordinal) == true)
            {
                verdict = "INCOMPLETE (the run did not finish; reanalysed checkpoint): " + verdict;
            }

            raw["verdict"] = verdict;
            raw["analysis"] = new JsonArray(verdicts.Select(v => (JsonNode)v.ToJson()).ToArray());
            File.WriteAllText(path, raw.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
            File.WriteAllText(Path.ChangeExtension(path, ".md"), Report.Markdown(raw));
            Console.WriteLine($"{path}: {verdict}");
            worst = Math.Max(worst, verdict.StartsWith("FAIL", StringComparison.Ordinal) ? 1 : 0);
        }

        return worst;
    }

    private static MemorySample Sample(JsonNode s) => new(
        (double)s["t_s"]!,
        (string)s["phase"]!,
        (int)s["cycle"]!,
        (bool)s["settled"]!,
        (double)s["managed_heap_mb"]!,
        (double)s["working_set_mb"]!,
        (double)s["private_mb"]!,
        (double?)s["rss_anon_mb"],
        (double?)s["rss_file_mb"],
        (double?)s["vm_data_mb"],
        (double?)s["gpu_mb"],
        (double?)s["anon_huge_pages_mb"]);
}
