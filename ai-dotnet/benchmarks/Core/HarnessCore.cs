using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Starlings.Benchmarks;

/// <summary>Monotonic clock in nanoseconds (CLOCK_MONOTONIC on Linux, the clock Python's <c>time.monotonic_ns</c> reads).</summary>
internal static class MonotonicClock
{
    private static readonly double s_nsPerTick = 1e9 / Stopwatch.Frequency;

    public static long NowNs() => (long)(Stopwatch.GetTimestamp() * s_nsPerTick);

    public static double ElapsedMs(long startTimestamp) => Stopwatch.GetElapsedTime(startTimestamp).TotalMilliseconds;
}

/// <summary>Command-line options of the form <c>--name value</c>.</summary>
internal sealed class ArgMap
{
    private readonly Dictionary<string, string> _values;

    private ArgMap(Dictionary<string, string> values) => _values = values;

    public IEnumerable<KeyValuePair<string, string>> All => _values;

    public static ArgMap Parse(IReadOnlyList<string> args, string usage)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Count; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Count)
            {
                throw new ArgumentException($"Expected '--name value', got '{args[i]}'.\n{usage}");
            }

            map[args[i][2..]] = args[++i];
        }

        return new ArgMap(map);
    }

    public string? Get(string key) => _values.GetValueOrDefault(key);

    public string Get(string key, string fallback) => _values.TryGetValue(key, out var v) ? v : fallback;

    public int? GetInt(string key, int? fallback = null) => _values.TryGetValue(key, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;

    public double GetDouble(string key, double fallback) => _values.TryGetValue(key, out var v) ? double.Parse(v, CultureInfo.InvariantCulture) : fallback;

    public string[]? GetList(string key) => _values.TryGetValue(key, out var v) ? v.Split(',', StringSplitOptions.RemoveEmptyEntries) : null;
}

/// <summary>Appends records to a raw result file (JSON lines).</summary>
internal static class RawWriter
{
    public static void Write(string? path, IEnumerable<JsonObject> records)
    {
        if (string.IsNullOrEmpty(path))
        {
            throw new ArgumentException("--out <raw.jsonl> is required.");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var writer = new StreamWriter(path, append: true);
        foreach (var record in records)
        {
            writer.WriteLine(record.ToJsonString(new JsonSerializerOptions { WriteIndented = false }));
        }
    }
}

/// <summary>Raw timings of one timed series: per-sample totals, stages and metrics, and the series wall time.</summary>
internal sealed record TimedSamples(double[] Totals, double[,] Stages, double[,] Metrics, double WallMs)
{
    public JsonArray TotalsJson() => new(Totals.Select(x => (JsonNode)x).ToArray());

    public JsonObject StagesJson(IReadOnlyList<string> names) => Columns(Stages, names);

    public JsonObject MetricsJson(IReadOnlyList<string> names) => Columns(Metrics, names);

    private JsonObject Columns(double[,] values, IReadOnlyList<string> names)
    {
        var json = new JsonObject();
        for (int s = 0; s < names.Count; s++)
        {
            json[names[s]] = new JsonArray(Enumerable.Range(0, Totals.Length).Select(i => (JsonNode)values[i, s]).ToArray());
        }

        return json;
    }
}

/// <summary>The timed loop shared by every harness: <c>n</c> calls on <c>concurrency</c> threads, each sample timed around the call.</summary>
internal static class SeriesTimer
{
    /// <summary>
    /// Runs <paramref name="call"/> <paramref name="n"/> times. Arguments: sample index, worker index, and per-worker
    /// buffers the call fills with its stage times (ms) and metrics, copied into the sample after the call.
    /// </summary>
    public static TimedSamples Time(int n, int concurrency, int stageCount, int metricCount, Action<int, int, double[], double[]> call)
    {
        var totals = new double[n];
        var stages = new double[n, stageCount];
        var metrics = new double[n, metricCount];
        int next = -1;
        void Worker(int worker)
        {
            var localStages = new double[Math.Max(4, stageCount)];
            var localMetrics = new double[metricCount];
            while (true)
            {
                int i = Interlocked.Increment(ref next);
                if (i >= n)
                {
                    return;
                }

                long t = Stopwatch.GetTimestamp();
                call(i, worker, localStages, localMetrics);
                totals[i] = MonotonicClock.ElapsedMs(t);
                for (int s = 0; s < stageCount; s++)
                {
                    stages[i, s] = localStages[s];
                }

                for (int m = 0; m < metricCount; m++)
                {
                    metrics[i, m] = localMetrics[m];
                }
            }
        }

        long start = Stopwatch.GetTimestamp();
        if (concurrency == 1)
        {
            Worker(0);
        }
        else
        {
            var threads = Enumerable.Range(0, concurrency).Select(k => new Thread(() => Worker(k))).ToList();
            threads.ForEach(t => t.Start());
            threads.ForEach(t => t.Join());
        }

        return new TimedSamples(totals, stages, metrics, MonotonicClock.ElapsedMs(start));
    }

    public static double Median(IEnumerable<double> values)
    {
        var sorted = values.Order().ToArray();
        return sorted[sorted.Length / 2];
    }
}

/// <summary>Cold starts: one fresh process per sample, started with its spawn time on CLOCK_MONOTONIC.</summary>
internal static class ColdRunner
{
    public const string SpawnEnvironmentVariable = "STARLING_BENCH_SPAWN_MONOTONIC_NS";

    /// <summary>
    /// Starts this executable <paramref name="processes"/> times as <c>cold-child</c> with <paramref name="arguments"/>;
    /// returns each child's last stdout line (a JSON object) and its spawn-to-exit wall time.
    /// </summary>
    public static (List<JsonObject> Children, List<double> Walls) Spawn(int processes, IEnumerable<(string Key, string Value)> arguments)
    {
        string self = Environment.ProcessPath ?? throw new InvalidOperationException("No process path.");
        var args = arguments.ToList();
        var children = new List<JsonObject>();
        var walls = new List<double>();
        for (int i = 0; i < processes; i++)
        {
            var psi = new ProcessStartInfo(self)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            if (Path.GetFileNameWithoutExtension(self) == "dotnet")
            {
                // Started as 'dotnet <harness>.dll': the child needs the assembly too.
                psi.ArgumentList.Add(System.Reflection.Assembly.GetEntryAssembly()!.Location);
            }

            psi.ArgumentList.Add("cold-child");
            foreach (var (key, value) in args)
            {
                psi.ArgumentList.Add("--" + key);
                psi.ArgumentList.Add(value);
            }

            long t0 = Stopwatch.GetTimestamp();
            psi.Environment[SpawnEnvironmentVariable] = MonotonicClock.NowNs().ToString(CultureInfo.InvariantCulture);
            using var process = Process.Start(psi)!;
            var stderr = process.StandardError.ReadToEndAsync();
            string stdout = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            walls.Add(MonotonicClock.ElapsedMs(t0));
            if (process.ExitCode != 0)
            {
                throw new InvalidOperationException($"cold child failed ({process.ExitCode}): {stderr.Result}");
            }

            children.Add(JsonNode.Parse(stdout.Trim().Split('\n')[^1])!.AsObject());
        }

        return (children, walls);
    }

    /// <summary>The spawn time passed by the parent, in CLOCK_MONOTONIC nanoseconds.</summary>
    public static long SpawnNs() =>
        long.Parse(Environment.GetEnvironmentVariable(SpawnEnvironmentVariable) ?? throw new InvalidOperationException("Not started by 'cold'."), CultureInfo.InvariantCulture);

    /// <summary>Per-stage arrays over the children (every numeric field except <paramref name="skip"/>), plus <c>process_wall</c>.</summary>
    public static JsonObject StageArrays(IReadOnlyList<JsonObject> children, IReadOnlyList<double> walls, IReadOnlyCollection<string> skip)
    {
        var stages = new JsonObject();
        foreach (string name in children[0].Select(kv => kv.Key).Where(k => !skip.Contains(k)))
        {
            stages[name] = new JsonArray(children.Select(c => (JsonNode)c[name]!.GetValue<double>()).ToArray());
        }

        stages["process_wall"] = new JsonArray(walls.Select(w => (JsonNode)w).ToArray());
        return stages;
    }
}
