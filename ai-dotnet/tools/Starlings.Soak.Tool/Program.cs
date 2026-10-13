// Gate G4 soak (plan 10.2): repeated requests and load/unload cycles for one model profile, with memory sampling and
// a post-warmup plateau and growth analysis.
//
//   dotnet run --project ai-dotnet/tools/Starlings.Soak.Tool -c Release -- <model|all> --out <dir> [options]
//
// Models: e5, gliner2, embeddinggemma2 (text), embeddinggemma2-image, qwen35 (fp32x), or all (each model in its own
// child process, one after the other; a model whose files are not in the cache is reported as skipped).
//
// Every cycle loads the model (tokenizer and sessions), checks that each fixed input gives the reference output of the
// first cycle (state-leakage check, not fidelity), serves --requests-per-cycle requests from --concurrency callers in
// parallel, of which every --deadline-every-th is a long request with a deadline (CancelAfter, 5 to 150 ms) whose
// cancellation latency is recorded, takes a settled memory sample (full GC) while loaded, disposes everything, and
// takes a settled sample after unloading. A background thread samples memory every --sample-interval. After the run,
// the settled series past the warmup are analysed for a plateau and a growth slope (Analysis.cs).
//
// Writes <out>/<model>-<utc>.json (raw: environment manifest, options, cycles, samples, latencies, verdicts) and
// <out>/<model>-<utc>.md (report). Exit code: 0 no unbounded growth, no wrong output, no unexpected error; 1 a failure
// (any growth verdict, wrong output or unexpected exception); 2 usage error; 3 model files not available.
//
// Options:
//   --duration <t>            Run length: 1h (default; the plan 10.2 qualification length), 5m, 300s, or seconds.
//                             Below one hour the report is labelled a short development run.
//   --out <dir>               Output directory (required).
//   --cuda                    CUDA provider (build with -p:UseCuda=true). Default CPU.
//   --threads <n>             ONNX Runtime intra-op threads (default: runtime choice).
//   --concurrency <n>         Parallel callers per cycle (default 2).
//   --requests-per-cycle <n>  Requests per load/unload cycle (default per model).
//   --deadline-every <n>      Every n-th request is a long request with a deadline (default 10; 0 disables).
//   --sample-interval <t>     Background sampling interval (default 5s).
//   --warmup <fraction>       Fraction of the run excluded as warmup (default 0.2; at least the first 2 cycles).
//   --growth-limit-mb <mb>    Floor of the growth limit (default 32; the limit is max(floor, 2% of the plateau)).
//   --note <text>             Free-text machine note recorded in the manifest.
//   --seed <n>                Seed of the deadline delays (default 1).
//   --malloc-trim             Diagnostic (glibc only): after each unloaded sample, call malloc_trim(0) and sample again
//                             (series "trimmed"). Memory that malloc_trim returns was freed but kept by the allocator,
//                             not leaked. Changes allocator behaviour for the rest of the run; off by default.
//
// Reanalysis: `-- reanalyze <raw.json>...` recomputes the verdicts and the markdown report of existing raw results
// with the current analysis (the run's warmup and growth-limit options), rewriting both files; the samples are unchanged.
//
// Gating: a growth verdict fails the run for managed heap, working set, RssAnon and GPU memory. On Linux, .NET's
// private bytes is the VmData address space and is reported only (Memory.cs, MemorySample.Gates). Growth between loads
// that the next load reuses (unloaded series grows, loaded series plateaus) is "retention", reported only
// (SoakRunner.Analyze).
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.Error.WriteLine("usage: Starlings.Soak.Tool <e5|gliner2|embeddinggemma2|embeddinggemma2-image|qwen35|all> --out <dir> [--duration 1h] [options]; see Program.cs");
    return args.Length == 0 ? 2 : 0;
}

if (args[0] == "reanalyze")
{
    return Reanalysis.Run(args[1..]);
}

SoakOptions options;
try
{
    options = SoakOptions.Parse(args[1..]);
}
catch (ArgumentException ex)
{
    Console.Error.WriteLine(ex.Message);
    return 2;
}

if (args[0] == "all")
{
    return SoakRunner.RunAll(args[1..]);
}

var workload = SoakWorkload.All.FirstOrDefault(w => w.Name == args[0]);
if (workload is null)
{
    Console.Error.WriteLine($"unknown model '{args[0]}'; expected one of {string.Join(", ", SoakWorkload.All.Select(w => w.Name))}, all");
    return 2;
}

return new SoakRunner(workload, options).Run();

internal sealed record SoakOptions
{
    public TimeSpan Duration { get; init; } = TimeSpan.FromHours(1);

    public string Out { get; init; } = string.Empty;

    public bool Cuda { get; init; }

    public int? Threads { get; init; }

    public int Concurrency { get; init; } = 2;

    public int? RequestsPerCycle { get; init; }

    public int DeadlineEvery { get; init; } = 10;

    public TimeSpan SampleInterval { get; init; } = TimeSpan.FromSeconds(5);

    public double Warmup { get; init; } = 0.2;

    public double GrowthLimitMb { get; init; } = 32;

    public string? Note { get; init; }

    public int Seed { get; init; } = 1;

    public bool MallocTrim { get; init; }

    public bool IsQualificationLength => Duration >= TimeSpan.FromHours(1);

    public static SoakOptions Parse(string[] args)
    {
        var o = new SoakOptions();
        for (int i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
            o = args[i] switch
            {
                "--duration" => o with { Duration = ParseTime(Next()) },
                "--out" => o with { Out = Next() },
                "--cuda" => o with { Cuda = true },
                "--threads" => o with { Threads = Positive(Next(), "--threads") },
                "--concurrency" => o with { Concurrency = Positive(Next(), "--concurrency") },
                "--requests-per-cycle" => o with { RequestsPerCycle = Positive(Next(), "--requests-per-cycle") },
                "--deadline-every" => o with { DeadlineEvery = int.Parse(Next(), CultureInfo.InvariantCulture) },
                "--sample-interval" => o with { SampleInterval = ParseTime(Next()) },
                "--warmup" => o with { Warmup = double.Parse(Next(), CultureInfo.InvariantCulture) },
                "--growth-limit-mb" => o with { GrowthLimitMb = double.Parse(Next(), CultureInfo.InvariantCulture) },
                "--note" => o with { Note = Next() },
                "--seed" => o with { Seed = int.Parse(Next(), CultureInfo.InvariantCulture) },
                "--malloc-trim" => o with { MallocTrim = true },
                _ => throw new ArgumentException($"unknown option '{args[i]}'"),
            };
        }

        if (string.IsNullOrEmpty(o.Out))
        {
            throw new ArgumentException("--out <dir> is required.");
        }

        if (o.Warmup is < 0 or >= 1 || o.DeadlineEvery < 0 || o.Duration <= TimeSpan.Zero || o.SampleInterval <= TimeSpan.Zero)
        {
            throw new ArgumentException("--warmup must be in [0, 1); --deadline-every >= 0; --duration and --sample-interval positive.");
        }

        return o;
    }

    private static int Positive(string s, string name) =>
        int.TryParse(s, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : throw new ArgumentException($"{name} must be a positive integer.");

    private static TimeSpan ParseTime(string s)
    {
        double Number(string n) => double.Parse(n, CultureInfo.InvariantCulture);
        return s.EndsWith('h') ? TimeSpan.FromHours(Number(s[..^1]))
            : s.EndsWith("ms", StringComparison.Ordinal) ? TimeSpan.FromMilliseconds(Number(s[..^2]))
            : s.EndsWith('m') ? TimeSpan.FromMinutes(Number(s[..^1]))
            : s.EndsWith('s') ? TimeSpan.FromSeconds(Number(s[..^1]))
            : TimeSpan.FromSeconds(Number(s));
    }
}

/// <summary>One load/serve/unload cycle.</summary>
internal sealed record CycleRecord(
    int Index,
    double StartSeconds,
    double EndSeconds,
    double LoadMs,
    double UnloadMs,
    int Requests,
    int DeadlineRequests,
    double RequestP50Ms,
    double RequestP95Ms,
    double RequestMaxMs,
    MemorySample Loaded,
    MemorySample Unloaded,
    MemorySample? Trimmed);

internal sealed class SoakRunner(SoakWorkload workload, SoakOptions options)
{
    private const int DeadlineMinMs = 5;
    private const int DeadlineMaxMs = 150;

    private readonly ConcurrentBag<double> _cancelLatencies = [];
    private readonly ConcurrentQueue<string> _mismatches = new();
    private readonly ConcurrentQueue<string> _errors = new();
    private int _deadlineCompleted;
    private int _deadlineEvery;

    public static int RunAll(string[] rest)
    {
        int worst = 0;
        foreach (var w in SoakWorkload.All)
        {
            var info = new ProcessStartInfo(Environment.ProcessPath!) { UseShellExecute = false };
            if (!Environment.ProcessPath!.EndsWith("Starlings.Soak.Tool", StringComparison.Ordinal) &&
                !Environment.ProcessPath.EndsWith("Starlings.Soak.Tool.exe", StringComparison.Ordinal))
            {
                info.ArgumentList.Add(typeof(SoakRunner).Assembly.Location);
            }

            info.ArgumentList.Add(w.Name);
            foreach (string a in rest)
            {
                info.ArgumentList.Add(a);
            }

            using var child = Process.Start(info)!;
            child.WaitForExit();
            Console.WriteLine($"[all] {w.Name}: exit {child.ExitCode}");
            if (child.ExitCode is 1 or 2)
            {
                worst = Math.Max(worst, child.ExitCode == 1 ? 1 : 2);
            }
        }

        return worst;
    }

    public int Run()
    {
        Directory.CreateDirectory(options.Out);
        string stamp = DateTime.UtcNow.ToString("yyyyMMdd'T'HHmmss'Z'", CultureInfo.InvariantCulture);
        string basePath = Path.Combine(options.Out, $"{workload.Name}-{stamp}");
        var loadBefore = SoakEnvironment.Load();
        Console.WriteLine($"[{workload.Name}] verifying files (SHA-256, offline)...");
        string? missing = workload.Locate(options);
        if (missing is not null)
        {
            Console.Error.WriteLine($"[{workload.Name}] skipped: {missing}");
            File.WriteAllText(basePath + ".skipped.txt", missing + "\n");
            return 3;
        }

        var clock = Stopwatch.StartNew();
        var cycles = new List<CycleRecord>();
        SoakResult[]? reference = null;
        IReadOnlyList<string> providers = [];
        int requestsPerCycle = options.RequestsPerCycle ?? workload.DefaultRequestsPerCycle;

        // At least one deadline request per cycle when the cycle is shorter than the interval.
        _deadlineEvery = options.DeadlineEvery == 0 ? 0 : Math.Min(options.DeadlineEvery, requestsPerCycle);
        var baseline = MemoryProbe.Sample(0, "baseline", -1, settle: true);
        var lastCheckpoint = TimeSpan.Zero;
        using (var sampler = new BackgroundSampler(clock, options.SampleInterval))
        {
            for (int cycle = 0; clock.Elapsed < options.Duration; cycle++)
            {
                double start = clock.Elapsed.TotalSeconds;
                sampler.SetPhase("load", cycle);
                long t = Stopwatch.GetTimestamp();
                ISoakInstance instance;
                try
                {
                    instance = workload.Load(options);
                }
                catch (Exception ex)
                {
                    _errors.Enqueue($"cycle {cycle} load: {ex.GetType().Name}: {ex.Message}");
                    break;
                }

                double loadMs = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
                providers = instance.SessionProviders;
                sampler.SetPhase("serve", cycle);
                if (reference is null)
                {
                    reference = Enumerable.Range(0, SoakWorkload.InputCount).Select(i => instance.Run(i, CancellationToken.None)).ToArray();
                }

                var (latencies, deadlines) = Serve(instance, reference, cycle, requestsPerCycle);
                var loaded = MemoryProbe.Sample(clock.Elapsed.TotalSeconds, "loaded", cycle, settle: true);
                sampler.SetPhase("unload", cycle);
                t = Stopwatch.GetTimestamp();
                instance.Dispose();
                double unloadMs = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
                var unloaded = MemoryProbe.Sample(clock.Elapsed.TotalSeconds, "unloaded", cycle, settle: true);
                MemorySample? trimmed = null;
                if (options.MallocTrim && MemoryProbe.MallocTrim())
                {
                    trimmed = MemoryProbe.Sample(clock.Elapsed.TotalSeconds, "trimmed", cycle, settle: false);
                }

                cycles.Add(new CycleRecord(
                    cycle, start, clock.Elapsed.TotalSeconds, loadMs, unloadMs, latencies.Count, deadlines,
                    SeriesVerdict.PercentileOf(latencies, 50), SeriesVerdict.PercentileOf(latencies, 95), latencies.Count == 0 ? double.NaN : latencies.Max(),
                    loaded, unloaded, trimmed));
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"[{workload.Name}] cycle {cycle} t={clock.Elapsed.TotalSeconds:F0}s load {loadMs:F0} ms, {latencies.Count} requests p50 {cycles[^1].RequestP50Ms:F1} ms, loaded rss_anon {loaded.RssAnonMb:F0} MB, unloaded rss_anon {unloaded.RssAnonMb:F0} MB, managed {unloaded.ManagedHeapMb:F1} MB"));
                if (!_errors.IsEmpty && cycle == 0)
                {
                    break;
                }

                // Checkpoint, so a run killed from outside (for example by an out-of-memory daemon) leaves its data.
                if (clock.Elapsed - lastCheckpoint > TimeSpan.FromSeconds(60))
                {
                    lastCheckpoint = clock.Elapsed;
                    Write(final: false);
                }
            }

            sampler.SetPhase("end", cycles.Count);
            string verdict = Write(final: true);
            Console.WriteLine($"[{workload.Name}] {verdict}; wrote {basePath}.json and .md");
            return verdict.StartsWith("FAIL", StringComparison.Ordinal) ? 1 : 0;

            string Write(bool final)
            {
                var verdicts = Analyze(cycles, clock.Elapsed.TotalSeconds, options.Warmup, options.GrowthLimitMb);
                List<MemorySample> samples;
                lock (sampler.Samples)
                {
                    samples = [.. sampler.Samples];
                }

                var environment = SoakEnvironment.Collect(loadBefore, options.Note);
                string verdict = OverallVerdict(cycles.Count, verdicts, _mismatches.Count, _errors.Count);
                if (!final)
                {
                    verdict = $"INCOMPLETE (checkpoint at {clock.Elapsed.TotalSeconds:F0} s of {options.Duration.TotalSeconds:F0} s; the run did not finish): " + verdict;
                }

                var json = ToJson(environment, providers, requestsPerCycle, baseline, cycles, samples, verdicts, verdict, clock.Elapsed.TotalSeconds);
                File.WriteAllText(basePath + ".json", json.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
                File.WriteAllText(basePath + ".md", Report.Markdown(json));
                return verdict;
            }
        }
    }

    private (List<double> Latencies, int Deadlines) Serve(ISoakInstance instance, SoakResult[] reference, int cycle, int requests)
    {
        var latencies = new ConcurrentBag<double>();
        int deadlines = 0;
        int next = -1;
        var workers = Enumerable.Range(0, options.Concurrency).Select(worker => Task.Factory.StartNew(() =>
        {
            var random = new Random(HashCode.Combine(options.Seed, cycle, worker));
            int k;
            while ((k = Interlocked.Increment(ref next)) < requests)
            {
                if (_deadlineEvery > 0 && k % _deadlineEvery == _deadlineEvery - 1)
                {
                    Interlocked.Increment(ref deadlines);
                    Deadline(instance, random.Next(DeadlineMinMs, DeadlineMaxMs + 1), cycle);
                    continue;
                }

                int input = k % reference.Length;
                long t = Stopwatch.GetTimestamp();
                try
                {
                    var result = instance.Run(input, CancellationToken.None);
                    latencies.Add(Stopwatch.GetElapsedTime(t).TotalMilliseconds);
                    if (result.Difference(reference[input]) is { } diff)
                    {
                        _mismatches.Enqueue($"cycle {cycle} request {k} input {input}: {diff}");
                    }
                }
                catch (Exception ex)
                {
                    _errors.Enqueue($"cycle {cycle} request {k}: {ex.GetType().Name}: {ex.Message}");
                }
            }
        }, CancellationToken.None, TaskCreationOptions.None, TaskScheduler.Default)).ToArray();
        Task.WaitAll(workers);
        return ([.. latencies], deadlines);
    }

    /// <summary>A long request with a deadline (a timed-out request); records the time from the deadline to the throw.</summary>
    private void Deadline(ISoakInstance instance, int deadlineMs, int cycle)
    {
        using var cts = new CancellationTokenSource();
        long cancelledAt = 0;
        using var registration = cts.Token.Register(() => Volatile.Write(ref cancelledAt, Stopwatch.GetTimestamp()));
        cts.CancelAfter(deadlineMs);
        try
        {
            instance.RunLong(cts.Token);
            Interlocked.Increment(ref _deadlineCompleted);
        }
        catch (OperationCanceledException) when (cts.IsCancellationRequested)
        {
            long now = Stopwatch.GetTimestamp();
            long at = Volatile.Read(ref cancelledAt);
            _cancelLatencies.Add(at == 0 ? 0 : (now - at) * 1000.0 / Stopwatch.Frequency);
        }
        catch (Exception ex)
        {
            _errors.Enqueue($"cycle {cycle} deadline request: {ex.GetType().Name}: {ex.Message}");
        }
    }

    /// <summary>
    /// Plateau and growth verdicts of the settled series past the warmup. Growth of an unloaded series whose loaded
    /// series plateaus is reclassified as <see cref="SeriesVerdict.Retention"/>: memory that persisted across an unload
    /// but was reused by the next load (otherwise the loaded series would grow by the same amount), which is allocator
    /// retention (glibc keeps freed heap memory and raises its mmap threshold), not a leak. It does not fail the run
    /// and is reported, because the resident footprint between loads is still larger.
    /// </summary>
    internal static List<SeriesVerdict> Analyze(List<CycleRecord> cycles, double totalSeconds, double warmup, double growthLimitMb)
    {
        double cutoff = warmup * totalSeconds;
        var kept = cycles.Where(c => c.Index >= 2 && c.EndSeconds >= cutoff).ToList();
        var verdicts = new List<SeriesVerdict>();
        var series = new List<(string, Func<CycleRecord, MemorySample?>)> { ("unloaded", c => c.Unloaded), ("loaded", c => c.Loaded) };
        if (kept.Any(c => c.Trimmed is not null))
        {
            series.Add(("trimmed", c => c.Trimmed));
        }

        foreach (var (name, pick) in series)
        {
            foreach (string metric in MemorySample.Metrics)
            {
                var points = kept.Select(c => (c.EndSeconds, pick(c)?.Get(metric))).Where(p => p.Item2 is not null).Select(p => (p.EndSeconds, p.Item2!.Value)).ToList();
                if (metric == "gpu" && (points.Count == 0 || points.All(p => p.Value == 0)))
                {
                    continue;
                }

                verdicts.Add(SeriesVerdict.Analyze(name, metric, points, growthLimitMb) with { Gates = MemorySample.Gates(metric) });
            }
        }

        for (int i = 0; i < verdicts.Count; i++)
        {
            var v = verdicts[i];
            if (v.Series is "unloaded" or "trimmed" && v.Verdict == SeriesVerdict.Growth &&
                verdicts.Any(l => l.Series == "loaded" && l.Metric == v.Metric && l.Verdict == SeriesVerdict.Plateau))
            {
                verdicts[i] = v with { Verdict = SeriesVerdict.Retention };
            }
        }

        return verdicts;
    }

    internal static string OverallVerdict(int cycles, List<SeriesVerdict> verdicts, int mismatches, int errors)
    {
        var reasons = new List<string>();
        if (cycles == 0)
        {
            reasons.Add("no cycle completed");
        }

        foreach (var v in verdicts.Where(v => v.Verdict == SeriesVerdict.Growth && v.Gates))
        {
            reasons.Add($"{v.Series} {v.Metric} grows {v.FittedGrowthMb:F0} MB ({v.SlopeMbPerHour:F0} MB/h)");
        }

        if (mismatches > 0)
        {
            reasons.Add($"{mismatches} wrong outputs");
        }

        if (errors > 0)
        {
            reasons.Add($"{errors} unexpected errors");
        }

        if (reasons.Count > 0)
        {
            return "FAIL: " + string.Join("; ", reasons);
        }

        string reportOnly = string.Join("; ", verdicts
            .Where(v => (v.Verdict == SeriesVerdict.Growth && !v.Gates) || (v.Verdict == SeriesVerdict.Retention && v.Gates))
            .Select(v => v.Verdict == SeriesVerdict.Retention
                ? $"{v.Series} {v.Metric} retention {v.FittedGrowthMb:F0} MB between loads, reused by the next load"
                : $"{v.Series} {v.Metric} (address space, report-only) grows {v.FittedGrowthMb:F0} MB"));
        string note = reportOnly.Length > 0 ? " (note: " + reportOnly + ")" : string.Empty;
        return verdicts.Any(v => v.Verdict == SeriesVerdict.Insufficient && v.Gates)
            ? "INCONCLUSIVE: too few post-warmup cycles for a growth verdict; no wrong output or error" + note
            : "PASS: every gating settled series plateaus after warmup; no wrong output or error" + note;
    }

    private JsonObject ToJson(JsonObject environment, IReadOnlyList<string> providers, int requestsPerCycle, MemorySample baseline,
        List<CycleRecord> cycles, List<MemorySample> samples, List<SeriesVerdict> verdicts, string verdict, double seconds)
    {
        var cancel = _cancelLatencies.ToArray();
        return new JsonObject
        {
            ["schema"] = "starlings-soak-raw/1",
            ["model"] = workload.Name,
            ["profile"] = workload.Profile(options),
            ["run_kind"] = options.IsQualificationLength ? "qualification-length" : "short-development",
            ["verdict"] = verdict,
            ["artifacts"] = workload.Artifacts(),
            ["session_providers"] = new JsonArray(providers.Select(p => (JsonNode)p).ToArray()),
            ["environment"] = environment,
            ["options"] = new JsonObject
            {
                ["duration_s"] = options.Duration.TotalSeconds,
                ["elapsed_s"] = seconds,
                ["provider"] = options.Cuda ? "cuda" : "cpu",
                ["threads"] = options.Threads,
                ["concurrency"] = options.Concurrency,
                ["requests_per_cycle"] = requestsPerCycle,
                ["deadline_every"] = _deadlineEvery,
                ["deadline_ms_range"] = new JsonArray(DeadlineMinMs, DeadlineMaxMs),
                ["sample_interval_s"] = options.SampleInterval.TotalSeconds,
                ["warmup_fraction"] = options.Warmup,
                ["growth_limit_floor_mb"] = options.GrowthLimitMb,
                ["seed"] = options.Seed,
                ["malloc_trim"] = options.MallocTrim,
                ["gpu_sampling"] = MemoryProbe.GpuSamplingAvailable,
            },
            ["baseline"] = Sample(baseline),
            ["cycles"] = new JsonArray(cycles.Select(c => (JsonNode)new JsonObject
            {
                ["index"] = c.Index,
                ["start_s"] = c.StartSeconds,
                ["end_s"] = c.EndSeconds,
                ["load_ms"] = c.LoadMs,
                ["unload_ms"] = c.UnloadMs,
                ["requests"] = c.Requests,
                ["deadline_requests"] = c.DeadlineRequests,
                ["request_p50_ms"] = Finite(c.RequestP50Ms),
                ["request_p95_ms"] = Finite(c.RequestP95Ms),
                ["request_max_ms"] = Finite(c.RequestMaxMs),
                ["loaded"] = Sample(c.Loaded),
                ["unloaded"] = Sample(c.Unloaded),
                ["trimmed"] = c.Trimmed is null ? null : Sample(c.Trimmed),
            }).ToArray()),
            ["cancellation"] = new JsonObject
            {
                ["deadline_requests_cancelled"] = cancel.Length,
                ["deadline_requests_completed_before_deadline"] = _deadlineCompleted,
                ["latency_ms_p50"] = Finite(SeriesVerdict.PercentileOf(cancel, 50)),
                ["latency_ms_p95"] = Finite(SeriesVerdict.PercentileOf(cancel, 95)),
                ["latency_ms_max"] = cancel.Length == 0 ? null : cancel.Max(),
                ["latencies_ms"] = new JsonArray(cancel.Select(x => (JsonNode)Math.Round(x, 3)).ToArray()),
            },
            ["wrong_outputs"] = new JsonArray(_mismatches.Take(50).Select(m => (JsonNode)m).ToArray()),
            ["wrong_output_count"] = _mismatches.Count,
            ["errors"] = new JsonArray(_errors.Take(50).Select(m => (JsonNode)m).ToArray()),
            ["error_count"] = _errors.Count,
            ["analysis"] = new JsonArray(verdicts.Select(v => (JsonNode)v.ToJson()).ToArray()),
            ["samples"] = new JsonArray(samples.Select(s => (JsonNode)Sample(s)).ToArray()),
        };
    }

    private static double? Finite(double x) => double.IsFinite(x) ? Math.Round(x, 3) : null;

    private static JsonObject Sample(MemorySample s) => new()
    {
        ["t_s"] = Math.Round(s.Seconds, 3),
        ["phase"] = s.Phase,
        ["cycle"] = s.Cycle,
        ["settled"] = s.Settled,
        ["managed_heap_mb"] = Math.Round(s.ManagedHeapMb, 3),
        ["working_set_mb"] = Math.Round(s.WorkingSetMb, 3),
        ["private_mb"] = Math.Round(s.PrivateMb, 3),
        ["rss_anon_mb"] = s.RssAnonMb is { } a ? Math.Round(a, 3) : null,
        ["rss_file_mb"] = s.RssFileMb is { } f ? Math.Round(f, 3) : null,
        ["vm_data_mb"] = s.VmDataMb is { } d ? Math.Round(d, 3) : null,
        ["gpu_mb"] = s.GpuMb,
    };
}

/// <summary>Markdown report from the raw JSON.</summary>
internal static class Report
{
    public static string Markdown(JsonObject r)
    {
        var sb = new StringBuilder();
        var o = r["options"]!.AsObject();
        var env = r["environment"]!.AsObject();
        var cycles = r["cycles"]!.AsArray();
        var cancel = r["cancellation"]!.AsObject();
        string F(JsonNode? n, string format = "F1") => n is null ? "n/a" : n.GetValue<double>().ToString(format, CultureInfo.InvariantCulture);

        sb.AppendLine(CultureInfo.InvariantCulture, $"# G4 soak: {r["profile"]}");
        sb.AppendLine();
        if ((string?)r["run_kind"] != "qualification-length")
        {
            sb.AppendLine(CultureInfo.InvariantCulture,
                $"> **Short development run ({F(o["duration_s"], "F0")} s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.");
            sb.AppendLine();
        }

        sb.AppendLine(CultureInfo.InvariantCulture, $"**Verdict:** {r["verdict"]}");
        sb.AppendLine();
        sb.AppendLine("## Run");
        sb.AppendLine();
        sb.AppendLine("| Item | Value |");
        sb.AppendLine("| --- | --- |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Model lock | {r["artifacts"]?["repository"]}@{r["artifacts"]?["revision"]} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Provider (session) | {o["provider"]} ({string.Join(", ", r["session_providers"]!.AsArray().Select(p => (string?)p))}) |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Duration requested / elapsed | {F(o["duration_s"], "F0")} s / {F(o["elapsed_s"], "F0")} s |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Load/unload cycles | {cycles.Count} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Requests per cycle, concurrency | {o["requests_per_cycle"]}, {o["concurrency"]} callers |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Intra-op threads | {o["threads"]?.ToString() ?? "runtime default"} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Warmup excluded | first 2 cycles and the first {F(o["warmup_fraction"], "F2")} of the run |");
        int requests = cycles.Sum(c => (int)c!["requests"]!);
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Requests checked against the first-cycle reference | {requests}; wrong outputs {r["wrong_output_count"]}; unexpected errors {r["error_count"]} |");
        if (cycles.Count > 0)
        {
            var load = cycles.Select(c => (double)c!["load_ms"]!).ToList();
            var unload = cycles.Select(c => (double)c!["unload_ms"]!).ToList();
            var p50 = cycles.Select(c => c!["request_p50_ms"]).Where(n => n is not null).Select(n => (double)n!).ToList();
            sb.AppendLine(CultureInfo.InvariantCulture, $"| Load ms (median, max) | {SeriesVerdict.MedianOf(load):F0}, {load.Max():F0} |");
            sb.AppendLine(CultureInfo.InvariantCulture, $"| Unload ms (median, max) | {SeriesVerdict.MedianOf(unload):F1}, {unload.Max():F1} |");
            if (p50.Count > 0)
            {
                sb.AppendLine(CultureInfo.InvariantCulture, $"| Request p50 ms (median over cycles) | {SeriesVerdict.MedianOf(p50):F1} |");
            }
        }

        sb.AppendLine();
        sb.AppendLine("## Timed-out requests and cancellation latency");
        sb.AppendLine();
        sb.AppendLine(CultureInfo.InvariantCulture,
            $"Every {o["deadline_every"]}th request is a long request with a deadline of {o["deadline_ms_range"]![0]} to {o["deadline_ms_range"]![1]} ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).");
        sb.AppendLine();
        sb.AppendLine("| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |");
        sb.AppendLine("| --- | --- | --- | --- | --- |");
        sb.AppendLine(CultureInfo.InvariantCulture,
            $"| {cancel["deadline_requests_cancelled"]} | {cancel["deadline_requests_completed_before_deadline"]} | {F(cancel["latency_ms_p50"], "F2")} | {F(cancel["latency_ms_p95"], "F2")} | {F(cancel["latency_ms_max"], "F2")} |");
        sb.AppendLine();
        sb.AppendLine("## Memory after warmup (settled samples, MiB)");
        sb.AppendLine();
        sb.AppendLine("`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.");
        sb.AppendLine();
        sb.AppendLine("| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |");
        sb.AppendLine("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
        foreach (var v in r["analysis"]!.AsArray())
        {
            sb.AppendLine(CultureInfo.InvariantCulture,
                $"| {v!["series"]} | {v["metric"]} | {((bool)v["gates"]! ? "yes" : "no")} | {v["points"]} | {F(v["first_mb"])} | {F(v["last_mb"])} | {F(v["median_mb"])} | {F(v["min_mb"])} | {F(v["max_mb"])} | {F(v["slope_mb_per_hour"])} | {F(v["fitted_growth_mb"])} | {F(v["first_half_growth_mb"])} | {F(v["second_half_growth_mb"])} | {F(v["limit_mb"])} | {v["verdict"]} |");
        }

        sb.AppendLine();
        sb.AppendLine("Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.");
        if (r["error_count"]!.GetValue<int>() > 0 || r["wrong_output_count"]!.GetValue<int>() > 0)
        {
            sb.AppendLine();
            sb.AppendLine("## Errors and wrong outputs (first 50)");
            sb.AppendLine();
            foreach (var e in r["errors"]!.AsArray().Concat(r["wrong_outputs"]!.AsArray()))
            {
                sb.AppendLine(CultureInfo.InvariantCulture, $"- {e}");
            }
        }

        sb.AppendLine();
        sb.AppendLine("## Environment");
        sb.AppendLine();
        sb.AppendLine("| Item | Value |");
        sb.AppendLine("| --- | --- |");
        foreach (string key in new[] { "cpu_model", "logical_cpus", "ram_total_mb", "os", "kernel", "runtime", "server_gc", "ort_version", "genai_version", "genai_native_package", "gpu_identity", "cpu_governor", "allocator_and_runtime_env", "git_commit", "git_dirty", "command_line", "machine_note" })
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $"| {key} | {env[key]?.ToJsonString().Trim('"').Replace("\\n", "; ", StringComparison.Ordinal).Replace("|", "/", StringComparison.Ordinal) ?? "n/a"} |");
        }

        sb.AppendLine(CultureInfo.InvariantCulture, $"| load before (loadavg, MemAvailable MB) | {env["load_before"]?["loadavg"]?.ToJsonString()}, {env["load_before"]?["mem_available_mb"]} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| load after (loadavg, MemAvailable MB) | {env["load_after"]?["loadavg"]?.ToJsonString()}, {env["load_after"]?["mem_available_mb"]} |");
        return sb.ToString();
    }
}
