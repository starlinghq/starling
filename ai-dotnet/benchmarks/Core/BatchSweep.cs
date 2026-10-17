using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Starlings.Benchmarks;

/// <summary>
/// The batching-sweep mode shared by the e5, EmbeddingGemma 2 (text) and GLiNER2 harnesses: for each workload set
/// (<c>short</c>, <c>long</c>, <c>mixed</c>; <see cref="SweepWorkloadFile"/>), batch size and concurrency level, a timed
/// cell of calls through the product API, with per-call latency, stage times, real tokens and padded cells, the
/// process's peak RSS (and the device's peak memory on CUDA), and every output compared with the same item run alone.
/// Writes <c>starlings-bench-sweep/1</c> (<c>model-pipeline/schemas/benchmark-sweep.schema.json</c>).
/// </summary>
internal static class BatchSweep
{
    public const string Schema = "starlings-bench-sweep/1";

    public const string Usage = """
        sweep --out <sweep.jsonl> [options]   batching sweep (starlings-bench-sweep/1; model-pipeline bench sweep <file>)
          --sets short,long,mixed       workload sets of the sweep file (also: medium)
          --batches 1,4,8,16,32,64      items per call       --concurrency 1,4,16   callers issuing calls at once
          --min-seconds S (default 2)   --min-calls N (default 10)   --max-calls N (default 5000)   per cell
          --warmup N (default 3)        untimed calls per cell before the timed calls
          --pool N                      items per pool (default: the sweep file's pool)
          --max-batch-size default|call|N  the library's rows per ORT run: its default, the largest swept batch
                                        (default: one run per call), or N
          --quick true                  reduced grid: batches 1,8,32, concurrency 1,4, 1 s per cell, pool 32
          --preliminary true|false      the harness's fixture quality sweep first (default true)
          --vram auto|off               sample device memory with nvidia-smi on CUDA (default auto)
          --max-inflight-items N        skip cells whose batch x concurrency exceeds N (memory guard; listed as skipped)
          --threads N                   intra_op_num_threads (sweep it with separate processes; see run-batching-sweep.sh)
          --workloads <file>            default ai-dotnet/benchmarks/workloads/batching-sweep-v1.json
        Hooks of the option contract, rejected until the integrate step wires them (exit 2):
          --sort-by-length true|false   --max-batch-tokens N (wired for EmbeddingGemma 2, whose option exists)
          --microbatch-wait-ms N        --microbatch-max-rows N
        """;

    /// <summary>
    /// Parses the sweep options, loads the target and runs the sweep. Exit codes as the other commands: 0 pass,
    /// 1 quality failed, 2 usage, 3 assets missing.
    /// </summary>
    public static int Command(string[] args, string harnessName, bool cudaBuild, Func<HarnessContext, SweepOptions, ISweepTarget> load, IReadOnlyCollection<string>? wiredHooks = null)
    {
        try
        {
            var map = ArgMap.Parse(args, Usage);
            string provider = HarnessHost.ParseProvider(map.Get("provider", cudaBuild ? "cuda" : "cpu")!, harnessName);
            var file = SweepWorkloadFile.Load(map.Get("workloads"));
            var context = new HarnessContext(map, file.Path, map.GetInt("threads"), map.GetInt("inter-threads", 1), map.Get("hash-model", "false") == "true", provider);
            var options = SweepOptions.Parse(map, wiredHooks);
            using var target = load(context, options);
            return Run(target, context, options, file);
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }
        catch (BenchAssetsMissingException ex)
        {
            Console.Error.WriteLine($"ASSETS MISSING ({harnessName}): {ex.Message}");
            return 3;
        }
    }

    public static int Run(ISweepTarget target, HarnessContext context, SweepOptions o, SweepWorkloadFile file)
    {
        var loadBefore = BenchEnvironment.LoadSnapshot();
        string started = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
        string runId = Guid.NewGuid().ToString();
        string output = o.Out;
        long tPools = Stopwatch.GetTimestamp();
        var pools = file.Build(target.Model, target.CountTokens, target.MaxTokens, o.Pool);
        double poolMs = MonotonicClock.ElapsedMs(tPools);
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"pools built in {poolMs:F0} ms: {pools}"));
        foreach (string set in o.Sets)
        {
            if (!pools.Pools.TryGetValue(set, out var p) || p.Length == 0)
            {
                throw new ArgumentException($"--sets: '{set}' is not a non-empty pool of {target.Model} (have {string.Join(", ", pools.Pools.Keys)}).");
            }
        }

        JsonObject? preliminary = null;
        if (o.Preliminary)
        {
            preliminary = target.PreliminaryQuality();
            Console.Error.WriteLine($"preliminary quality: pass={preliminary?["pass"]}");
        }

        // Every item alone, once, before any timed call: the reference each batched output is compared with.
        long tSingles = Stopwatch.GetTimestamp();
        var singles = new Dictionary<string, object>(StringComparer.Ordinal);
        var scratch = new double[Math.Max(4, target.Stages.Length)];
        foreach (var item in o.Sets.SelectMany(s => pools.Pools[s]))
        {
            if (!singles.ContainsKey(item.Id))
            {
                singles[item.Id] = target.Single(item, target.Call([item], scratch));
            }
        }

        double singlesMs = MonotonicClock.ElapsedMs(tSingles);
        var run = new JsonObject
        {
            ["schema"] = Schema,
            ["kind"] = "run",
            ["run_id"] = runId,
            ["started_utc"] = started,
            ["mode"] = "sweep",
            ["model"] = target.Model,
            ["profile"] = target.Profile,
            ["harness"] = new JsonObject { ["name"] = target.HarnessName, ["language"] = "csharp", ["impl"] = "library", ["version"] = "1" },
            ["artifacts"] = target.Artifacts(file),
            ["environment"] = BenchEnvironment.Environment(context.IsCuda, target.SessionProviders, loadBefore, context.Args.Get("note")),
            ["settings"] = target.Settings,
            ["load_ms"] = target.LoadStages.DeepClone(),
            ["sweep"] = new JsonObject
            {
                ["sets"] = Json(o.Sets),
                ["batches"] = Json(o.Batches),
                ["concurrency"] = Json(o.Concurrency),
                ["min_seconds"] = o.MinSeconds,
                ["min_calls"] = o.MinCalls,
                ["max_calls"] = o.MaxCalls,
                ["warmup_calls"] = o.Warmup,
                ["max_batch_size"] = o.MaxBatchSizeLabel,
                ["seed"] = file.Seed,
                ["pools"] = pools.Describe(),
                ["pool_build_ms"] = poolMs,
                ["singles_ms"] = singlesMs,
                ["quality_check"] = target.QualityCheck,
                ["tolerance"] = target.Tolerance,
                ["tolerance_file"] = target.ToleranceFile,
                ["queue_time"] = "not separable: no scheduler in the library yet (stage 'queue' appears once micro-batching is wired)",
                ["rss"] = "VmHWM of /proc/self/status after resetting it through /proc/self/clear_refs before each cell's warmup; includes managed heap, native arenas and everything the process mapped (ORT arenas do not shrink, so a cell's peak can carry an earlier cell's high-water mark)",
                ["vram"] = context.IsCuda ? "device memory.used (whole GPU, all processes) sampled by nvidia-smi -lms 100; the CUDA arena does not shrink, so peaks are cumulative in cell order" : "n/a (CPU)",
            },
            ["quality"] = preliminary,
        };
        RawWriter.Write(output, [run]);

        using var gpu = context.IsCuda && o.Vram ? GpuMemorySampler.TryStart() : null;
        var failed = new JsonArray();
        var skipped = new JsonArray();
        int cells = 0;
        foreach (string set in o.Sets)
        {
            var pool = pools.Pools[set];
            foreach (int batch in o.Batches)
            {
                foreach (int concurrency in o.Concurrency)
                {
                    if (o.MaxInflightItems is int cap && batch * concurrency > cap)
                    {
                        skipped.Add((JsonNode)$"{set}/b{batch}/c{concurrency}");
                        Console.Error.WriteLine($"{set} b={batch} c={concurrency}: skipped (batch x concurrency > --max-inflight-items {cap})");
                        continue;
                    }

                    var cell = Cell(target, pools, pool, set, batch, concurrency, o, singles, gpu);
                    cell["schema"] = Schema;
                    cell["kind"] = "cell";
                    cell["run_id"] = runId;
                    RawWriter.Write(output, [cell]);
                    cells++;
                    var q = cell["quality"]!;
                    if (!q["pass"]!.GetValue<bool>())
                    {
                        failed.Add((JsonNode)$"{set}/b{batch}/c{concurrency}");
                    }

                    Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"{set} b={batch} c={concurrency}: {cell["items_per_s"]!.GetValue<double>():F1} items/s, p50 {cell["latency_p50_ms"]!.GetValue<double>():F2} ms, padding eff {cell["padding_efficiency"]!.GetValue<double>():F3}, peak RSS {cell["rss"]!["peak_bytes"]!.GetValue<long>() / 1048576.0:F0} MiB, quality {q["max_diff"]!.GetValue<double>():E2} pass={q["pass"]}"));
                }
            }
        }

        bool preliminaryPass = preliminary is null || (preliminary["pass"]?.GetValue<bool>() ?? false);
        bool pass = preliminaryPass && failed.Count == 0;
        RawWriter.Write(output, [new JsonObject
        {
            ["schema"] = Schema,
            ["kind"] = "end",
            ["run_id"] = runId,
            ["finished_utc"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            ["cells"] = cells,
            ["preliminary_pass"] = preliminary is null ? null : preliminaryPass,
            ["failed_cells"] = failed,
            ["skipped_cells"] = skipped,
            ["pass"] = pass,
            ["load_after"] = BenchEnvironment.LoadSnapshot(),
        }]);
        if (!pass)
        {
            Console.Error.WriteLine($"QUALITY FAILED: preliminary pass={preliminaryPass}, failed cells: {failed.ToJsonString()}. Timings of this run are not valid results.");
        }

        return pass ? 0 : 1;
    }

    private static JsonObject Cell(ISweepTarget target, SweepPools pools, SweepItem[] pool, string set, int batch, int concurrency, SweepOptions o, Dictionary<string, object> singles, GpuMemorySampler? gpu)
    {
        SweepItem[] Request(long call)
        {
            var items = new SweepItem[batch];
            for (int k = 0; k < batch; k++)
            {
                items[k] = pool[(int)((call * batch + k) % pool.Length)];
            }

            return items;
        }

        long rssBefore = ProcessMemory.ResetPeak();
        double? vramBefore = gpu?.Reset();
        var scratch = new double[Math.Max(4, target.Stages.Length)];
        for (int i = 0; i < o.Warmup; i++)
        {
            target.Call(Request(i), scratch);
        }

        int capacity = o.MaxCalls;
        var latency = new double[capacity];
        var offset = new double[capacity];
        var worker = new int[capacity];
        var stages = new double[capacity, target.Stages.Length];
        var realTokens = new long[capacity];
        var paddedCells = new long[capacity];
        var runs = new int[capacity];
        var completed = new bool[capacity];
        int next = -1;
        var check = SweepCheck.Empty;
        var gate = new object();
        double minMs = o.MinSeconds * 1000;
        long start = Stopwatch.GetTimestamp();

        void Worker(int w)
        {
            var local = new double[Math.Max(4, target.Stages.Length)];
            var lengths = new int[batch];
            while (true)
            {
                int i = Interlocked.Increment(ref next);
                if (i >= capacity || (i >= o.MinCalls && MonotonicClock.ElapsedMs(start) >= minMs))
                {
                    return;
                }

                // Calls are numbered across workers, so call i always carries the same items (offset by the warmup).
                var items = Request(o.Warmup + i);
                long t = Stopwatch.GetTimestamp();
                var result = target.Call(items, local);
                latency[i] = MonotonicClock.ElapsedMs(t);
                offset[i] = Stopwatch.GetElapsedTime(start, t).TotalMilliseconds;
                worker[i] = w;
                for (int s = 0; s < target.Stages.Length; s++)
                {
                    stages[i, s] = local[s];
                }

                long real = 0;
                for (int k = 0; k < batch; k++)
                {
                    lengths[k] = pools.Tokens(items[k]);
                    real += lengths[k];
                }

                var (cells, chunkCount) = SweepMath.Padding(lengths, target.PlanChunks(lengths));
                realTokens[i] = real;
                paddedCells[i] = cells;
                runs[i] = chunkCount;

                // Outside the timed region: every output against the same item run alone at load.
                var c = target.Compare(items, result, singles);
                lock (gate)
                {
                    check = check.Merge(c);
                }

                completed[i] = true;
            }
        }

        if (concurrency == 1)
        {
            Worker(0);
        }
        else
        {
            var threads = Enumerable.Range(0, concurrency).Select(k => new Thread(() => Worker(k)) { IsBackground = true }).ToList();
            threads.ForEach(t => t.Start());
            threads.ForEach(t => t.Join());
        }

        double wall = MonotonicClock.ElapsedMs(start);
        // A worker can stop on the time limit while another, which drew a later index a moment earlier, still runs:
        // keep the calls that ran, in index order.
        var ran = Enumerable.Range(0, capacity).Where(i => completed[i]).ToArray();
        int n = ran.Length;
        latency = ran.Select(i => latency[i]).ToArray();
        offset = ran.Select(i => offset[i]).ToArray();
        worker = ran.Select(i => worker[i]).ToArray();
        realTokens = ran.Select(i => realTokens[i]).ToArray();
        paddedCells = ran.Select(i => paddedCells[i]).ToArray();
        runs = ran.Select(i => runs[i]).ToArray();
        long rssPeak = ProcessMemory.PeakBytes();
        double? vramPeak = gpu?.Peak();
        long items = (long)n * batch;
        long sumReal = realTokens.Take(n).Sum(), sumPadded = paddedCells.Take(n).Sum();
        var sorted = latency.Take(n).Order().ToArray();
        var stageJson = new JsonObject();
        for (int s = 0; s < target.Stages.Length; s++)
        {
            int column = s;
            stageJson[target.Stages[s]] = Json(ran.Select(i => Math.Round(stages[i, column], 4)));
        }

        double tolerance = target.Tolerance;
        bool pass = check.Items == items && check.Finite && check.Mismatches == 0 && check.MaxDiff <= tolerance;
        return new JsonObject
        {
            ["model"] = target.Model,
            ["provider"] = target.Provider,
            ["set"] = set,
            ["batch"] = batch,
            ["concurrency"] = concurrency,
            ["intra_op_num_threads"] = target.IntraOpThreads,
            ["max_batch_size"] = target.MaxBatchSize,
            ["max_batch_tokens"] = target.MaxBatchTokens,
            ["warmup_calls"] = o.Warmup,
            ["n"] = n,
            ["items"] = items,
            ["wall_ms"] = wall,
            ["items_per_s"] = n == 0 ? 0 : items / (wall / 1000),
            ["calls_per_s"] = n == 0 ? 0 : n / (wall / 1000),
            ["latency_p50_ms"] = n == 0 ? 0 : sorted[n / 2],
            ["latency_p95_ms"] = n < 20 ? null : sorted[Math.Min(n - 1, (int)Math.Ceiling(n * 0.95) - 1)],
            ["padding_efficiency"] = sumPadded == 0 ? 1 : (double)sumReal / sumPadded,
            ["real_tokens"] = sumReal,
            ["padded_cells"] = sumPadded,
            ["ms"] = Json(latency.Take(n).Select(x => Math.Round(x, 4))),
            ["offset_ms"] = Json(offset.Take(n).Select(x => Math.Round(x, 3))),
            ["worker"] = Json(worker.Take(n)),
            ["stages_ms"] = stageJson,
            ["call_real_tokens"] = Json(realTokens.Take(n)),
            ["call_padded_cells"] = Json(paddedCells.Take(n)),
            ["call_runs"] = Json(runs.Take(n)),
            ["rss"] = new JsonObject { ["before_bytes"] = rssBefore, ["peak_bytes"] = rssPeak },
            ["vram"] = gpu is null ? null : new JsonObject { ["before_mib"] = vramBefore, ["peak_mib"] = vramPeak },
            ["quality"] = new JsonObject
            {
                ["check"] = target.QualityCheck,
                ["items_checked"] = check.Items,
                ["max_diff"] = check.MaxDiff,
                ["tolerance"] = tolerance,
                ["finite"] = check.Finite,
                ["mismatches"] = check.Mismatches,
                ["worst"] = check.Worst,
                ["pass"] = pass,
            },
        };
    }

    private static JsonArray Json<T>(IEnumerable<T> values) => new(values.Select(v => JsonValue.Create(v)).Cast<JsonNode?>().ToArray());
}

/// <summary>A model harness driver with a sweep mode (<c>sweep</c> command of <see cref="HarnessHost"/>).</summary>
internal interface ISweepDriver
{
    /// <summary>Hook flags of <see cref="SweepOptions.Hooks"/> this driver maps to an option that already exists.</summary>
    IReadOnlyCollection<string> WiredSweepHooks => [];

    /// <summary>Loads the model for the sweep; throws <see cref="BenchAssetsMissingException"/> when an asset is absent.</summary>
    ISweepTarget LoadSweep(HarnessContext context, SweepOptions options);
}

/// <summary>A model under the sweep: the product API on one loaded model.</summary>
internal interface ISweepTarget : IDisposable
{
    string Model { get; }

    string Profile { get; }

    string HarnessName { get; }

    string Provider { get; }

    int? IntraOpThreads { get; }

    /// <summary>The library's rows per ORT run in effect.</summary>
    int MaxBatchSize { get; }

    /// <summary>The library's padded-cell budget per ORT run in effect, or null when it has none.</summary>
    int? MaxBatchTokens { get; }

    /// <summary>Per-sequence token limit; longer pool items are dropped.</summary>
    int MaxTokens { get; }

    string[] Stages { get; }

    JsonObject Settings { get; }

    JsonObject LoadStages { get; }

    IReadOnlyList<string>? SessionProviders { get; }

    string QualityCheck { get; }

    double Tolerance { get; }

    string ToleranceFile { get; }

    JsonObject Artifacts(SweepWorkloadFile file);

    /// <summary>The harness's fixture quality sweep (null when it has none); must contain <c>pass</c>.</summary>
    JsonObject? PreliminaryQuality();

    /// <summary>Model tokens of an item as the call runs it (prefix, prompt or schema included).</summary>
    int CountTokens(SweepItem item);

    /// <summary>One call of the product API on <paramref name="items"/>; stage times (ms) into <paramref name="stages"/>.</summary>
    object Call(SweepItem[] items, double[] stages);

    /// <summary>The per-item reference kept from a call on <paramref name="item"/> alone.</summary>
    object Single(SweepItem item, object output);

    /// <summary>Rows per ORT run, in order, for a call with these token lengths (the library's chunk plan).</summary>
    IEnumerable<int> PlanChunks(IReadOnlyList<int> lengths);

    /// <summary>Every output of <paramref name="output"/> against the item's single-call reference.</summary>
    SweepCheck Compare(SweepItem[] items, object output, IReadOnlyDictionary<string, object> singles);
}

