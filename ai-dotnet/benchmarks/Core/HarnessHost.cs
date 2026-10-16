using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Starlings.Runtime.Onnx;

namespace Starlings.Benchmarks;

/// <summary>A required asset (model, tokenizer, fixture) is absent. The harness exits with code 3 and the reason.</summary>
internal sealed class BenchAssetsMissingException(string message) : Exception(message);

/// <summary>Settings shared by every model harness, parsed by <see cref="HarnessHost"/>.</summary>
internal sealed record HarnessContext(ArgMap Args, string WorkloadPath, int? IntraOpThreads, int? InterOpThreads, bool HashModelAtLoad, string Provider = "cpu")
{
    /// <summary>True for <c>--provider cuda</c> (only in the <c>.Cuda</c> builds).</summary>
    public bool IsCuda => Provider == "cuda";

    /// <summary>
    /// ONNX Runtime session options for these settings (the SHA-256 at load only with --hash-model true). CUDA: device 0
    /// of <c>CUDA_VISIBLE_DEVICES</c>, TF32 off (<see cref="OnnxExecutionProvider.Cuda"/> sets <c>use_tf32=0</c> and checks it).
    /// </summary>
    public OnnxSessionOptions SessionOptions() => new()
    {
        Provider = IsCuda ? OnnxExecutionProvider.Cuda : OnnxExecutionProvider.Cpu,
        CudaDeviceId = 0,
        IntraOpNumThreads = IntraOpThreads,
        InterOpNumThreads = InterOpThreads,
        ComputeModelSha256 = HashModelAtLoad,
        CpuMemoryArena = CpuArena,
    };

    /// <summary>--cpu-arena on|off: ONNX Runtime's CPU memory arena forced on or off (A/B runs); absent, the library default.</summary>
    public bool? CpuArena => Args.Get("cpu-arena", "") switch
    {
        "on" => true,
        "off" => false,
        "" => null,
        var v => throw new ArgumentException($"--cpu-arena is on or off, not '{v}'."),
    };

    /// <summary>
    /// The session settings block every run records (sequential, all optimizations); <paramref name="providerOptions"/>
    /// are the options the CUDA provider reports after configuration (the session diagnostics).
    /// </summary>
    public JsonObject SessionSettings(IReadOnlyDictionary<string, string>? providerOptions = null) => new()
    {
        ["provider"] = Provider,
        ["intra_op_num_threads"] = IntraOpThreads,
        ["inter_op_num_threads"] = InterOpThreads,
        ["execution_mode"] = "sequential",
        ["graph_optimization_level"] = "all",
        ["provider_options"] = IsCuda && providerOptions is not null
            ? new JsonObject { ["CUDAExecutionProvider"] = new JsonObject(providerOptions.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)kv.Value))) }
            : new JsonObject(),
        ["model_sha256_check_at_load"] = HashModelAtLoad,
        ["cpu_memory_arena"] = CpuArena switch { true => "on", false => "off", null => "library default" },
    };

    /// <summary>The registry profile of <paramref name="kind"/> (for example <c>onnx-fp32</c>) on this provider: <c>onnx-fp32-cpu</c> or <c>onnx-fp32-cuda</c>.</summary>
    public string ProfileOf(string kind) => $"{kind}-{Provider}";
}

/// <summary>One model harness: loads the model and fixtures and describes its workloads.</summary>
internal interface IBenchDriver
{
    /// <summary>Registry model id, for example <c>gliner2-base-v1</c>.</summary>
    string Model { get; }

    /// <summary>Registry profile on the context's provider, for example <c>onnx-fp32-cpu</c>.</summary>
    string Profile(HarnessContext context);

    string HarnessName { get; }

    /// <summary>Repository-relative path of the reference outputs the quality checks compare with.</summary>
    string Reference { get; }

    /// <summary>Default workload file name under <c>ai-dotnet/benchmarks/workloads/</c>.</summary>
    string WorkloadFile { get; }

    /// <summary>Usage lines of the model-specific options.</summary>
    string Usage { get; }

    /// <summary>False when the native runtime is driven by ONNX Runtime GenAI and the managed ORT environment must not be created.</summary>
    bool UsesManagedOrt => true;

    /// <summary>Loads the model; throws <see cref="BenchAssetsMissingException"/> when an asset is absent.</summary>
    IBenchRun Load(HarnessContext context);

    /// <summary>The artifact block of a run record (hashes the model files; called once per run record, not per process).</summary>
    JsonObject Artifacts(HarnessContext context);
}

/// <summary>A loaded model in one process.</summary>
internal interface IBenchRun : IDisposable
{
    /// <summary>Load stage times in ms (for example <c>tokenizer_load</c>, <c>session_create</c>), in order.</summary>
    JsonObject LoadStages { get; }

    JsonObject Settings { get; }

    IReadOnlyList<string>? SessionProviders { get; }

    /// <summary>The preliminary sweep over the reference cases: must contain <c>reference</c>, <c>cases</c> and <c>pass</c>.</summary>
    JsonObject PreliminaryQuality();

    IEnumerable<BenchSeries> WarmSeries();

    /// <summary>The single request of a cold start (request 0 is used).</summary>
    BenchSeries ColdSeries();
}

/// <summary>
/// One series: a workload at one path, batch and concurrency. <see cref="Call"/> runs request <c>r</c>, writes its stage
/// times and metrics, and returns the output; <see cref="Judge"/> rates outputs against the reference and returns a
/// quality object with <c>pass</c>.
/// </summary>
internal sealed class BenchSeries
{
    public required string Workload { get; init; }

    public required string Path { get; init; }

    public required string Input { get; init; }

    public required int Batch { get; init; }

    public int Concurrency { get; init; } = 1;

    public required int Samples { get; init; }

    public int WarmupCalls { get; init; } = 1;

    public required int Requests { get; init; }

    public required string[] Stages { get; init; }

    public string[] Metrics { get; init; } = [];

    public required Func<int, double[], double[], object> Call { get; init; }

    public required Func<IReadOnlyList<(int Request, object Output)>, JsonObject> Judge { get; init; }
}

/// <summary>
/// The warm, cold and cold-child commands for any <see cref="IBenchDriver"/>. Writes <c>starlings-bench-raw/1</c>
/// (model-pipeline/schemas/benchmark-raw.schema.json) like the e5 harness.
/// </summary>
internal static class HarnessHost
{
#if ORT_CUDA
    /// <summary>True in the <c>.Cuda</c> builds, which reference the CUDA natives and run only <c>--provider cuda</c>.</summary>
    public const bool CudaBuild = true;

    private const string NativePackage = "Microsoft.ML.OnnxRuntime.Gpu.Linux";
#else
    /// <summary>True in the <c>.Cuda</c> builds, which reference the CUDA natives and run only <c>--provider cuda</c>.</summary>
    public const bool CudaBuild = false;

    private const string NativePackage = "Microsoft.ML.OnnxRuntime";
#endif

    private static readonly string[] s_notForwarded = ["out", "processes", "note"];

    public static int Run(string[] args, IBenchDriver driver)
    {
        long mainTimestamp = Stopwatch.GetTimestamp();
        long mainNs = MonotonicClock.NowNs();
        string usage = Usage(driver);
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            Console.Error.WriteLine(usage);
            return args.Length == 0 ? 2 : 0;
        }

        try
        {
            var map = ArgMap.Parse(args.Skip(1).ToArray(), usage);
            string provider = ParseProvider(map.Get("provider", CudaBuild ? "cuda" : "cpu")!, driver.HarnessName);
            var context = new HarnessContext(
                map,
                Path.GetFullPath(map.Get("workloads") ?? BenchPaths.Workload(driver.WorkloadFile)),
                map.GetInt("threads"),
                map.GetInt("inter-threads", 1),
                map.Get("hash-model", "false") == "true",
                provider);
            if (args[0] is "warm" or "cold" && string.IsNullOrEmpty(map.Get("out")))
            {
                throw new ArgumentException("--out <raw.jsonl> is required.");
            }

            return args[0] switch
            {
                "warm" => Warm(driver, context),
                "cold" => Cold(driver, context),
                "cold-child" => ColdChild(driver, context, mainNs, mainTimestamp),
                _ => throw new ArgumentException($"Unknown command '{args[0]}'.\n{usage}"),
            };
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }
        catch (BenchAssetsMissingException ex)
        {
            Console.Error.WriteLine($"ASSETS MISSING ({driver.Model}): {ex.Message}");
            return 3;
        }
    }

    /// <summary>
    /// <c>cpu</c> or <c>cuda</c>, matching this build's native package: a CPU result never comes from the GPU natives and
    /// a CUDA run needs the <c>.Cuda</c> build and an explicit <c>CUDA_VISIBLE_DEVICES</c>.
    /// </summary>
    internal static string ParseProvider(string provider, string harnessName)
    {
        if (provider is not ("cpu" or "cuda"))
        {
            throw new ArgumentException("--provider cpu|cuda.");
        }

        if (provider == "cuda" && !CudaBuild)
        {
            throw new ArgumentException($"--provider cuda needs the CUDA build ({harnessName}.Cuda); this build references the CPU native runtime.");
        }

        if (provider == "cpu" && CudaBuild)
        {
            throw new ArgumentException($"This is the CUDA build; run CPU benchmarks with {harnessName} (the CPU native package).");
        }

        if (provider == "cuda" && string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")))
        {
            throw new ArgumentException("CUDA runs require an explicit device: set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES.");
        }

        return provider;
    }

    private static string Usage(IBenchDriver driver) => $"""
        usage: {driver.HarnessName} warm|cold --out <raw.jsonl> [--provider cpu|cuda] [options]
          --threads N               intra_op_num_threads (default: ORT default)    --inter-threads N (default 1)
          --workloads <file>        default ai-dotnet/benchmarks/workloads/{driver.WorkloadFile}
          --only <names>            warm: only these workloads   --samples-scale F   warm: scale every sample count
          --processes N             cold: fresh processes (default 10)
          --hash-model true|false   check the graph SHA-256 at load (default false)
          --cpu-arena on|off        force ONNX Runtime's CPU memory arena (A/B runs; default: the library's choice)
          --note <text>             machine note recorded in the environment block
        {driver.Usage}
        --provider: cpu in this CPU build, cuda in the CUDA build ({driver.HarnessName}.Cuda; CUDA_DEVICE_ORDER=PCI_BUS_ID
        and an explicit CUDA_VISIBLE_DEVICES; device 0 of it, TF32 off). Exit codes: 0 pass, 1 quality failed (timings invalid), 2 usage, 3 assets missing.
        """;

    private static int Warm(IBenchDriver driver, HarnessContext context)
    {
        var loadBefore = BenchEnvironment.LoadSnapshot();
        string started = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
        using var run = driver.Load(context);
        var quality = run.PreliminaryQuality();
        Console.Error.WriteLine($"preliminary quality: pass={quality["pass"]}");
        string runId = Guid.NewGuid().ToString();
        var only = context.Args.GetList("only")?.ToHashSet(StringComparer.Ordinal);
        double scale = context.Args.GetDouble("samples-scale", 1.0);
        var series = new List<JsonObject>();
        foreach (var spec in run.WarmSeries())
        {
            if (only is not null && !only.Contains(spec.Workload))
            {
                continue;
            }

            int n = Math.Max(1, (int)Math.Round(spec.Samples * scale));
            var s = Measure(spec, n);
            s["schema"] = BenchEnvironment.Schema;
            s["kind"] = "series";
            s["run_id"] = runId;
            s["mode"] = "warm";
            series.Add(s);
            double median = SeriesTimer.Median(s["ms"]!.AsArray().Select(x => x!.GetValue<double>()));
            Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{spec.Workload} {spec.Path} c={spec.Concurrency}: median {median:F3} ms (n={n}) quality pass={s["quality"]!["pass"]}"));
        }

        var record = Header(driver, context, "warm", runId, started);
        record["artifacts"] = driver.Artifacts(context);
        record["environment"] = BenchEnvironment.Environment(context.IsCuda, run.SessionProviders, loadBefore, context.Args.Get("note"), driver.UsesManagedOrt);
        record["settings"] = run.Settings;
        record["load_ms"] = run.LoadStages.DeepClone();
        record["quality"] = quality;
        bool pass = QualityGate.ApplyRecorded(quality, series);
        RawWriter.Write(context.Args.Get("out"), [record, .. series]);
        ReportFailures(quality);
        return pass ? 0 : 1;
    }

    private static JsonObject Measure(BenchSeries spec, int n)
    {
        var scratchStages = new double[Math.Max(4, spec.Stages.Length)];
        var scratchMetrics = new double[spec.Metrics.Length];
        for (int i = 0; i < spec.WarmupCalls; i++)
        {
            spec.Call(i % spec.Requests, scratchStages, scratchMetrics);
        }

        var timed = SeriesTimer.Time(n, spec.Concurrency, spec.Stages.Length, spec.Metrics.Length, (i, worker, stages, metrics) =>
            spec.Call((i + worker) % spec.Requests, stages, metrics));

        // After the timed loop: every request once more, rated against the reference.
        var outputs = Enumerable.Range(0, spec.Requests).Select(r => (r, spec.Call(r, scratchStages, scratchMetrics))).ToList();
        var series = new JsonObject
        {
            ["path"] = spec.Path,
            ["workload"] = spec.Workload,
            ["input"] = spec.Input,
            ["batch"] = spec.Batch,
            ["concurrency"] = spec.Concurrency,
            ["warmup_calls"] = spec.WarmupCalls,
            ["n"] = n,
            ["wall_ms"] = timed.WallMs,
            ["ms"] = timed.TotalsJson(),
            ["stages_ms"] = timed.StagesJson(spec.Stages),
        };
        if (spec.Metrics.Length > 0)
        {
            series["metrics"] = timed.MetricsJson(spec.Metrics);
        }

        series["quality"] = spec.Judge(outputs);
        return series;
    }

    private static int Cold(IBenchDriver driver, HarnessContext context)
    {
        var loadBefore = BenchEnvironment.LoadSnapshot();
        string started = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
        int processes = context.Args.GetInt("processes", 10)!.Value;
        var forwarded = context.Args.All.Where(kv => !s_notForwarded.Contains(kv.Key)).Select(kv => (kv.Key, kv.Value)).ToList();
        if (context.Args.Get("workloads") is null)
        {
            forwarded.Add(("workloads", context.WorkloadPath));
        }

        var (children, walls) = ColdRunner.Spawn(processes, forwarded);
        var first = children[0];
        var stages = ColdRunner.StageArrays(children, walls, ["quality", "settings", "series", "spawn_to_result"]);
        var qualities = children.Select(c => c["quality"]!.AsObject()).ToList();
        int failed = qualities.Count(q => !(q["pass"]?.GetValue<bool>() ?? false));
        string runId = Guid.NewGuid().ToString();
        var spec = first["series"]!.AsObject();
        var series = new JsonObject
        {
            ["schema"] = BenchEnvironment.Schema,
            ["kind"] = "series",
            ["run_id"] = runId,
            ["mode"] = "cold",
            ["path"] = spec["path"]!.DeepClone(),
            ["workload"] = "cold-start",
            ["input"] = spec["input"]!.DeepClone(),
            ["batch"] = spec["batch"]!.DeepClone(),
            ["concurrency"] = 1,
            ["warmup_calls"] = 0,
            ["n"] = children.Count,
            ["wall_ms"] = walls.Sum(),
            ["ms"] = new JsonArray(children.Select(c => (JsonNode)c["spawn_to_result"]!.GetValue<double>()).ToArray()),
            ["stages_ms"] = stages,
            ["quality"] = new JsonObject
            {
                ["pass"] = failed == 0,
                ["processes"] = children.Count,
                ["failed_processes"] = failed,
                ["first_process"] = qualities[0].DeepClone(),
            },
        };
        var record = Header(driver, context, "cold", runId, started);
        record["artifacts"] = driver.Artifacts(context);
        record["environment"] = BenchEnvironment.Environment(context.IsCuda, null, loadBefore, context.Args.Get("note"), driver.UsesManagedOrt);
        record["settings"] = first["settings"]!.DeepClone();
        var quality = new JsonObject
        {
            ["reference"] = driver.Reference,
            ["cases"] = spec["requests"]!.DeepClone(),
            ["cold_first_request"] = qualities[0].DeepClone(),
            ["pass"] = true,
        };
        record["quality"] = quality;
        bool pass = QualityGate.ApplyRecorded(quality, [series]);
        RawWriter.Write(context.Args.Get("out"), [record, series]);
        ReportFailures(quality);
        double median = SeriesTimer.Median(series["ms"]!.AsArray().Select(x => x!.GetValue<double>()));
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"cold-start: median spawn-to-result {median:F1} ms (n={children.Count})"));
        return pass ? 0 : 1;
    }

    private static int ColdChild(IBenchDriver driver, HarnessContext context, long mainNs, long mainTimestamp)
    {
        long spawnNs = ColdRunner.SpawnNs();
        using var run = driver.Load(context);
        var spec = run.ColdSeries();
        var stages = new double[Math.Max(4, spec.Stages.Length)];
        var metrics = new double[spec.Metrics.Length];
        var output = spec.Call(0, stages, metrics);
        double spawnToResult = (MonotonicClock.NowNs() - spawnNs) / 1e6;
        var result = new JsonObject { ["spawn_to_main"] = (mainNs - spawnNs) / 1e6 };
        foreach (var (name, value) in run.LoadStages)
        {
            result[name] = value!.DeepClone();
        }

        for (int i = 0; i < spec.Stages.Length; i++)
        {
            result[spec.Stages[i]] = stages[i];
        }

        result["spawn_to_result"] = spawnToResult;
        result["main_to_result"] = MonotonicClock.ElapsedMs(mainTimestamp);
        result["quality"] = spec.Judge([(0, output)]);
        result["settings"] = run.Settings;
        result["series"] = new JsonObject { ["path"] = spec.Path, ["input"] = spec.Input, ["batch"] = spec.Batch, ["requests"] = 1 };
        Console.WriteLine(result.ToJsonString());
        return 0;
    }

    private static JsonObject Header(IBenchDriver driver, HarnessContext context, string mode, string runId, string started) => new()
    {
        ["schema"] = BenchEnvironment.Schema,
        ["kind"] = "run",
        ["run_id"] = runId,
        ["started_utc"] = started,
        ["mode"] = mode,
        ["model"] = driver.Model,
        ["profile"] = driver.Profile(context),
        ["harness"] = new JsonObject
        {
            ["name"] = driver.HarnessName,
            ["language"] = "csharp",
            ["impl"] = "library",
            ["version"] = "1",
        },
    };

    /// <summary>The package block for a model harness: <see cref="BenchEnvironment.CommonPackages"/> plus <paramref name="extra"/>.</summary>
    public static JsonObject Packages(params (string Name, string? Version)[] extra)
    {
        var packages = BenchEnvironment.CommonPackages(NativePackage);
        foreach (var (name, version) in extra)
        {
            packages[name] = version;
        }

        return packages;
    }

    private static void ReportFailures(JsonObject quality)
    {
        if (quality["pass"]!.GetValue<bool>())
        {
            return;
        }

        Console.Error.WriteLine($"QUALITY FAILED: preliminary sweep pass={quality["preliminary_pass"]}, failed series: {quality["failed_series"]!.ToJsonString()}. Timings of this run are not valid results.");
    }
}
