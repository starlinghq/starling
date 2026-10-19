using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Benchmarks;
using Starlings.Runtime.Onnx;

// Realistic-workload harness for e5-small (P-008, plan section 11). Writes starlings-bench-raw/1 JSON lines; see
// ai-dotnet/benchmarks/README.md and model-pipeline/schemas/benchmark-raw.schema.json.
return ToolMain.Run(ToolBody);

int ToolBody()
{
    long mainTimestamp = Stopwatch.GetTimestamp();
    long mainNs = MonotonicClock.NowNs();

    if (args.Length == 0 || args[0] is "-h" or "--help")
    {
        Console.Error.WriteLine(Harness.Usage);
        return args.Length == 0 ? 2 : 0;
    }

    if (args[0] == "sweep")
    {
#if ORT_CUDA
        const bool cudaBuild = true;
#else
        const bool cudaBuild = false;
#endif
        return BatchSweep.Command(args[1..], "Starlings.Benchmarks.E5Harness", cudaBuild, (context, sweep) => new E5SweepTarget(context, sweep), SweepOptions.AllHooks);
    }

    try
    {
        var options = HarnessOptions.Parse(args.Skip(1).ToArray());
        if (args[0] is "warm" or "cold" && string.IsNullOrEmpty(options.Out))
        {
            throw new ArgumentException("--out <raw.jsonl> is required.");
        }

        return args[0] switch
        {
            "warm" => Harness.Warm(options),
            "cold" => Harness.Cold(options),
            "cold-child" => Harness.ColdChild(options, mainNs, mainTimestamp),
            _ => throw new ArgumentException($"Unknown command '{args[0]}'.\n{Harness.Usage}"),
        };
    }
    catch (ArgumentException ex)
    {
        Console.Error.WriteLine(ex.Message);
        return 2;
    }
}

/// <summary>Command-line options shared by the commands.</summary>
internal sealed record HarnessOptions
{
    public required PipelineSettings Settings { get; init; }

    public required string Impl { get; init; }

    public string? Snapshot { get; init; }

    public string? Workloads { get; init; }

    public string? Out { get; init; }

    public string? Note { get; init; }

    public string[] Paths { get; init; } = ["ids", "text"];

    public int[] Concurrency { get; init; } = [1, 4];

    public HashSet<string>? Only { get; init; }

    public double SamplesScale { get; init; } = 1.0;

    public int Processes { get; init; } = 20;

    public static HarnessOptions Parse(string[] args)
    {
        var map = ArgMap.Parse(args, Harness.Usage);
        string Get(string key, string fallback) => map.Get(key, fallback);
        int? GetInt(string key, int? fallback) => map.GetInt(key, fallback);
        var provider = Get("provider", string.Empty) switch
        {
            "cpu" => OnnxExecutionProvider.Cpu,
            "cuda" => OnnxExecutionProvider.Cuda,
            _ => throw new ArgumentException("--provider cpu|cuda is required."),
        };
        if (provider == OnnxExecutionProvider.Cuda && string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")))
        {
            throw new ArgumentException("CUDA runs require an explicit device: set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES.");
        }

        string impl = Get("impl", "library");
        if (impl is not ("library" or "direct"))
        {
            throw new ArgumentException("--impl library|direct.");
        }

        return new HarnessOptions
        {
            Settings = new PipelineSettings(provider, GetInt("threads", null), GetInt("inter-threads", 1), GetInt("cuda-device", 0)!.Value, Get("hash-model", "false") == "true"),
            Impl = impl,
            Snapshot = map.Get("snapshot") ?? Environment.GetEnvironmentVariable("STARLING_E5_SNAPSHOT"),
            Workloads = map.Get("workloads"),
            Out = map.Get("out"),
            Note = map.Get("note"),
            Paths = Get("paths", "ids,text").Split(','),
            Concurrency = Get("concurrency", "1,4").Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray(),
            Only = map.GetList("only")?.ToHashSet(StringComparer.Ordinal),
            SamplesScale = double.Parse(Get("samples-scale", "1"), CultureInfo.InvariantCulture),
            Processes = GetInt("processes", 20)!.Value,
        };
    }
}

/// <summary>The warm, cold and cold-child commands.</summary>
internal static class Harness
{
    public const string Usage = """
        usage: E5Harness warm|cold --provider cpu|cuda --out <raw.jsonl> [options]
               E5Harness sweep --out <sweep.jsonl> [options]   batching sweep (sweep --help; --snapshot as below)
          --impl library|direct     library: E5Tokenizer + E5Encoder public API (default); direct: raw InferenceSession
          --threads N               intra_op_num_threads (default: ORT default)    --inter-threads N (default 1)
          --snapshot <dir>          e5 snapshot (default: $STARLING_E5_SNAPSHOT, else the acquisition cache)
          --workloads <file>        default ai-dotnet/benchmarks/workloads/e5-v1.json
          --paths ids,text  --concurrency 1,4  --only <names>  --samples-scale F   (warm)
          --processes N             fresh processes for cold (default 20)
          --hash-model true|false   hash the model at load (default false, as the Python runner)
          --note <text>             machine note recorded in the environment block
        CUDA needs an explicit device: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>, and the CUDA build
        (Starlings.Benchmarks.E5Harness.Cuda).
        """;

    public static int Warm(HarnessOptions o)
    {
        var loadBefore = BenchEnvironment.LoadSnapshot();
        string started = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
        var spec = BenchWorkloadSpec.Load(o.Workloads);
        var files = BenchFixtures.Files(o.Snapshot);
        using var pipeline = Create(o, files);
        var quality = RunQuality(pipeline, o.Settings.Provider);
        string runId = Guid.NewGuid().ToString();
        string providerKey = o.Settings.IsCuda ? "cuda" : "cpu";
        var series = new List<JsonObject>();
        foreach (var w in spec.Workloads)
        {
            if (o.Only is not null && !o.Only.Contains(w.Name))
            {
                continue;
            }

            foreach (string path in o.Paths)
            {
                foreach (int c in w.Concurrency.Where(o.Concurrency.Contains))
                {
                    int n = Math.Max(1, (int)Math.Round(w.Samples[providerKey] * o.SamplesScale));
                    var s = MeasureSeries(pipeline, w, path == "text", c, n, spec.WarmupCalls);
                    s["schema"] = BenchEnvironment.Schema;
                    s["kind"] = "series";
                    s["run_id"] = runId;
                    s["mode"] = "warm";
                    series.Add(s);
                    var ms = s["ms"]!.AsArray().Select(x => x!.GetValue<double>()).Order().ToArray();
                    Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{w.Name} {path} c={c}: median {ms[ms.Length / 2]:F3} ms (n={n})"));
                }
            }
        }

        var run = Header(o, "warm", runId, started);
        run["artifacts"] = BenchFixtures.Artifacts(files, pipeline.ModelSha256, spec.Path, NativePackage);
        run["environment"] = BenchEnvironment.Environment(o.Settings.IsCuda, pipeline.SessionProviders, loadBefore, o.Note);
        run["settings"] = pipeline.Settings;
        run["quality"] = quality;
        bool pass = QualityGate.Apply(quality, series, E5Tolerance(o.Settings.Provider));
        RawWriter.Write(o.Out, [run, .. series]);
        ReportFailures(quality);
        return pass ? 0 : 1;
    }

    public static int Cold(HarnessOptions o)
    {
        var loadBefore = BenchEnvironment.LoadSnapshot();
        string started = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
        var spec = BenchWorkloadSpec.Load(o.Workloads);
        var files = BenchFixtures.Files(o.Snapshot);
        var (children, walls) = ColdRunner.Spawn(o.Processes, ChildArguments(o, files, spec));
        var stages = ColdRunner.StageArrays(children, walls, ["quality", "settings", "spawn_to_result"]);
        var q = children.Select(c => new Quality(c["quality"]!["max_abs"]!.GetValue<double>(), c["quality"]!["min_cosine"]!.GetValue<double>(), c["quality"]!["finite"]!.GetValue<bool>()))
            .Aggregate((a, b) => a.Merge(b));
        string runId = Guid.NewGuid().ToString();
        var series = new JsonObject
        {
            ["schema"] = BenchEnvironment.Schema,
            ["kind"] = "series",
            ["run_id"] = runId,
            ["mode"] = "cold",
            ["path"] = "text",
            ["workload"] = "cold-start",
            ["input"] = "short",
            ["batch"] = spec.ColdRequest.Length,
            ["concurrency"] = 1,
            ["warmup_calls"] = 0,
            ["n"] = children.Count,
            ["wall_ms"] = walls.Sum(),
            ["ms"] = new JsonArray(children.Select(c => (JsonNode)c["spawn_to_result"]!.GetValue<double>()).ToArray()),
            ["stages_ms"] = stages,
            ["quality"] = q.ToJson(),
        };
        double tolerance = E5Tolerance(o.Settings.Provider);
        var run = Header(o, "cold", runId, started);
        run["artifacts"] = BenchFixtures.Artifacts(files, null, spec.Path, NativePackage);
        run["environment"] = BenchEnvironment.Environment(o.Settings.IsCuda, null, loadBefore, o.Note);
        run["settings"] = children[0]["settings"]!.DeepClone();
        run["quality"] = new JsonObject
        {
            ["reference"] = ReferencePath,
            ["tolerance_file"] = TolerancePath(o.Settings.Provider),
            ["embedding_max_abs_tolerance"] = tolerance,
            ["cases"] = spec.ColdRequest.Length,
            ["cold_first_request"] = q.ToJson(),
            ["pass"] = true,
        };
        bool pass = QualityGate.Apply(run["quality"]!.AsObject(), [series], tolerance);
        RawWriter.Write(o.Out, [run, series]);
        ReportFailures(run["quality"]!.AsObject());
        var sorted = series["ms"]!.AsArray().Select(x => x!.GetValue<double>()).Order().ToArray();
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"cold-start: median spawn-to-result {sorted[sorted.Length / 2]:F1} ms (n={sorted.Length})"));
        return pass ? 0 : 1;
    }

    private static void ReportFailures(JsonObject quality)
    {
        if (quality["pass"]!.GetValue<bool>())
        {
            return;
        }

        Console.Error.WriteLine($"QUALITY FAILED: preliminary sweep pass={quality["preliminary_pass"]}, failed series: {quality["failed_series"]!.ToJsonString()}. Timings of this run are not valid results.");
    }

    public static int ColdChild(HarnessOptions o, long mainNs, long mainTimestamp)
    {
        long spawnNs = ColdRunner.SpawnNs();
        long tFixtures = Stopwatch.GetTimestamp();
        var spec = BenchWorkloadSpec.Load(o.Workloads);
        var files = BenchFixtures.Files(o.Snapshot);
        double fixtureMs = MonotonicClock.ElapsedMs(tFixtures);
        using var pipeline = Create(o, files);
        var stages = new double[4];
        var output = pipeline.Embed(spec.ColdRequest, text: true, stages);
        double spawnToResult = (MonotonicClock.NowNs() - spawnNs) / 1e6;
        var result = new JsonObject
        {
            ["spawn_to_main"] = (mainNs - spawnNs) / 1e6,
            ["tokenizer_load"] = pipeline.TokenizerLoadMs,
            ["session_create"] = pipeline.SessionCreateMs,
        };
        var names = pipeline.StageNames(text: true);
        for (int i = 0; i < names.Length; i++)
        {
            result[names[i]] = stages[i];
        }

        result["spawn_to_result"] = spawnToResult;
        result["fixture_read"] = fixtureMs;
        result["main_to_result"] = MonotonicClock.ElapsedMs(mainTimestamp);
        result["quality"] = Quality.Of(output, BenchFixtures.Embeddings(spec.ColdRequest)).ToJson();
        result["settings"] = pipeline.Settings;
        Console.WriteLine(result.ToJsonString());
        return 0;
    }

#if ORT_CUDA
    private const string NativePackage = "Microsoft.ML.OnnxRuntime.Gpu.Linux";
#else
    private const string NativePackage = "Microsoft.ML.OnnxRuntime";
#endif

    private const string ReferencePath = "model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl";

    private static string TolerancePath(OnnxExecutionProvider provider) =>
        $"model-pipeline/protected/multilingual-e5-small/{(provider == OnnxExecutionProvider.Cuda ? "onnx-fp32-cuda" : "onnx-fp32-cpu")}/tolerances.yaml";

    private static double E5Tolerance(OnnxExecutionProvider provider) =>
        Starlings.TestSupport.E5Tolerances.Load(BenchFixtures.FixtureDirectory(provider)).EmbeddingMaxAbs;

    private static IEmbedPipeline Create(HarnessOptions o, Starlings.Models.E5.E5ModelFiles files) =>
        o.Impl == "direct" ? new DirectPipeline(files, o.Settings) : new LibraryPipeline(files, o.Settings);

    private static IEnumerable<(string, string)> ChildArguments(HarnessOptions o, Starlings.Models.E5.E5ModelFiles files, BenchWorkloadSpec spec)
    {
        yield return ("provider", o.Settings.IsCuda ? "cuda" : "cpu");
        yield return ("impl", o.Impl);
        yield return ("snapshot", Path.GetDirectoryName(Path.GetDirectoryName(files.ModelPath))!);
        yield return ("workloads", spec.Path);
        yield return ("cuda-device", o.Settings.CudaDeviceId.ToString(CultureInfo.InvariantCulture));
        yield return ("hash-model", o.Settings.HashModelAtLoad ? "true" : "false");
        if (o.Settings.IntraOpThreads is int intra)
        {
            yield return ("threads", intra.ToString(CultureInfo.InvariantCulture));
        }

        if (o.Settings.InterOpThreads is int inter)
        {
            yield return ("inter-threads", inter.ToString(CultureInfo.InvariantCulture));
        }
    }

    private static JsonObject Header(HarnessOptions o, string mode, string runId, string started) => new()
    {
        ["schema"] = BenchEnvironment.Schema,
        ["kind"] = "run",
        ["run_id"] = runId,
        ["started_utc"] = started,
        ["mode"] = mode,
        ["model"] = "multilingual-e5-small",
        ["harness"] = new JsonObject
        {
            ["name"] = "Starlings.Benchmarks.E5Harness",
            ["language"] = "csharp",
            ["impl"] = o.Impl,
            ["version"] = "1",
        },
    };

    private static JsonObject RunQuality(IEmbedPipeline pipeline, OnnxExecutionProvider provider)
    {
        var cases = BenchFixtures.Ordered;
        double tolerance = E5Tolerance(provider);
        Span<double> stages = stackalloc double[4];
        var result = new JsonObject
        {
            ["reference"] = ReferencePath,
            ["tolerance_file"] = TolerancePath(provider),
            ["embedding_max_abs_tolerance"] = tolerance,
            ["cases"] = cases.Count,
        };
        bool pass = true;
        foreach (bool text in new[] { false, true })
        {
            var q = Quality.Empty;
            foreach (var c in cases)
            {
                q = q.Merge(Quality.Of(pipeline.Embed([c], text, stages), [c.Embedding]));
            }

            result[text ? "text_single" : "ids_single"] = q.ToJson();
            pass &= q.MaxAbs <= tolerance && q.Finite;
        }

        var b = Quality.Empty;
        for (int start = 0; start < cases.Count; start += 16)
        {
            var chunk = cases.Skip(start).Take(16).ToArray();
            b = b.Merge(Quality.Of(pipeline.Embed(chunk, false, stages), BenchFixtures.Embeddings(chunk)));
        }

        result["ids_batch16"] = b.ToJson();
        result["pass"] = pass && b.MaxAbs <= tolerance && b.Finite;
        return result;
    }

    private static JsonObject MeasureSeries(IEmbedPipeline pipeline, BenchWorkload w, bool text, int concurrency, int n, int warmup)
    {
        var requests = w.Requests;
        int warmupCalls = Math.Max(warmup, requests.Length);
        var scratch = new double[4];
        for (int i = 0; i < warmupCalls; i++)
        {
            pipeline.Embed(requests[i % requests.Length], text, scratch);
        }

        var names = pipeline.StageNames(text);
        var timed = SeriesTimer.Time(n, concurrency, names.Length, 0, (i, worker, stages, _) =>
            pipeline.Embed(requests[(i + worker) % requests.Length], text, stages));
        var q = Quality.Empty;
        foreach (var request in requests)
        {
            q = q.Merge(Quality.Of(pipeline.Embed(request, text, scratch), BenchFixtures.Embeddings(request)));
        }

        return new JsonObject
        {
            ["path"] = text ? "text" : "ids",
            ["workload"] = w.Name,
            ["input"] = w.Input,
            ["batch"] = w.Batch,
            ["concurrency"] = concurrency,
            ["warmup_calls"] = warmupCalls,
            ["n"] = n,
            ["wall_ms"] = timed.WallMs,
            ["ms"] = timed.TotalsJson(),
            ["stages_ms"] = timed.StagesJson(names),
            ["quality"] = q.ToJson(),
        };
    }
}
