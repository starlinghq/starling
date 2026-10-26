using System.Diagnostics;
using System.Text.Json.Nodes;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>
/// Qwen3.5-0.8B greedy generation (<c>ortgenai-fp32x-cpu</c>; the CUDA build: <c>ortgenai-fp32x-cuda</c> and, with
/// <c>--variant fp16x</c>, <c>ortgenai-fp16x-cuda</c>) through <see cref="Qwen35ChatModel.GenerateStreaming(Qwen35Prompt, Qwen35ChatOptions?, CancellationToken)"/>.
/// Stages: <c>template</c> (chat template and tokenizer, text path only; 0 on the ids path), <c>prefill</c> and
/// <c>decode</c> (the generator's own timings: prompt append, then every generated token including the stop token).
/// Metrics per sample: <c>ttft_ms</c> (call start to the first streamed update), token counts, prefill and decode
/// tokens per second, and the mean inter-token latency.
/// </summary>
internal sealed class Qwen35Driver : IBenchDriver
{
    public const string ReferencePath = "model-pipeline/protected/qwen3.5-0.8b/reference/generation.jsonl";

    public string Model => "qwen3.5-0.8b";

    public string Profile(HarnessContext context) => context.ProfileOf($"ortgenai-{VariantName(context)}");

    public string Reference => ReferencePath;

    public string HarnessName => "Starlings.Benchmarks.Qwen35Harness";

    public string WorkloadFile => "qwen35-v1.json";

    public bool UsesManagedOrt => false;

    public string Usage => $"""
          --variant fp32x|fp16x     the bundle (default fp32x; fp16x is built for CUDA, as Qwen35ChatModel.Load)
          model: ${Qwen35TestAssets.SnapshotEnvironmentVariable} (snapshot layout) or $STARLING_MODELS; never downloads. --inter-threads is ignored (GenAI session).
        """;

    public IBenchRun Load(HarnessContext context) => new Qwen35Run(context, Files(Variant(context)), Path.Combine(BenchPaths.RepositoryRoot, TolerancePath(context)));

    /// <summary>The bundle variant of <c>--variant</c>.</summary>
    internal static Qwen35Variant Variant(HarnessContext context)
    {
        var variant = context.Args.Get("variant", "fp32x") switch
        {
            "fp32x" => Qwen35Variant.Fp32x,
            "fp16x" => Qwen35Variant.Fp16x,
            _ => throw new ArgumentException("--variant fp32x|fp16x."),
        };
        return variant == Qwen35Variant.Fp16x && !context.IsCuda
            ? throw new ArgumentException("--variant fp16x is built for CUDA (as Qwen35ChatModel.Load); use the CUDA build.")
            : variant;
    }

    internal static string VariantName(HarnessContext context) => Variant(context) == Qwen35Variant.Fp16x ? "fp16x" : "fp32x";

    /// <summary>The profile's binding tolerances (its <c>discrete.greedy_ids</c> rule judges every generation).</summary>
    internal static string TolerancePath(HarnessContext context) => $"model-pipeline/protected/qwen3.5-0.8b/{context.ProfileOf($"ortgenai-{VariantName(context)}")}/tolerances.yaml";

    public JsonObject Artifacts(HarnessContext context)
    {
        var variant = Variant(context);
        var files = Files(variant);
        string graph = Path.Combine(files.ModelDirectory, "model.onnx");
        var artifacts = BenchEnvironment.Artifacts(
            graph,
            null,
            files.TokenizerPath,
            context.WorkloadPath,
            BenchEnvironment.FileDigests(Fixture(), Path.Combine(BenchPaths.RepositoryRoot, TolerancePath(context))),
            new JsonObject
            {
                ["Microsoft.ML.OnnxRuntimeGenAI.Managed"] = GenAIRuntimeInfo.ManagedVersion,
                ["GenAI native"] = GenAIRuntimeInfo.NativePackage,
                [OrtNativePackage] = BenchEnvironment.PackageVersion(OrtNativePackage),
                ["Starlings.Generation.OnnxGenAI"] = BenchEnvironment.InformationalVersion(typeof(TextGenerator).Assembly),
                ["Starlings.Models.Qwen35"] = BenchEnvironment.InformationalVersion(typeof(Qwen35ChatModel).Assembly),
                ["Starlings.Tokenizers"] = BenchEnvironment.TokenizerVersion,
            });
        artifacts["model_id"] = $"{Qwen35ModelAssets.Repository}@{Qwen35ModelAssets.Revision} ({Qwen35ModelAssets.VariantDirectory(variant)}, builder {Qwen35ModelAssets.BuilderVersion} at {Qwen35ModelAssets.BuilderSourceCommit})";
        artifacts["bundle_sha256"] = BenchEnvironment.FileDigests(
            Path.Combine(files.ModelDirectory, "model.onnx.data"),
            Path.Combine(files.ModelDirectory, "genai_config.json"),
            files.ChatTemplatePath);
        artifacts["tokenizer_version"] = BenchEnvironment.TokenizerVersion;
        artifacts["tokenizer_profile"] = Qwen35ModelAssets.TokenizerProfile;
        return artifacts;
    }

#if ORT_CUDA
    /// <summary>The ONNX Runtime natives under the GenAI natives: Microsoft.ML.OnnxRuntimeGenAI.Cuda brings the GPU build.</summary>
    private const string OrtNativePackage = "Microsoft.ML.OnnxRuntime.Gpu";
#else
    private const string OrtNativePackage = "Microsoft.ML.OnnxRuntime";
#endif

    internal static Qwen35ModelFiles Files(Qwen35Variant variant)
    {
        var (files, reason) = Qwen35TestAssets.FindModel(variant);
        return files ?? throw new BenchAssetsMissingException($"Qwen3.5 {Qwen35ModelAssets.VariantDirectory(variant)} bundle not found. " + reason);
    }

    internal static string Fixture() =>
        File.Exists(Qwen35TestAssets.FixturePath) ? Qwen35TestAssets.FixturePath : throw new BenchAssetsMissingException($"reference generation fixture not found at {Qwen35TestAssets.FixturePath}.");
}

/// <summary>What one generation produced.</summary>
internal sealed record Qwen35Output(GenerationCase Case, int MaxNewTokens, int[] PromptIds, GenerationResult Result);

internal sealed class Qwen35Run : IBenchRun
{
    private static readonly string[] s_stages = ["template", "prefill", "decode"];
    private static readonly string[] s_metrics = ["ttft_ms", "prompt_tokens", "completion_tokens", "prefill_tokens_per_s", "decode_tokens_per_s", "inter_token_ms"];
    private readonly HarnessContext _context;
    private readonly Qwen35ChatModel _model;
    private readonly IReadOnlyDictionary<string, GenerationCase> _cases;
    private readonly WorkloadSpec _spec;
    private readonly string _variant;
    private readonly string _tolerancePath;
    private readonly double? _nearTieMargin;

    public Qwen35Run(HarnessContext context, Qwen35ModelFiles files, string tolerancePath)
    {
        _context = context;
        _variant = Qwen35Driver.VariantName(context);
        _tolerancePath = BenchEnvironment.RepoRelative(tolerancePath);
        _nearTieMargin = NearTieMargin(tolerancePath);
        long t = Stopwatch.GetTimestamp();
        _spec = WorkloadSpec.Load(context.WorkloadPath);
        _cases = Qwen35TestAssets.ReadJsonLines(Qwen35Driver.Fixture()).Select(GenerationCase.Parse).ToDictionary(c => c.Id, StringComparer.Ordinal);
        double fixtureMs = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        _model = Qwen35ChatModel.Load(files, new Qwen35ModelOptions
        {
            Provider = context.IsCuda ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu,
            IntraOpThreads = context.IntraOpThreads,
        });
        LoadStages = new JsonObject { ["fixture_read"] = fixtureMs, ["model_load"] = MonotonicClock.ElapsedMs(t) };
    }

    public JsonObject LoadStages { get; }

    public JsonObject Settings => new()
    {
        ["provider"] = _context.Provider,
        ["variant"] = _variant,
        ["intra_op_num_threads"] = _context.IntraOpThreads,
        ["inter_op_num_threads"] = null,
        ["execution_mode"] = "onnxruntime-genai",
        ["graph_optimization_level"] = "genai_config.json",
        ["provider_options"] = new JsonObject(_model.Info.ProviderOptions.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)kv.Value))),
        ["genai_version"] = _model.Info.GenAIVersion,
        ["config_overlays"] = new JsonArray(_model.Info.ConfigOverlays.Select(o => (JsonNode)o).ToArray()),
        ["context_length"] = _model.Info.ContextLength,
        ["serializes_native_calls"] = _model.Info.SerializesNativeCalls,
        ["sampling"] = "greedy",
        ["prefill_chunk_tokens"] = _context.IsCuda ? Qwen35ChatOptions.DefaultCudaPrefillChunkTokens : Qwen35ChatOptions.DefaultCpuPrefillChunkTokens,
        ["greedy_ids_rule"] = GreedyRule,
        ["tolerance_file"] = _tolerancePath,
    };

    public IReadOnlyList<string>? SessionProviders => null;

    /// <summary>Every prompt on the ids path, greedy for 16 tokens, against the reference ids; and the text path's prompt ids.</summary>
    public JsonObject PreliminaryQuality()
    {
        var cases = _cases.Values.ToArray();
        var stages = new double[4];
        var metrics = new double[s_metrics.Length];
        var ids = Judge(cases.Select((c, i) => (i, (object)Generate(c, false, 16, stages, metrics))).ToList());
        int textCases = 0, textEqual = 0;
        foreach (var c in cases.Where(c => !c.HasTools))
        {
            textCases++;
            var prompt = _model.CreatePrompt(c.Messages!, c.EnableThinking);
            textEqual += prompt.Text == c.Rendered && prompt.TokenIds.SequenceEqual(c.PromptIds) ? 1 : 0;
        }

        return new JsonObject
        {
            ["reference"] = Qwen35Driver.ReferencePath,
            ["cases"] = cases.Length,
            ["greedy_ids_16_tokens"] = ids,
            ["template_cases"] = textCases,
            ["template_exact"] = textEqual,
            ["pass"] = ids["pass"]!.GetValue<bool>() && textEqual == textCases,
        };
    }

    public IEnumerable<BenchSeries> WarmSeries()
    {
        foreach (var w in _spec.Workloads)
        {
            var requests = WorkloadSpec.Strings(w.Raw.GetProperty("requests")).Select(Case).ToArray();
            int max = w.Raw.GetProperty("max_new_tokens").GetInt32();
            foreach (int c in w.Concurrency)
            {
                yield return Series(w.Name, w.Path, w.Input, c, w.Samples, w.WarmupCalls ?? _spec.WarmupCalls, requests, max);
            }
        }
    }

    public BenchSeries ColdSeries()
    {
        var requests = WorkloadSpec.Strings(_spec.Cold.GetProperty("request")).Select(Case).ToArray();
        return Series("cold-start", "ids", "short prompt", 1, 1, 0, requests, _spec.Cold.GetProperty("max_new_tokens").GetInt32());
    }

    public void Dispose() => _model.Dispose();

    private GenerationCase Case(string id) => _cases.TryGetValue(id, out var c) ? c : throw new ArgumentException($"Workload names unknown prompt '{id}'.");

    private BenchSeries Series(string name, string path, string input, int concurrency, int samples, int warmup, GenerationCase[] requests, int maxNewTokens)
    {
        bool text = path == "text";
        if (text && requests.Any(c => c.HasTools))
        {
            throw new ArgumentException($"{name}: prompts with tools run on the ids path only.");
        }

        return new BenchSeries
        {
            Workload = name,
            Path = path,
            Input = input,
            Batch = 1,
            Concurrency = concurrency,
            Samples = samples,
            WarmupCalls = warmup,
            Requests = requests.Length,
            Stages = s_stages,
            Metrics = s_metrics,
            Call = (r, stages, metrics) => Generate(requests[r], text, maxNewTokens, stages, metrics),
            Judge = Judge,
        };
    }

    private Qwen35Output Generate(GenerationCase c, bool text, int maxNewTokens, double[] stages, double[] metrics)
    {
        long t0 = Stopwatch.GetTimestamp();
        var prompt = text ? _model.CreatePrompt(c.Messages!, c.EnableThinking) : Qwen35Prompt.FromTokenIds(c.PromptIds);
        double template = text ? MonotonicClock.ElapsedMs(t0) : 0;
        double ttft = -1;
        GenerationResult? result = null;
        foreach (var update in _model.GenerateStreaming(prompt, new Qwen35ChatOptions { MaxNewTokens = maxNewTokens }))
        {
            if (ttft < 0)
            {
                ttft = MonotonicClock.ElapsedMs(t0);
            }

            result = update.Result ?? result;
        }

        result = result ?? throw new InvalidOperationException("The stream ended without a result.");
        double prefill = result.Timings.Prefill.TotalMilliseconds;
        double decode = result.Timings.Decode.TotalMilliseconds;
        int steps = result.Usage.CompletionTokens + (result.StopTokenId is null ? 0 : 1);
        stages[0] = template;
        stages[1] = prefill;
        stages[2] = decode;
        metrics[0] = ttft;
        metrics[1] = result.Usage.PromptTokens;
        metrics[2] = result.Usage.CompletionTokens;
        metrics[3] = result.Usage.PromptTokens / (prefill / 1000);
        metrics[4] = steps == 0 ? 0 : steps / (decode / 1000);
        metrics[5] = steps == 0 ? 0 : decode / steps;
        return new Qwen35Output(c, maxNewTokens, [.. prompt.TokenIds], result);
    }

    private string GreedyRule => _nearTieMargin is double m
        ? string.Create(System.Globalization.CultureInfo.InvariantCulture, $"greedy ids equal to the reference up to the first divergence, which must fall where the reference top-1 margin is below {m} (discrete.greedy_ids of {_tolerancePath})")
        : "greedy ids equal to the reference up to max_new_tokens (exact)";

    /// <summary>
    /// The margin of the profile's <c>discrete.greedy_ids: equal_until_first_divergence_at_margin_below_X</c> rule
    /// (<c>ortgenai-fp16x-cuda</c>), or null for the exact rule of the fp32x profiles.
    /// </summary>
    internal static double? NearTieMargin(string tolerancePath)
    {
        const string prefix = "equal_until_first_divergence_at_margin_below_";
        foreach (string raw in File.ReadAllLines(tolerancePath))
        {
            string line = raw.Split('#')[0].Trim();
            if (line.StartsWith("greedy_ids:", StringComparison.Ordinal))
            {
                string rule = line["greedy_ids:".Length..].Trim();
                return rule.StartsWith(prefix, StringComparison.Ordinal)
                    ? double.Parse(rule[prefix.Length..], System.Globalization.CultureInfo.InvariantCulture)
                    : null;
            }
        }

        throw new InvalidDataException($"{tolerancePath} has no discrete.greedy_ids rule.");
    }

    /// <summary>
    /// Generated ids (with the stop token) against the reference output up to max_new_tokens, and the prompt ids equal
    /// to the reference prompt ids. fp32x profiles: exact (a divergence at a near tie also fails, with its reference
    /// margin reported). fp16x: the protected rule, equal up to the first divergence, which must fall where the
    /// reference top-1 margin is below the profile's bound (as the CUDA tests' <c>GreedyIdsDivergeOnlyAtNearTies</c>).
    /// </summary>
    private JsonObject Judge(IReadOnlyList<(int Request, object Output)> outputs)
    {
        int nearTies = 0;
        int generations = 0, equal = 0;
        var failures = new JsonArray();
        foreach (var (_, o) in outputs)
        {
            var x = (Qwen35Output)o;
            generations++;
            var c = x.Case;
            int[] expected = c.OutputIds.Length > x.MaxNewTokens ? c.OutputIds[..x.MaxNewTokens] : c.OutputIds;
            int[] actual = x.Result.StopTokenId is int stop ? [.. x.Result.TokenIds, stop] : [.. x.Result.TokenIds];
            bool promptEqual = x.PromptIds.SequenceEqual(c.PromptIds);
            if (promptEqual && expected.SequenceEqual(actual))
            {
                equal++;
                continue;
            }

            int n = 0;
            while (n < expected.Length && n < actual.Length && expected[n] == actual[n])
            {
                n++;
            }

            if (promptEqual && _nearTieMargin is double bound && (n >= expected.Length || (n < c.StepMargin.Length && c.StepMargin[n] < bound)))
            {
                // Diverges only at a near tie (or runs past a reference that stopped): allowed by the fp16x rule.
                equal++;
                nearTies++;
                continue;
            }

            string margin = n < c.StepMargin.Length ? c.StepMargin[n].ToString("R", System.Globalization.CultureInfo.InvariantCulture) : "n/a";
            failures.Add((JsonNode)(promptEqual ? $"{c.Id}: diverges at step {n} (reference top-1 margin {margin})" : $"{c.Id}: prompt ids differ"));
        }

        return new JsonObject
        {
            ["check"] = GreedyRule,
            ["generations"] = generations,
            ["equal"] = equal,
            ["diverged_at_near_tie"] = nearTies,
            ["failures"] = failures,
            ["pass"] = equal == generations,
        };
    }
}
