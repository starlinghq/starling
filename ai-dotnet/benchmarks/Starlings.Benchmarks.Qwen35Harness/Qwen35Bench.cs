using System.Diagnostics;
using System.Text.Json.Nodes;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>
/// Qwen3.5-0.8B greedy generation (<c>ortgenai-fp32x-cpu</c>) through <see cref="Qwen35ChatModel.GenerateStreaming(Qwen35Prompt, Qwen35ChatOptions?, CancellationToken)"/>.
/// Stages: <c>template</c> (chat template and tokenizer, text path only; 0 on the ids path), <c>prefill</c> and
/// <c>decode</c> (the generator's own timings: prompt append, then every generated token including the stop token).
/// Metrics per sample: <c>ttft_ms</c> (call start to the first streamed update), token counts, prefill and decode
/// tokens per second, and the mean inter-token latency.
/// </summary>
internal sealed class Qwen35Driver : IBenchDriver
{
    public const string ReferencePath = "model-pipeline/protected/qwen3.5-0.8b/reference/generation.jsonl";

    public string Model => "qwen3.5-0.8b";

    public string Profile => "ortgenai-fp32x-cpu";

    public string Reference => ReferencePath;

    public string HarnessName => "Starlings.Benchmarks.Qwen35Harness";

    public string WorkloadFile => "qwen35-v1.json";

    public bool UsesManagedOrt => false;

    public string Usage => $"  model: ${Qwen35TestAssets.SnapshotEnvironmentVariable} (snapshot layout) or $STARLING_MODELS; never downloads. --inter-threads is ignored (GenAI session).";

    public IBenchRun Load(HarnessContext context) => new Qwen35Run(context, Files());

    public JsonObject Artifacts(HarnessContext context)
    {
        var files = Files();
        string graph = Path.Combine(files.ModelDirectory, "model.onnx");
        var artifacts = BenchEnvironment.Artifacts(
            graph,
            null,
            files.TokenizerPath,
            context.WorkloadPath,
            BenchEnvironment.FileDigests(Fixture(), Path.Combine(BenchPaths.RepositoryRoot, "model-pipeline", "protected", "qwen3.5-0.8b", "ortgenai-fp32x-cpu", "tolerances.yaml")),
            new JsonObject
            {
                ["Microsoft.ML.OnnxRuntimeGenAI.Managed"] = GenAIRuntimeInfo.ManagedVersion,
                ["GenAI native"] = GenAIRuntimeInfo.NativePackage,
                ["Microsoft.ML.OnnxRuntime"] = BenchEnvironment.PackageVersion("Microsoft.ML.OnnxRuntime"),
                ["Starlings.Generation.OnnxGenAI"] = BenchEnvironment.InformationalVersion(typeof(TextGenerator).Assembly),
                ["Starlings.Models.Qwen35"] = BenchEnvironment.InformationalVersion(typeof(Qwen35ChatModel).Assembly),
                ["Starlings.Tokenizers"] = BenchEnvironment.TokenizerVersion,
            });
        artifacts["model_id"] = $"{Qwen35ModelAssets.Repository}@{Qwen35ModelAssets.Revision} ({Qwen35ModelAssets.VariantDirectory(Qwen35Variant.Fp32x)}, builder {Qwen35ModelAssets.BuilderVersion} at {Qwen35ModelAssets.BuilderSourceCommit})";
        artifacts["bundle_sha256"] = BenchEnvironment.FileDigests(
            Path.Combine(files.ModelDirectory, "model.onnx.data"),
            Path.Combine(files.ModelDirectory, "genai_config.json"),
            files.ChatTemplatePath);
        artifacts["tokenizer_version"] = BenchEnvironment.TokenizerVersion;
        artifacts["tokenizer_profile"] = Qwen35ModelAssets.TokenizerProfile;
        return artifacts;
    }

    internal static Qwen35ModelFiles Files()
    {
        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Fp32x);
        return files ?? throw new BenchAssetsMissingException("Qwen3.5 fp32x bundle not found. " + reason);
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

    public Qwen35Run(HarnessContext context, Qwen35ModelFiles files)
    {
        _context = context;
        long t = Stopwatch.GetTimestamp();
        _spec = WorkloadSpec.Load(context.WorkloadPath);
        _cases = Qwen35TestAssets.ReadJsonLines(Qwen35Driver.Fixture()).Select(GenerationCase.Parse).ToDictionary(c => c.Id, StringComparer.Ordinal);
        double fixtureMs = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        _model = Qwen35ChatModel.Load(files, new Qwen35ModelOptions { Provider = GenAIExecutionProvider.Cpu, IntraOpThreads = context.IntraOpThreads });
        LoadStages = new JsonObject { ["fixture_read"] = fixtureMs, ["model_load"] = MonotonicClock.ElapsedMs(t) };
    }

    public JsonObject LoadStages { get; }

    public JsonObject Settings => new()
    {
        ["provider"] = "cpu",
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

    /// <summary>
    /// Generated ids (with the stop token) equal to the reference output up to max_new_tokens, and the prompt ids equal
    /// to the reference prompt ids. Strict: a divergence at a near tie also fails, with its reference margin reported.
    /// </summary>
    private static JsonObject Judge(IReadOnlyList<(int Request, object Output)> outputs)
    {
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

            string margin = n < c.StepMargin.Length ? c.StepMargin[n].ToString("R", System.Globalization.CultureInfo.InvariantCulture) : "n/a";
            failures.Add((JsonNode)(promptEqual ? $"{c.Id}: diverges at step {n} (reference top-1 margin {margin})" : $"{c.Id}: prompt ids differ"));
        }

        return new JsonObject
        {
            ["check"] = "greedy ids equal to the reference up to max_new_tokens (exact)",
            ["generations"] = generations,
            ["equal"] = equal,
            ["failures"] = failures,
            ["pass"] = equal == generations,
        };
    }
}
