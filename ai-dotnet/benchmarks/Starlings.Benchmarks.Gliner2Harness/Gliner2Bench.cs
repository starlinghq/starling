using System.Diagnostics;
using System.Text.Json.Nodes;
using Starlings.Models.Gliner2;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>
/// GLiNER2 entity extraction (<c>gliner2-base-v1</c>, <c>onnx-fp32-cpu</c>) through the public API, with the three
/// stages timed separately: <c>prepare</c> (normalization, word splitting, tokenization, schema and index tensors:
/// <see cref="Gliner2EntityExtractor.Prepare"/>), <c>inference</c> (padding and the ORT run:
/// <see cref="Gliner2EntityExtractor.RunModel"/>) and <c>decode</c> (span decoding to entities:
/// <see cref="Gliner2EntityExtractor.Decode"/>).
/// </summary>
internal sealed class Gliner2Driver : IBenchDriver
{
    public const string ReferencePath = "model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/extraction.jsonl";

    public string Model => "gliner2-base-v1";

    public string Reference => ReferencePath;

    public string Profile => "onnx-fp32-cpu";

    public string HarnessName => "Starlings.Benchmarks.Gliner2Harness";

    public string WorkloadFile => "gliner2-v1.json";

    public string Usage => $"  model: ${Gliner2TestAssets.SnapshotEnvironmentVariable} or $STARLING_MODELS (acquisition cache); never downloads.";

    public IBenchRun Load(HarnessContext context) => new Gliner2Run(context, Files());

    public JsonObject Artifacts(HarnessContext context)
    {
        var files = Files();
        var artifacts = BenchEnvironment.Artifacts(
            files.ModelPath,
            null,
            files.TokenizerPath,
            context.WorkloadPath,
            BenchEnvironment.FileDigests(FixturePath(), Path.Combine(Gliner2TestAssets.ExtractionFixtureDirectory, "tolerances.yaml")),
            HarnessHost.Packages(("Starlings.Models.Gliner2", BenchEnvironment.InformationalVersion(typeof(Gliner2EntityExtractor).Assembly))));
        artifacts["model_id"] = $"{Gliner2ModelAssets.Repository}@{Gliner2ModelAssets.Revision} ({Gliner2ModelAssets.ModelFile}, source {Gliner2ModelAssets.SourceRevision})";
        artifacts["expected_model_sha256"] = Gliner2ModelAssets.ModelSha256;
        artifacts["tokenizer_version"] = BenchEnvironment.TokenizerVersion;
        return artifacts;
    }

    internal static Gliner2ModelFiles Files()
    {
        var (files, reason) = Gliner2TestAssets.FindModel();
        return files ?? throw new BenchAssetsMissingException(reason);
    }

    internal static string FixturePath()
    {
        var (path, reason) = Gliner2TestAssets.FindExtractionFixtures();
        return path ?? throw new BenchAssetsMissingException(reason);
    }
}

/// <summary>One extraction item: text, labels, descriptions and threshold, and its fixture case when it has one.</summary>
internal sealed record Gliner2Item(string Id, string Text, IReadOnlyList<string> Labels, IReadOnlyDictionary<string, string>? Descriptions, float Threshold, Gliner2Case? Case);

/// <summary>What one call produced for one item.</summary>
internal sealed record Gliner2Output(Gliner2Item Item, Gliner2ModelInput Input, Gliner2ModelOutput Output, Gliner2ExtractionResult Result);

internal sealed class Gliner2Run : IBenchRun
{
    private static readonly string[] s_stages = ["prepare", "inference", "decode"];
    private readonly HarnessContext _context;
    private readonly Gliner2EntityExtractor _extractor;
    private readonly IReadOnlyDictionary<string, Gliner2Case> _cases;
    private readonly Gliner2Tolerances _tolerances;
    private readonly WorkloadSpec _spec;
    private readonly Dictionary<string, Gliner2ExtractionResult> _selfReference = new(StringComparer.Ordinal);

    public Gliner2Run(HarnessContext context, Gliner2ModelFiles files)
    {
        _context = context;
        long t0 = Stopwatch.GetTimestamp();
        _cases = Gliner2Fixtures.LoadCases(Gliner2Driver.FixturePath()).ToDictionary(c => c.Id, StringComparer.Ordinal);
        _tolerances = Gliner2Tolerances.Load("onnx-fp32-cpu");
        _spec = WorkloadSpec.Load(context.WorkloadPath);
        double fixtureMs = MonotonicClock.ElapsedMs(t0);
        long t1 = Stopwatch.GetTimestamp();
        var session = OnnxModelSession.Create(files.ModelPath, context.SessionOptions() with
        {
            ExpectedModelSha256 = context.HashModelAtLoad ? Gliner2ModelAssets.ModelSha256 : null,
        });
        double sessionMs = MonotonicClock.ElapsedMs(t1);
        long t2 = Stopwatch.GetTimestamp();
        _extractor = Gliner2EntityExtractor.Create(session, files.TokenizerPath, new Gliner2ExtractorOptions(), ownsSession: true);
        double tokenizerMs = MonotonicClock.ElapsedMs(t2);
        LoadStages = new JsonObject { ["fixture_read"] = fixtureMs, ["session_create"] = sessionMs, ["tokenizer_load"] = tokenizerMs };
    }

    public JsonObject LoadStages { get; }

    public JsonObject Settings
    {
        get
        {
            var settings = _context.SessionSettings();
            settings["max_sequence_length"] = _extractor.Options.MaxSequenceLength;
            settings["max_batch_size"] = _extractor.Options.MaxBatchSize;
            settings["threshold"] = "per case (fixture)";
            return settings;
        }
    }

    public IReadOnlyList<string>? SessionProviders => _extractor.Session.Diagnostics.SessionProviders;

    public JsonObject PreliminaryQuality()
    {
        var items = _cases.Values.Where(c => c.OutOfContract is null).Select(FromCase).ToList();
        var outputs = items.Select((item, i) => (i, (object)Extract([item], new double[4]))).ToList();
        var quality = Judge(outputs);
        quality["reference"] = Gliner2Driver.ReferencePath;
        quality["tolerance_file"] = _tolerances.Source;
        quality["cases"] = items.Count;
        quality["excluded_out_of_contract"] = _cases.Count - items.Count;
        return quality;
    }

    public IEnumerable<BenchSeries> WarmSeries()
    {
        foreach (var w in _spec.Workloads)
        {
            var requests = Requests(w);
            foreach (int c in w.Concurrency)
            {
                yield return Series(w.Name, w.Path, w.Input, w.Batch, c, w.Samples, w.WarmupCalls ?? _spec.WarmupCalls, requests);
            }
        }
    }

    public BenchSeries ColdSeries()
    {
        var request = WorkloadSpec.Strings(_spec.Cold.GetProperty("request")).Select(id => FromCase(Case(id))).ToArray();
        return Series("cold-start", "text", "short", request.Length, 1, 1, 0, [request]);
    }

    public void Dispose() => _extractor.Dispose();

    private BenchSeries Series(string name, string path, string input, int batch, int concurrency, int samples, int warmup, Gliner2Item[][] requests) => new()
    {
        Workload = name,
        Path = path,
        Input = input,
        Batch = batch,
        Concurrency = concurrency,
        Samples = samples,
        WarmupCalls = Math.Max(warmup, requests.Length),
        Requests = requests.Length,
        Stages = s_stages,
        Call = (r, stages, _) => Extract(requests[r], stages),
        Judge = Judge,
    };

    private Gliner2Item[][] Requests(WorkloadEntry w)
    {
        if (w.Raw.TryGetProperty("synthetic", out var synthetic))
        {
            string sourceId = synthetic.GetProperty("case").GetString()!;
            var source = _cases.TryGetValue(sourceId, out var found) ? found : throw new ArgumentException($"Workload names unknown fixture case '{sourceId}'.");
            int words = synthetic.GetProperty("words").GetInt32();
            string text = string.Join(' ', source.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Take(words));
            var item = new Gliner2Item($"{source.Id}:first-{words}-words", text, source.LabelsRequested, source.Descriptions, source.Threshold, null);
            if (!_selfReference.ContainsKey(item.Id))
            {
                _selfReference[item.Id] = _extractor.Extract(item.Text, item.Labels, item.Threshold);
            }

            return [[item]];
        }

        return w.Raw.GetProperty("requests").EnumerateArray()
            .Select(r => WorkloadSpec.Strings(r).Select(id => FromCase(Case(id))).ToArray())
            .ToArray();
    }

    private Gliner2Case Case(string id)
    {
        var c = _cases.TryGetValue(id, out var found) ? found : throw new ArgumentException($"Workload names unknown fixture case '{id}'.");
        return c.OutOfContract is null ? c : throw new ArgumentException($"Fixture case '{id}' is outside the C# contract ({c.OutOfContract}).");
    }

    private static Gliner2Item FromCase(Gliner2Case c) => new(c.Id, c.Text, c.LabelsRequested, c.Descriptions, c.Threshold, c);

    private Gliner2Output[] Extract(Gliner2Item[] request, double[] stages)
    {
        long t = Stopwatch.GetTimestamp();
        var inputs = new Gliner2ModelInput[request.Length];
        for (int i = 0; i < request.Length; i++)
        {
            inputs[i] = _extractor.Prepare(request[i].Text, request[i].Labels, request[i].Descriptions);
        }

        stages[0] = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        var outputs = _extractor.RunModel(inputs);
        stages[1] = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        var result = new Gliner2Output[request.Length];
        for (int i = 0; i < request.Length; i++)
        {
            result[i] = new Gliner2Output(request[i], inputs[i], outputs[i], _extractor.Decode(inputs[i], outputs[i], request[i].Threshold));
        }

        stages[2] = MonotonicClock.ElapsedMs(t);
        return result;
    }

    /// <summary>
    /// Fixture items: entities equal to the reference under the protected order and flip rules, confidence within
    /// <c>float.confidence_max_abs</c>, logits finite. Synthetic items: entities equal to the same text extracted alone
    /// at load, confidence within the same bound.
    /// </summary>
    private JsonObject Judge(IReadOnlyList<(int Request, object Output)> outputs)
    {
        int items = 0, exact = 0, synthetic = 0;
        double confidence = 0;
        bool finite = true;
        var failures = new JsonArray();
        foreach (var (_, o) in outputs)
        {
            foreach (var x in (Gliner2Output[])o)
            {
                items++;
                finite &= AllFinite(x.Output.CountLogits) && AllFinite(x.Output.SpanLogits);
                if (x.Item.Case is { } c)
                {
                    var comparison = Gliner2ParityChecks.CompareEntities(c, Gliner2ParityChecks.UpstreamEntities(x.Input, x.Output, c.Threshold), _tolerances);
                    exact += comparison.Exact ? 1 : 0;
                    confidence = Math.Max(confidence, comparison.ConfidenceMaxAbs);
                    if (!comparison.WithinRules)
                    {
                        failures.Add((JsonNode)$"{c.Id}: {comparison.Detail}");
                    }
                }
                else
                {
                    synthetic++;
                    var reference = _selfReference[x.Item.Id].Entities;
                    var actual = x.Result.Entities;
                    bool same = reference.Count == actual.Count && reference.Zip(actual).All(p =>
                        p.First.Label == p.Second.Label && p.First.ScalarStart == p.Second.ScalarStart && p.First.ScalarEnd == p.Second.ScalarEnd);
                    if (same)
                    {
                        exact++;
                        confidence = Math.Max(confidence, reference.Zip(actual).Select(p => Math.Abs((double)p.First.Confidence - p.Second.Confidence)).DefaultIfEmpty(0).Max());
                    }
                    else
                    {
                        failures.Add((JsonNode)$"{x.Item.Id}: entities differ from the single extraction at load");
                    }
                }
            }
        }

        double tolerance = _tolerances.ConfidenceMaxAbs;
        return new JsonObject
        {
            ["items"] = items,
            ["synthetic_items"] = synthetic,
            ["entities_exact"] = exact,
            ["confidence_max_abs"] = confidence,
            ["confidence_tolerance"] = tolerance,
            ["finite"] = finite,
            ["failures"] = failures,
            ["pass"] = failures.Count == 0 && finite && confidence <= tolerance,
        };
    }

    private static bool AllFinite(ReadOnlySpan<float> values)
    {
        foreach (float v in values)
        {
            if (!float.IsFinite(v))
            {
                return false;
            }
        }

        return true;
    }
}
