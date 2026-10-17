using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Models.Gliner2;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>
/// GLiNER2 under the batching sweep (<see cref="BatchSweep"/>): every item of a call shares the WNUT-17 label set (one
/// GLiNER2 call runs one label set). Stages as the warm harness: <c>prepare</c>, <c>inference</c>
/// (<see cref="Gliner2EntityExtractor.RunModel"/>: padding and the ORT runs, including any wait on the CUDA run gate),
/// <c>decode</c>. Each output against the same text run alone: valid span logits within
/// <c>float.padded_batch_vs_single_span_logits_relative</c> and entities identical.
/// </summary>
internal sealed class Gliner2SweepTarget : ISweepTarget
{
    private static readonly string[] s_stages = ["prepare", "inference", "decode"];
    private readonly HarnessContext _context;
    private readonly Gliner2EntityExtractor _extractor;
    private readonly Gliner2Tolerances _tolerances;
    private readonly string[] _labels;

    public Gliner2SweepTarget(HarnessContext context, SweepOptions options)
    {
        _context = context;
        var files = Gliner2Driver.Files();
        _tolerances = Gliner2Tolerances.Load(context.ProfileOf("onnx-fp32"));
        _labels = Labels(SweepWorkloadFile.Load(context.WorkloadPath));
        long t0 = Stopwatch.GetTimestamp();
        var session = OnnxModelSession.Create(files.ModelPath, context.SessionOptions());
        double sessionMs = MonotonicClock.ElapsedMs(t0);
        long t1 = Stopwatch.GetTimestamp();
        var defaults = new Gliner2ExtractorOptions();
        _extractor = Gliner2EntityExtractor.Create(session, files.TokenizerPath, defaults with
        {
            MaxBatchSize = options.MaxBatchSize ?? defaults.MaxBatchSize,
        }, ownsSession: true);
        LoadStages = new JsonObject { ["session_create"] = sessionMs, ["tokenizer_load"] = MonotonicClock.ElapsedMs(t1) };
    }

    public string Model => "gliner2-base-v1";

    public string Profile => _context.ProfileOf("onnx-fp32");

    public string HarnessName => "Starlings.Benchmarks.Gliner2Harness";

    public string Provider => _context.Provider;

    public int? IntraOpThreads => _context.IntraOpThreads;

    public int MaxBatchSize => _extractor.Options.MaxBatchSize;

    public int? MaxBatchTokens => null;

    public int MaxTokens => _extractor.Options.MaxSequenceLength;

    public string[] Stages => s_stages;

    public JsonObject LoadStages { get; }

    public JsonObject Settings
    {
        get
        {
            var settings = _context.SessionSettings(_extractor.Session.Diagnostics.ProviderOptions);
            settings["max_batch_size"] = _extractor.Options.MaxBatchSize;
            settings["max_sequence_length"] = _extractor.Options.MaxSequenceLength;
            settings["threshold"] = _extractor.Options.Threshold;
            settings["serializes_native_calls"] = _extractor.SerializesNativeCalls;
            settings["labels"] = new JsonArray(_labels.Select(l => (JsonNode)l).ToArray());
            return settings;
        }
    }

    public IReadOnlyList<string>? SessionProviders => _extractor.Session.Diagnostics.SessionProviders;

    public string QualityCheck => "valid span logits relative difference |a-e|/max(1,|e|) vs the same text run alone (float.padded_batch_vs_single_span_logits_relative); entities identical (mismatches)";

    public double Tolerance => _tolerances.PaddedBatchVsSingleRelative;

    public string ToleranceFile => _tolerances.Source;

    public JsonObject Artifacts(SweepWorkloadFile file)
    {
        var files = Gliner2Driver.Files();
        var artifacts = BenchEnvironment.Artifacts(files.ModelPath, null, files.TokenizerPath, file.Path,
            BenchEnvironment.FileDigests([.. file.SourceFiles(Model), Gliner2Driver.FixturePath(), Path.Combine(BenchPaths.RepositoryRoot, _tolerances.Source)]),
            HarnessHost.Packages(("Starlings.Models.Gliner2", BenchEnvironment.InformationalVersion(typeof(Gliner2EntityExtractor).Assembly))));
        artifacts["model_id"] = $"{Gliner2ModelAssets.Repository}@{Gliner2ModelAssets.Revision} ({Gliner2ModelAssets.ModelFile})";
        return artifacts;
    }

    /// <summary>Every in-contract protected case alone: entities within the protected rules, as the warm harness's preliminary sweep.</summary>
    public JsonObject? PreliminaryQuality()
    {
        int n = 0, exact = 0;
        double confidence = 0;
        var failures = new JsonArray();
        foreach (var c in Gliner2Fixtures.LoadCases(Gliner2Driver.FixturePath()).Where(c => c.OutOfContract is null))
        {
            var input = _extractor.Prepare(c.Text, c.LabelsRequested, c.Descriptions);
            var output = _extractor.RunModel([input])[0];
            var comparison = Gliner2ParityChecks.CompareEntities(c, Gliner2ParityChecks.UpstreamEntities(input, output, c.Threshold), _tolerances);
            n++;
            exact += comparison.Exact ? 1 : 0;
            confidence = Math.Max(confidence, comparison.ConfidenceMaxAbs);
            if (!comparison.WithinRules)
            {
                failures.Add((JsonNode)$"{c.Id}: {comparison.Detail}");
            }
        }

        return new JsonObject
        {
            ["reference"] = Gliner2Driver.ReferencePath,
            ["cases"] = n,
            ["entities_exact"] = exact,
            ["confidence_max_abs"] = confidence,
            ["confidence_tolerance"] = _tolerances.ConfidenceMaxAbs,
            ["failures"] = failures,
            ["pass"] = failures.Count == 0 && confidence <= _tolerances.ConfidenceMaxAbs,
        };
    }

    public int CountTokens(SweepItem item) => _extractor.Prepare(item.Text, _labels).TokenCount;

    public object Call(SweepItem[] items, double[] stages)
    {
        long t = Stopwatch.GetTimestamp();
        var inputs = new Gliner2ModelInput[items.Length];
        for (int i = 0; i < items.Length; i++)
        {
            inputs[i] = _extractor.Prepare(items[i].Text, _labels);
        }

        stages[0] = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        var outputs = _extractor.RunModel(inputs);
        stages[1] = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        var result = new Gliner2SweepOutput[items.Length];
        for (int i = 0; i < items.Length; i++)
        {
            result[i] = new Gliner2SweepOutput(outputs[i], _extractor.Decode(inputs[i], outputs[i]));
        }

        stages[2] = MonotonicClock.ElapsedMs(t);
        return result;
    }

    public object Single(SweepItem item, object output) => ((Gliner2SweepOutput[])output)[0];

    public IEnumerable<int> PlanChunks(IReadOnlyList<int> lengths) => SweepMath.FixedChunks(lengths.Count, _extractor.Options.MaxBatchSize);

    public SweepCheck Compare(SweepItem[] items, object output, IReadOnlyDictionary<string, object> singles)
    {
        var outputs = (Gliner2SweepOutput[])output;
        var check = SweepCheck.Empty;
        for (int i = 0; i < items.Length; i++)
        {
            var single = (Gliner2SweepOutput)singles[items[i].Id];
            var errors = new Gliner2Errors();
            errors.Add(outputs[i].Output.GetValidSpanLogits(), single.Output.GetValidSpanLogits(), items[i].Id);
            var a = outputs[i].Result.Entities;
            var e = single.Result.Entities;
            bool same = a.Count == e.Count && a.Zip(e).All(p => p.First.Label == p.Second.Label && p.First.ScalarStart == p.Second.ScalarStart && p.First.ScalarEnd == p.Second.ScalarEnd);
            check = check.Merge(new SweepCheck(1, errors.MaxRelative, !errors.NonFinite, same ? 0 : 1, items[i].Id));
        }

        return check;
    }

    public void Dispose() => _extractor.Dispose();

    /// <summary>The label set of the NER source (every WNUT-17 row has the same labels; checked).</summary>
    private static string[] Labels(SweepWorkloadFile file)
    {
        string path = Path.Combine(BenchPaths.RepositoryRoot, file.Root.GetProperty("sources").GetProperty("ner").GetProperty("texts").GetString()!);
        string[]? labels = null;
        foreach (string line in File.ReadLines(path).Where(l => !string.IsNullOrWhiteSpace(l)))
        {
            using var doc = JsonDocument.Parse(line);
            var row = doc.RootElement.GetProperty("labels").EnumerateArray().Select(e => e.GetString()!).ToArray();
            if (labels is not null && !labels.SequenceEqual(row))
            {
                throw new InvalidDataException($"{path}: rows with different label sets; the sweep runs one label set per call.");
            }

            labels = row;
        }

        return labels ?? throw new InvalidDataException($"{path}: no rows.");
    }
}

/// <summary>One item's graph output and decoded entities.</summary>
internal sealed record Gliner2SweepOutput(Gliner2ModelOutput Output, Gliner2ExtractionResult Result);
