using System.Diagnostics;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Extensions.AI;
using Starlings.Runtime.Onnx.Batching;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>
/// EmbeddingGemma 2 text under the batching sweep (<see cref="BatchSweep"/>): <see cref="EmbeddingGemma2Tokenizer.Tokenize"/>
/// per text with the package prompt of its kind (queries <see cref="EmbeddingGemma2Prompt.Query"/>, passages
/// <see cref="EmbeddingGemma2Prompt.Document"/>; stage <c>tokenize</c>), then one
/// <see cref="EmbeddingGemma2Encoder.EmbedBatch(IReadOnlyList{EmbeddingGemma2Input}, CancellationToken)"/> (stage
/// <c>encode</c>). The encoder's chunk plan (rows and padded cells per run) is mirrored for the padding figures.
/// Outputs against the same text embedded alone (<c>float.padded_batch_vs_single_embedding_max_abs</c>). With
/// micro-batching (<see cref="SweepOptions.MicroBatching"/>) each text of a call is one
/// <see cref="EmbeddingGemma2EmbeddingGenerator.GenerateAsync"/> call with one value (prompt already applied, generator
/// prompt none) on a generator with <see cref="EmbeddingGemma2EmbeddingGeneratorOptions.MicroBatching"/> set (stage <c>call</c>).
/// </summary>
internal sealed class EmbeddingGemma2SweepTarget : ISweepTarget
{
    private static readonly string[] s_stages = ["tokenize", "encode"];
    private static readonly string[] s_microStages = ["call"];
    private readonly HarnessContext _context;
    private readonly EmbeddingGemma2Tokenizer _tokenizer;
    private readonly EmbeddingGemma2Encoder _encoder;
    private readonly EmbeddingGemma2EmbeddingGenerator? _generator;
    private readonly EmbeddingGemma2Tolerances _tolerances;

    public EmbeddingGemma2SweepTarget(HarnessContext context, SweepOptions options)
    {
        _context = context;
        var files = EmbeddingGemma2Driver.TextFiles();
        _tolerances = EmbeddingGemma2Tolerances.Load(context.ProfileOf("onnx-fp32"));
        long t0 = Stopwatch.GetTimestamp();
        _tokenizer = EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath);
        double tokenizerMs = MonotonicClock.ElapsedMs(t0);
        long t1 = Stopwatch.GetTimestamp();
        var defaults = new EmbeddingGemma2EncoderOptions();
        _encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, defaults with
        {
            Session = context.SessionOptions(),
            MaxBatchSize = options.MaxBatchSize ?? defaults.MaxBatchSize,
            MaxBatchTokens = options.MaxBatchTokens ?? defaults.MaxBatchTokens,
            SortByLength = options.SortByLength,
        });
        if (options.MicroBatching)
        {
            _generator = new EmbeddingGemma2EmbeddingGenerator(new EmbeddingGemma2TextEmbedder(_tokenizer, _encoder), new EmbeddingGemma2EmbeddingGeneratorOptions
            {
                MicroBatching = new MicroBatchingOptions
                {
                    MaxWait = TimeSpan.FromMilliseconds(options.MicroBatchWaitMs ?? 0),
                    MaxRows = options.MicroBatchRows,
                    MaxTokens = options.MicroBatchMaxTokens,
                },
            });
        }

        LoadStages = new JsonObject { ["tokenizer_load"] = tokenizerMs, ["session_create"] = MonotonicClock.ElapsedMs(t1) };
        ModelPath = files.ModelPath;
        TokenizerPath = files.TokenizerPath;
    }

    private string ModelPath { get; }

    private string TokenizerPath { get; }

    public string Model => "embeddinggemma-2";

    public string Profile => _context.ProfileOf("onnx-fp32");

    public string HarnessName => "Starlings.Benchmarks.EmbeddingGemma2Harness";

    public string Provider => _context.Provider;

    public int? IntraOpThreads => _context.IntraOpThreads;

    public int MaxBatchSize => _encoder.Options.MaxBatchSize;

    public int? MaxBatchTokens => _encoder.Options.MaxBatchTokens;

    public int MaxTokens => _encoder.Options.MaxSequenceLength;

    public string[] Stages => _generator is null ? s_stages : s_microStages;

    public JsonObject LoadStages { get; }

    public JsonObject Settings
    {
        get
        {
            var settings = _context.SessionSettings(_encoder.Session.Diagnostics.ProviderOptions);
            settings["max_batch_size"] = _encoder.Options.MaxBatchSize;
            settings["max_batch_tokens"] = _encoder.Options.MaxBatchTokens;
            settings["sort_by_length"] = _encoder.Options.SortByLength;
            settings["micro_batching"] = SweepTargets.MicroBatching(_generator?.Options.MicroBatching);
            settings["max_sequence_length"] = _encoder.Options.MaxSequenceLength;
            settings["dimension"] = _encoder.Options.Dimension;
            settings["pad_id"] = EmbeddingGemma2Encoder.PadTokenId;
            settings["prompts"] = "query (queries), document (passages)";
            return settings;
        }
    }

    public IReadOnlyList<string>? SessionProviders => _encoder.Session.Diagnostics.SessionProviders;

    public string QualityCheck => "embedding max abs vs the same text embedded alone (float.padded_batch_vs_single_embedding_max_abs)";

    public double Tolerance => _tolerances.PaddedBatchVsSingleMaxAbs;

    public string ToleranceFile => _tolerances.Source;

    public JsonObject Artifacts(SweepWorkloadFile file)
    {
        var artifacts = BenchEnvironment.Artifacts(ModelPath, null, TokenizerPath, file.Path,
            BenchEnvironment.FileDigests([.. file.SourceFiles(Model), Path.Combine(BenchPaths.RepositoryRoot, _tolerances.Source)]),
            HarnessHost.Packages(("Starlings.Models.EmbeddingGemma2", BenchEnvironment.InformationalVersion(typeof(EmbeddingGemma2Encoder).Assembly))));
        artifacts["model_id"] = $"{EmbeddingGemma2ModelAssets.Repository}@{EmbeddingGemma2ModelAssets.Revision} ({EmbeddingGemma2ModelAssets.ModelFile})";
        return artifacts;
    }

    /// <summary>The protected text rows below 512 tokens, each alone, against their reference embeddings.</summary>
    public JsonObject? PreliminaryQuality()
    {
        var rows = EmbeddingGemma2Fixtures.LoadRows().Where(r => r.InputIds.Length < 512).ToArray();
        var q = Quality.Empty;
        foreach (var row in rows)
        {
            q = q.Merge(Quality.Of(_encoder.EmbedBatch([_tokenizer.Tokenize(row.PackagePrompt.Apply(row.Text))]), [row.Embedding]));
        }

        double tolerance = _tolerances.EmbeddingMaxAbs;
        var json = q.ToJson();
        json["reference"] = EmbeddingGemma2Driver.TextReference;
        json["cases"] = rows.Length;
        json["scope"] = "rows below 512 tokens, each alone";
        json["tolerance"] = tolerance;
        json["pass"] = q.MaxAbs <= tolerance && q.Finite;
        return json;
    }

    public int CountTokens(SweepItem item) => _tokenizer.Tokenize(Text(item)).Length;

    public object Call(SweepItem[] items, double[] stages)
    {
        long t = Stopwatch.GetTimestamp();
        if (_generator is not null)
        {
            var calls = new Task<GeneratedEmbeddings<Embedding<float>>>[items.Length];
            for (int i = 0; i < items.Length; i++)
            {
                calls[i] = _generator.GenerateAsync([Text(items[i])]);
            }

            var vectors = new float[items.Length][];
            for (int i = 0; i < items.Length; i++)
            {
                vectors[i] = calls[i].GetAwaiter().GetResult()[0].Vector.ToArray();
            }

            stages[0] = MonotonicClock.ElapsedMs(t);
            return vectors;
        }

        var inputs = new EmbeddingGemma2Input[items.Length];
        for (int i = 0; i < items.Length; i++)
        {
            inputs[i] = _tokenizer.Tokenize(Text(items[i]));
        }

        stages[0] = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        var output = _encoder.EmbedBatch(inputs);
        stages[1] = MonotonicClock.ElapsedMs(t);
        return output;
    }

    public object Single(SweepItem item, object output) => ((float[][])output)[0];

    /// <summary>The encoder's chunk plan (BatchPlan, internal): rows, padded-cell budget and length sorting as configured.</summary>
    public (long Cells, int Runs) Padding(IReadOnlyList<int> lengths) =>
        SweepMath.Plan(lengths, _encoder.Options.MaxBatchSize, _encoder.Options.MaxBatchTokens, _encoder.Options.SortByLength);

    public SweepCheck Compare(SweepItem[] items, object output, IReadOnlyDictionary<string, object> singles) =>
        SweepCheck.Embeddings(items, (float[][])output, singles);

    public SweepBatcherStats? BatcherStatistics => SweepTargets.Stats(_generator?.MicroBatchingStatistics);

    public void Dispose()
    {
        _generator?.Dispose();
        _encoder.Dispose();
    }

    private static string Text(SweepItem item) => (item.Kind == "query" ? EmbeddingGemma2Prompt.Query : EmbeddingGemma2Prompt.Document).Apply(item.Text);
}
