using System.Diagnostics;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using Starlings.Models.E5;
using Starlings.Models.E5.Extensions.AI;
using Starlings.Runtime.Onnx.Batching;

namespace Starlings.Benchmarks;

/// <summary>
/// e5 under the batching sweep (<see cref="BatchSweep"/>): <see cref="E5Tokenizer.Tokenize"/> per text with the e5
/// prefix of its kind (<c>query: </c> or <c>passage: </c>; stage <c>tokenize</c>), then one
/// <see cref="E5Encoder.EmbedBatch(IReadOnlyList{EncoderInput}, CancellationToken)"/> (stage <c>encode</c>). Outputs are
/// compared with the same text embedded alone (<c>float.padded_batch_vs_single_embedding_max_abs</c>).
/// With micro-batching (<see cref="SweepOptions.MicroBatching"/>) each text of a call is one
/// <see cref="E5EmbeddingGenerator.GenerateAsync"/> call with one value on a generator with
/// <see cref="E5EmbeddingGeneratorOptions.MicroBatching"/> set (stage <c>call</c>: tokenize, queue and run).
/// </summary>
internal sealed class E5SweepTarget : ISweepTarget
{
    private static readonly string[] s_stages = ["tokenize", "encode"];
    private static readonly string[] s_microStages = ["call"];
    private readonly HarnessContext _context;
    private readonly E5ModelFiles _files;
    private readonly E5Tokenizer _tokenizer;
    private readonly E5Tokenizer _counter;
    private readonly E5Encoder _encoder;
    private readonly E5EmbeddingGenerator? _generator;
    private readonly Starlings.TestSupport.E5Tolerances _tolerances;

    public E5SweepTarget(HarnessContext context, SweepOptions options)
    {
        _context = context;
        _files = BenchFixtures.Files(context.Args.Get("snapshot") ?? Environment.GetEnvironmentVariable("STARLING_E5_SNAPSHOT"));
        long t0 = Stopwatch.GetTimestamp();
        _tokenizer = E5Tokenizer.FromFile(_files.TokenizerPath);
        _counter = E5Tokenizer.FromFile(_files.TokenizerPath, maxLength: 1 << 20);
        double tokenizerMs = MonotonicClock.ElapsedMs(t0);
        long t1 = Stopwatch.GetTimestamp();
        var defaults = new E5EncoderOptions();
        _encoder = E5Encoder.Create(_files.ModelPath, defaults with
        {
            Session = context.SessionOptions(),
            MaxBatchSize = options.MaxBatchSize ?? defaults.MaxBatchSize,
            MaxBatchTokens = options.MaxBatchTokens ?? defaults.MaxBatchTokens,
            SortByLength = options.SortByLength,
        });
        if (options.MicroBatching)
        {
            _generator = new E5EmbeddingGenerator(new E5TextEmbedder(_tokenizer, _encoder), new E5EmbeddingGeneratorOptions
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
        _tolerances = Starlings.TestSupport.E5Tolerances.Load(BenchFixtures.FixtureDirectory(_encoder.Options.Session.Provider));
    }

    public string Model => "multilingual-e5-small";

    public string Profile => _context.ProfileOf("onnx-fp32");

    public string HarnessName => "Starlings.Benchmarks.E5Harness";

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
            settings["pad_id"] = _encoder.Options.PadTokenId;
            settings["prefixes"] = "query: (queries), passage: (passages)";
            return settings;
        }
    }

    public IReadOnlyList<string>? SessionProviders => _encoder.Session.Diagnostics.SessionProviders;

    public string QualityCheck => "embedding max abs vs the same text embedded alone (float.padded_batch_vs_single_embedding_max_abs)";

    public double Tolerance => _tolerances.PaddedBatchVsSingleMaxAbs;

    public string ToleranceFile => BenchEnvironment.RepoRelative(_tolerances.Source);

    public JsonObject Artifacts(SweepWorkloadFile file)
    {
        var packages = HarnessHost.Packages(("Starlings.Models.E5", BenchEnvironment.InformationalVersion(typeof(E5Encoder).Assembly)));
        return BenchEnvironment.Artifacts(_files.ModelPath, null, _files.TokenizerPath, file.Path,
            BenchEnvironment.FileDigests([.. file.SourceFiles(Model), _tolerances.Source]), packages);
    }

    /// <summary>The 46 protected cases alone and in batches of 16 against the protected embeddings.</summary>
    public JsonObject? PreliminaryQuality()
    {
        var cases = BenchFixtures.Ordered;
        var single = Quality.Empty;
        foreach (var c in cases)
        {
            single = single.Merge(Quality.Of(_encoder.EmbedBatch([new EncoderInput(c.Ids)]), [c.Embedding]));
        }

        var batched = Quality.Empty;
        foreach (var chunk in cases.Chunk(16))
        {
            batched = batched.Merge(Quality.Of(_encoder.EmbedBatch(chunk.Select(c => new EncoderInput(c.Ids)).ToArray()), BenchFixtures.Embeddings(chunk)));
        }

        double tolerance = _tolerances.EmbeddingMaxAbs;
        return new JsonObject
        {
            ["reference"] = "model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl",
            ["cases"] = cases.Count,
            ["ids_single"] = single.ToJson(),
            ["ids_batch16"] = batched.ToJson(),
            ["embedding_max_abs_tolerance"] = tolerance,
            ["pass"] = single.MaxAbs <= tolerance && batched.MaxAbs <= tolerance && single.Finite && batched.Finite,
        };
    }

    public int CountTokens(SweepItem item) => _counter.Tokenize(Text(item)).InputIds.Length;

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

        var inputs = new EncoderInput[items.Length];
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

    public (long Cells, int Runs) Padding(IReadOnlyList<int> lengths) =>
        SweepMath.Plan(lengths, _encoder.Options.MaxBatchSize, _encoder.Options.MaxBatchTokens, _encoder.Options.SortByLength);

    public SweepBatcherStats? BatcherStatistics => SweepTargets.Stats(_generator?.MicroBatchingStatistics);

    public SweepCheck Compare(SweepItem[] items, object output, IReadOnlyDictionary<string, object> singles) =>
        SweepCheck.Embeddings(items, (float[][])output, singles);

    public void Dispose()
    {
        _generator?.Dispose();
        _encoder.Dispose();
    }

    private static string Text(SweepItem item) => (item.Kind == "query" ? "query: " : "passage: ") + item.Text;
}
