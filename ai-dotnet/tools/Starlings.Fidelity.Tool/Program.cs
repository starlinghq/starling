// Stage results for the staged fidelity evaluator (P-007).
//
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- e5 --out <dir> [options]
//
// Runs the C# e5 path on the protected fixture cases and writes, in the stage results format
// (model-pipeline/src/model_pipeline/results.py, schemas/stage-results.schema.json):
//   tokenizer.results.json   stage "tokenizer": E5Tokenizer ids and masks per fixture text.
//   dotnet.results.json      stage "dotnet": embeddings from fixture ids (E5Encoder), from text (E5TextEmbedder),
//                            the protected padded batch, all cases in padded batches of 16, one text batch, and
//                            (with --hidden) last hidden states from the same ONNX Runtime session.
// GLiNER2 (P-015): `-- gliner2 --out <dir>`; see Gliner2Dump.cs.
// EmbeddingGemma 2: `-- embeddinggemma2 --out <dir>`; see EmbeddingGemma2Dump.cs. Images: `-- embeddinggemma2-images
// --out <dir> --images <dir>`; see EmbeddingGemma2ImageDump.cs. Videos: `-- embeddinggemma2-videos --out <dir> --videos <dir>
// [--ffmpeg <dir>]`; see EmbeddingGemma2VideoDump.cs.
// Task quality (G3): `-- retrieval --model e5|embeddinggemma2 --dataset <dir> --out <file>` and `-- ner --dataset
// <gold.jsonl> --out <file>`; see TaskQualityDump.cs (inputs of `model-pipeline evaluate --g3`).
// Then `model-pipeline evaluate` compares them with the protected fixtures and tolerances. This tool applies
// no thresholds and writes nothing under model-pipeline/protected/.
//
// Options:
//   --out <dir>        Results directory (required). Put Python stage 2 results in the same directory.
//   --snapshot <dir>   Existing e5 snapshot (default: $STARLING_E5_SNAPSHOT, else the acquisition cache).
//   --fixtures <dir>   Protected fixtures (default: <repo>/model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu).
//   --profile <id>     Profile id written into the results (default: multilingual-e5-small/onnx-fp32-cpu).
//   --hidden           Also dump last hidden states (report-only comparison with Python ONNX Runtime).
//   --threads <n>      Bound ONNX Runtime intra-op threads (default: runtime choice).
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Runtime.Onnx;
using Microsoft.ML.OnnxRuntime;

if (args.Length >= 1 && args[0] is "retrieval" or "ner")
{
    return TaskQualityDump.Run(args[0], args[1..]);
}

if (args.Length >= 1 && args[0] == "gliner2")
{
    return Gliner2Dump.Run(args[1..]);
}

if (args.Length >= 1 && args[0] == "embeddinggemma2-videos")
{
    return EmbeddingGemma2VideoDump.Run(args[1..]);
}

if (args.Length >= 1 && args[0] == "embeddinggemma2-images")
{
    return EmbeddingGemma2ImageDump.Run(args[1..]);
}

if (args.Length >= 1 && args[0] == "embeddinggemma2")
{
    return EmbeddingGemma2Dump.Run(args[1..]);
}

if (args.Length < 1 || args[0] is "-h" or "--help" || args[0] != "e5")
{
    return Usage(args.Length > 0 && args[0] is not ("-h" or "--help") ? $"unknown model '{args[0]}'" : null);
}

string? outDir = null;
string? snapshot = Environment.GetEnvironmentVariable("STARLING_E5_SNAPSHOT");
string? fixturesDir = null;
string profile = "multilingual-e5-small/onnx-fp32-cpu";
bool hidden = false;
int? threads = null;
for (int i = 1; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--out" when i + 1 < args.Length:
            outDir = args[++i];
            break;
        case "--snapshot" when i + 1 < args.Length:
            snapshot = args[++i];
            break;
        case "--fixtures" when i + 1 < args.Length:
            fixturesDir = args[++i];
            break;
        case "--profile" when i + 1 < args.Length:
            profile = args[++i];
            break;
        case "--hidden":
            hidden = true;
            break;
        case "--threads" when i + 1 < args.Length:
            if (!int.TryParse(args[++i], NumberStyles.None, CultureInfo.InvariantCulture, out int count) || count <= 0)
            {
                return Usage("--threads must be a positive integer");
            }

            threads = count;
            break;
        default:
            return Usage($"unknown option {args[i]}");
    }
}

if (outDir is null)
{
    return Usage("--out is required");
}

string? repoRoot = ToolSupport.FindRepositoryRoot();
fixturesDir ??= repoRoot is null
    ? null
    : Path.Combine(repoRoot, "model-pipeline", "protected", "multilingual-e5-small", "onnx-fp32-cpu");
if (fixturesDir is null || !File.Exists(Path.Combine(fixturesDir, "embeddings.jsonl")))
{
    return Usage($"protected fixtures not found ({fixturesDir ?? "no repository root"}); pass --fixtures");
}

E5ModelFiles files;
try
{
    files = string.IsNullOrEmpty(snapshot)
        ? E5ModelAssets.OpenLocal()
        : E5ModelAssets.OpenDirectory(snapshot, CacheVerification.Sha256);
}
catch (ModelAcquisitionException ex)
{
    Console.Error.WriteLine(ex.Message);
    return 2;
}

var cases = Fixtures.LoadCases(fixturesDir);
var texts = Fixtures.LoadTexts(fixturesDir);
var batchRows = Fixtures.LoadBatchRows(fixturesDir);
Directory.CreateDirectory(outDir);

string? commit = ToolSupport.GitHead(repoRoot);
var tokenizer = E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256);
using var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
{
    Session = new OnnxSessionOptions
    {
        ExpectedModelSha256 = E5ModelAssets.ModelSha256,
        IntraOpNumThreads = threads,
        InterOpNumThreads = threads is null ? null : 1,
    },
});
var embedder = new E5TextEmbedder(tokenizer, encoder);
var diagnostics = encoder.Session.Diagnostics;
var tokenizerAsset = new Asset(E5ModelAssets.TokenizerFile, E5ModelAssets.TokenizerSha256);
var modelAsset = new Asset(E5ModelAssets.ModelFile, diagnostics.ModelSha256 ?? ToolSupport.Sha256(files.ModelPath));

// Stage 1: tokenizer.
var tokenizerResults = new StageWriter(profile, "tokenizer", "Starlings.Tokenizers via E5Tokenizer (profile xlmr-unigram-e5, right truncation to 512)", commit, StageEnvironment(diagnostics), [tokenizerAsset]);
foreach (var c in cases)
{
    var input = tokenizer.Tokenize(texts[c.Id]);
    tokenizerResults.Case(c.Id).Add("input_ids", input.InputIds.ToArray()).Add("attention_mask", input.AttentionMask.ToArray());
}

tokenizerResults.Write(Path.Combine(outDir, "tokenizer.results.json"));

// Stage 3: .NET, from ids and from text.
var dotnet = new StageWriter(profile, "dotnet", $"C# E5Encoder + E5TextEmbedder, ONNX Runtime {diagnostics.OrtVersion} CPU (managed binding {diagnostics.ManagedBindingVersion})", commit, StageEnvironment(diagnostics), [modelAsset, tokenizerAsset]);
foreach (var c in cases)
{
    dotnet.Case(c.Id)
        .Add("input_ids", c.InputIds)
        .Add("embedding", encoder.Embed(new EncoderInput(c.InputIds)))
        .Add("text_embedding", embedder.Embed(texts[c.Id]));
}

dotnet.Batch("fixture-padded", "embedding", batchRows.Select(r => r.Id).ToList(), encoder.EmbedBatch(batchRows.Select(r => new EncoderInput(r.Ids, r.AttentionMask)).ToList()));
var batched = new List<float[]>();
for (int start = 0; start < cases.Count; start += 16)
{
    batched.AddRange(encoder.EmbedBatch(cases.Skip(start).Take(16).Select(c => new EncoderInput(c.InputIds)).ToList()));
}

dotnet.Batch("all-cases-batch16", "embedding", cases.Select(c => c.Id).ToList(), batched);
dotnet.Batch("text-batch", "text_embedding", cases.Select(c => c.Id).ToList(), embedder.EmbedBatch(cases.Select(c => texts[c.Id]).ToList()));

if (hidden)
{
    dotnet.Tensor("last_hidden_state", outDir, "dotnet-last-hidden-state.f32", cases.Select(c => (c.Id, HiddenState(encoder.Session, c.InputIds))));
}

dotnet.Write(Path.Combine(outDir, "dotnet.results.json"));
Console.WriteLine($"wrote {Path.Combine(outDir, "tokenizer.results.json")} and dotnet.results.json: {cases.Count} cases");
return 0;

static (int[] Shape, float[] Values) HiddenState(OnnxModelSession session, long[] ids)
{
    long[] shape = [1, ids.Length];
    long[] mask = Enumerable.Repeat(1L, ids.Length).ToArray();
    long[] types = new long[ids.Length];
    using var idsValue = OrtValue.CreateTensorValueFromMemory(ids, shape);
    using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, shape);
    using var typesValue = OrtValue.CreateTensorValueFromMemory(types, shape);
    using var outputs = session.Run(
        [E5Encoder.InputIdsName, E5Encoder.AttentionMaskName, E5Encoder.TokenTypeIdsName],
        [idsValue, maskValue, typesValue],
        [E5Encoder.LastHiddenStateName]);
    var values = outputs[0].GetTensorDataAsSpan<float>().ToArray();
    return ([ids.Length, values.Length / ids.Length], values);
}

static Dictionary<string, object?> StageEnvironment(OnnxSessionDiagnostics d) => new()
{
    ["machine"] = System.Environment.MachineName,
    ["provider"] = "cpu",
    ["provider_options"] = d.ProviderOptions.ToDictionary(kv => kv.Key, kv => (object?)kv.Value),
    ["dotnet"] = RuntimeInformation.FrameworkDescription,
    ["os"] = RuntimeInformation.OSDescription,
    ["process_architecture"] = RuntimeInformation.ProcessArchitecture.ToString(),
    ["cpu_count"] = System.Environment.ProcessorCount,
    ["cpu"] = ToolSupport.CpuModel(),
    ["onnxruntime"] = d.OrtVersion,
    ["onnxruntime_managed"] = d.ManagedBindingVersion,
    ["session_providers"] = d.SessionProviders,
    ["intra_op_threads"] = d.IntraOpNumThreads,
    ["graph_optimization_level"] = d.GraphOptimizationLevel,
};

static int Usage(string? error = null)
{
    if (error is not null)
    {
        Console.Error.WriteLine($"error: {error}");
    }

    Console.Error.WriteLine("usage: Starlings.Fidelity.Tool e5 --out <dir> [--snapshot <dir>] [--fixtures <dir>] [--profile <id>] [--hidden] [--threads <n>]");
    Console.Error.WriteLine("       Starlings.Fidelity.Tool gliner2 --out <dir> [--snapshot <dir>] [--fixtures <dir>] [--profile <id>] [--threads <n>]");
    Console.Error.WriteLine("       Starlings.Fidelity.Tool embeddinggemma2 --out <dir> [--snapshot <dir>] [--fixtures <dir>] [--profile <id>] [--threads <n>]");
    return error is null ? 0 : 2;
}

internal sealed record Asset(string Path, string Sha256);

internal sealed record FixtureCase(string Id, long[] InputIds);

internal sealed record BatchRow(string Id, long[] Ids, long[] AttentionMask);

/// <summary>Read-only views of the protected e5 fixtures.</summary>
internal static class Fixtures
{
    public static List<FixtureCase> LoadCases(string dir) =>
        ReadJsonLines(Path.Combine(dir, "embeddings.jsonl"))
            .Select(r => new FixtureCase(r.GetProperty("id").GetString()!, Longs(r.GetProperty("input_ids"))))
            .ToList();

    public static Dictionary<string, string> LoadTexts(string dir) =>
        ReadJsonLines(Path.Combine(dir, "tokenizer.jsonl"))
            .Where(r => !r.TryGetProperty("pair", out _))
            .ToDictionary(r => r.GetProperty("id").GetString()!, r => r.GetProperty("text").GetString()!, StringComparer.Ordinal);

    public static List<BatchRow> LoadBatchRows(string dir)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "tokenizer-batch.json")));
        return doc.RootElement.GetProperty("rows").EnumerateArray()
            .Select(r => new BatchRow(r.GetProperty("id").GetString()!, Longs(r.GetProperty("ids")), Longs(r.GetProperty("attention_mask"))))
            .ToList();
    }

    private static long[] Longs(JsonElement array) => array.EnumerateArray().Select(e => e.GetInt64()).ToArray();

    // Split on LF only: fixture text contains U+2028 and U+0085, which other line readers may treat as breaks.
    private static List<JsonElement> ReadJsonLines(string path) =>
        File.ReadAllText(path, Encoding.UTF8).Split('\n')
            .Where(l => l.Trim().Length > 0)
            .Select(l => JsonDocument.Parse(l).RootElement.Clone())
            .ToList();
}

/// <summary>
/// Builds one stage results file: format version 1 (numeric outputs, embedding) when <paramref name="task"/> is null,
/// otherwise version 2 with the task and typed outputs (strings, string lists, entity results). Floats are written as
/// the shortest round-trip decimal.
/// </summary>
internal sealed class StageWriter(string profile, string stage, string candidate, string? commit, Dictionary<string, object?> environment, Asset[] assets, string? task = null)
{
    private readonly Dictionary<string, CaseOutputs> _cases = new(StringComparer.Ordinal);
    private readonly List<(string Name, string Output, List<string> Ids, IReadOnlyList<object> Rows)> _batches = [];
    private readonly List<(string Name, string File, string Sha256, List<(string Id, int[] Shape)> Cases)> _tensors = [];

    public CaseOutputs Case(string id)
    {
        if (!_cases.TryGetValue(id, out var outputs))
        {
            _cases[id] = outputs = new CaseOutputs();
        }

        return outputs;
    }

    public void Batch(string name, string output, List<string> ids, IReadOnlyList<float[]> rows) => _batches.Add((name, output, ids, rows.Cast<object>().ToList()));

    public void Batch(string name, string output, List<string> ids, IReadOnlyList<EntityResult> rows) => _batches.Add((name, output, ids, rows.Cast<object>().ToList()));

    public void Tensor(string name, string dir, string file, IEnumerable<(string Id, (int[] Shape, float[] Values) Data)> perCase)
    {
        if (!BitConverter.IsLittleEndian)
        {
            throw new PlatformNotSupportedException("Tensor sidecars are little-endian float32.");
        }

        var layout = new List<(string, int[])>();
        using var stream = File.Create(Path.Combine(dir, file));
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var (id, (shape, values)) in perCase)
        {
            var bytes = MemoryMarshal.AsBytes(values.AsSpan());
            stream.Write(bytes);
            hash.AppendData(bytes);
            layout.Add((id, shape));
        }

        _tensors.Add((name, file, Convert.ToHexStringLower(hash.GetHashAndReset()), layout));
    }

    public void Write(string path)
    {
        using var stream = File.Create(path);
        using var w = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = false });
        w.WriteStartObject();
        w.WriteString("format", "starlings.stage-results");
        w.WriteNumber("format_version", task is null ? 1 : 2);
        if (task is not null)
        {
            w.WriteString("task", task);
        }

        w.WriteString("profile", profile);
        w.WriteString("stage", stage);
        w.WriteString("candidate", candidate);
        w.WriteStartObject("producer");
        w.WriteString("name", "ai-dotnet/tools/Starlings.Fidelity.Tool");
        w.WriteString("repository_commit", commit);
        w.WriteEndObject();
        w.WritePropertyName("environment");
        JsonSerializer.Serialize(w, environment);
        w.WriteStartArray("assets");
        foreach (var a in assets)
        {
            w.WriteStartObject();
            w.WriteString("path", a.Path);
            w.WriteString("sha256", a.Sha256);
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteStartObject("cases");
        foreach (var (id, outputs) in _cases)
        {
            w.WriteStartObject(id);
            foreach (var (name, value) in outputs.Items)
            {
                w.WritePropertyName(name);
                WriteValue(w, value);
            }

            w.WriteEndObject();
        }

        w.WriteEndObject();
        w.WriteStartObject("batches");
        foreach (var (name, output, ids, rows) in _batches)
        {
            w.WriteStartObject(name);
            w.WriteString("output", output);
            w.WriteStartArray("case_ids");
            ids.ForEach(w.WriteStringValue);
            w.WriteEndArray();
            w.WriteStartArray("values");
            foreach (var row in rows)
            {
                WriteValue(w, row);
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        w.WriteEndObject();
        w.WriteStartObject("tensors");
        foreach (var (name, file, sha, layout) in _tensors)
        {
            w.WriteStartObject(name);
            w.WriteString("file", file);
            w.WriteString("dtype", "float32");
            w.WriteString("sha256", sha);
            w.WriteStartArray("cases");
            foreach (var (id, shape) in layout)
            {
                w.WriteStartObject();
                w.WriteString("id", id);
                w.WriteStartArray("shape");
                Array.ForEach(shape, w.WriteNumberValue);
                w.WriteEndArray();
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        w.WriteEndObject();
        w.WriteEndObject();
    }

    private void WriteValue(Utf8JsonWriter w, object value)
    {
        if (task is null && value is not (long[] or float[]))
        {
            throw new InvalidOperationException("Format version 1 allows numeric outputs only.");
        }

        switch (value)
        {
            case long[] longs:
                w.WriteStartArray();
                Array.ForEach(longs, w.WriteNumberValue);
                w.WriteEndArray();
                break;
            case float[] floats:
                WriteFloats(w, floats);
                break;
            case string text:
                w.WriteStringValue(text);
                break;
            case string[] strings:
                w.WriteStartArray();
                Array.ForEach(strings, w.WriteStringValue);
                w.WriteEndArray();
                break;
            case EntityResult result:
                w.WriteStartObject();
                w.WriteBoolean("gate_open", result.GateOpen);
                w.WriteStartArray("entities");
                foreach (var e in result.Entities)
                {
                    w.WriteStartObject();
                    w.WriteString("label", e.Label);
                    w.WriteString("text", e.Text);
                    w.WriteNumber("start", e.Start);
                    w.WriteNumber("end", e.End);
                    w.WritePropertyName("confidence");
                    WriteFloat(w, e.Confidence);
                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WriteEndObject();
                break;
            default:
                throw new InvalidOperationException($"Unsupported output type {value.GetType()}.");
        }
    }

    private static void WriteFloats(Utf8JsonWriter w, float[] values)
    {
        w.WriteStartArray();
        foreach (float v in values)
        {
            WriteFloat(w, v);
        }

        w.WriteEndArray();
    }

    private static void WriteFloat(Utf8JsonWriter w, float v)
    {
        if (float.IsFinite(v))
        {
            // Shortest decimal that round-trips to the same float32, as the Python fixtures write.
            w.WriteRawValue(v.ToString("R", CultureInfo.InvariantCulture), skipInputValidation: true);
        }
        else
        {
            // JSON has no NaN or Inf; Python's json reads these literals, and the evaluator fails them.
            w.WriteRawValue(float.IsNaN(v) ? "NaN" : v > 0 ? "Infinity" : "-Infinity", skipInputValidation: true);
        }
    }
}

/// <summary>One entity of an entity result (format version 2): offsets in the profile's offsets unit, exclusive end.</summary>
internal sealed record EntityItem(string Label, string Text, int Start, int End, float Confidence);

/// <summary>An entity result (format version 2): the gate decision and the items in output order.</summary>
internal sealed record EntityResult(bool GateOpen, IReadOnlyList<EntityItem> Entities);

internal sealed class CaseOutputs
{
    public List<(string Name, object Value)> Items { get; } = [];

    public CaseOutputs Add(string name, long[] values)
    {
        Items.Add((name, values));
        return this;
    }

    public CaseOutputs Add(string name, float[] values)
    {
        Items.Add((name, values));
        return this;
    }

    public CaseOutputs Add(string name, string value)
    {
        Items.Add((name, value));
        return this;
    }

    public CaseOutputs Add(string name, string[] values)
    {
        Items.Add((name, values));
        return this;
    }

    public CaseOutputs Add(string name, EntityResult value)
    {
        Items.Add((name, value));
        return this;
    }
}
