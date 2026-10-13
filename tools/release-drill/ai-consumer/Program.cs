// Clean-consumer check for the Starlings release-candidate drill (P-010).
// tools/release-drill-ai.sh copies this file into a temporary directory outside the repository and builds it against
// the packed packages from a local folder feed. It must compile and run with nothing but those packages and the
// ONNX Runtime CPU natives: no project references, no repository paths, no test framework.
//
//   dotnet AiConsumer.dll tokenizer <cache> <fixture dir>
//       Network: downloads tokenizer.json of the e5 lock into the empty cache, then offline: resolves it, checks the
//       owned tokenizer's ids against the protected fixtures and that the ONNX Runtime natives load.
//   dotnet AiConsumer.dll full <cache> <fixture dir> <embedding_max_abs> <padded_batch_vs_single_max_abs>
//       Offline only: resolves the full e5 lock from the cache and checks embeddings (typed API one per call and in
//       one batch, the Microsoft.Extensions.AI generator, and the ML.NET transform row-wise and batched) against the
//       protected fixtures within the bounds, which the drill reads from the protected tolerances file.
//
// Fixture dir: embeddings.jsonl and tokenizer.jsonl from model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu.
// Exit code 0 only if every check passes.
using System.Globalization;
using System.Reflection;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Models.E5.Extensions.AI;
using Starlings.Models.E5.MLNet;
using Starlings.Runtime.Onnx;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;

if (args.Length < 3 || (args[0], args.Length) is not (("tokenizer", 3) or ("full", 5)))
{
    Console.Error.WriteLine("usage: tokenizer <cache> <fixture dir> | full <cache> <fixture dir> <embedding_max_abs> <padded_batch_vs_single_max_abs>");
    return 2;
}

string mode = args[0];
string cache = args[1];
string fixtureDir = args[2];
const int ExpectedCases = 46;
const string ExpectedOrt = "1.30.0";

int failures = 0;
void Check(bool ok, string what)
{
    Console.WriteLine($"{(ok ? "ok:  " : "FAIL:")} {what}");
    if (!ok)
    {
        failures++;
    }
}

// Where the code comes from: package versions and the files actually loaded.
foreach (Type t in new[] { typeof(ModelAssetStore), typeof(OnnxModelSession), typeof(Starlings.Tokenizers.Tokenizer), typeof(E5Encoder), typeof(E5EmbeddingGenerator), typeof(E5EmbeddingTransformer), typeof(MLContext) })
{
    Assembly a = t.Assembly;
    Console.WriteLine($"assembly: {a.GetName().Name} {a.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion} ({a.Location})");
}

Console.WriteLine($"runtime:  {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription} {System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier}");

// Native dependency resolution (G6): the CPU natives from Microsoft.ML.OnnxRuntime load and match the managed binding.
Check(OnnxRuntimeInfo.ManagedVersion == ExpectedOrt && OnnxRuntimeInfo.NativeVersion == ExpectedOrt,
    $"ONNX Runtime native {OnnxRuntimeInfo.NativeVersion}, managed {OnnxRuntimeInfo.ManagedVersion} (expected {ExpectedOrt})");
Check(OnnxRuntimeInfo.AvailableProviders.Contains(OnnxRuntimeInfo.CpuProviderName),
    $"providers: {string.Join(", ", OnnxRuntimeInfo.AvailableProviders)}");

// Fixtures, read with a split on '\n' only (corpus text contains U+0085 and U+2028).
var texts = new Dictionary<string, string>();
foreach (string line in File.ReadAllText(Path.Combine(fixtureDir, "tokenizer.jsonl")).Split('\n'))
{
    if (line.Length > 0)
    {
        JsonElement row = JsonDocument.Parse(line).RootElement;
        texts[row.GetProperty("id").GetString()!] = row.GetProperty("text").GetString()!;
    }
}

var cases = new List<(string Id, string Text, long[] Ids, float[] Embedding)>();
foreach (string line in File.ReadAllText(Path.Combine(fixtureDir, "embeddings.jsonl")).Split('\n'))
{
    if (line.Length > 0)
    {
        JsonElement row = JsonDocument.Parse(line).RootElement;
        string id = row.GetProperty("id").GetString()!;
        cases.Add((id, texts[id],
            [.. row.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64())],
            [.. row.GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle())]));
    }
}

Check(cases.Count == ExpectedCases, $"fixture cases: {cases.Count} (expected {ExpectedCases})");

var options = new ModelAcquisitionOptions { CacheDirectory = cache, UseEnvironmentToken = false };
string tokenizerPath;
E5ModelFiles? files = null;
if (mode == "tokenizer")
{
    // A lock with only tokenizer.json, taken from the shipped e5 lock (same repository, commit, size and digest).
    var tokenizerLock = new ModelAssetLock("multilingual-e5-small-tokenizer", E5ModelAssets.Repository, E5ModelAssets.Revision,
        [E5ModelAssets.Lock.GetFile(E5ModelAssets.TokenizerFile)]);
    var progress = new FileProgress();
    using (var store = new ModelAssetStore(options))
    {
        await store.DownloadAsync(tokenizerLock, progress);
        Check(progress.Downloaded.Contains(E5ModelAssets.TokenizerFile),
            $"tokenizer.json downloaded from {options.Endpoint}{E5ModelAssets.Repository}/resolve/{E5ModelAssets.Revision}/ and verified ({E5ModelAssets.Lock.GetFile(E5ModelAssets.TokenizerFile).Size} bytes, sha256 {E5ModelAssets.TokenizerSha256})");
        ModelSnapshot local = store.OpenLocal(tokenizerLock);   // offline resolution, verified
        tokenizerPath = local.GetFilePath(E5ModelAssets.TokenizerFile);
    }
}
else
{
    files = E5ModelAssets.OpenLocal(options);                   // offline, size and SHA-256 of both files
    tokenizerPath = files.TokenizerPath;
}

Console.WriteLine($"tokenizer.json: {tokenizerPath}");
var tokenizer = E5Tokenizer.FromFile(tokenizerPath, E5ModelAssets.TokenizerSha256);
int idsExact = cases.Count(c =>
{
    EncoderInput input = tokenizer.Tokenize(c.Text);
    // An empty mask means all ones (EncoderInput); e5 fixtures have no padding.
    return input.InputIds.Span.SequenceEqual(c.Ids) && input.AttentionMask.ToArray().All(m => m == 1);
});
Check(idsExact == cases.Count, $"owned tokenizer ids and masks exact: {idsExact}/{cases.Count}");

if (files is not null)
{
    double embeddingBound = double.Parse(args[3], CultureInfo.InvariantCulture);
    double batchBound = double.Parse(args[4], CultureInfo.InvariantCulture);
    using var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
    {
        Session = new OnnxSessionOptions { ExpectedModelSha256 = E5ModelAssets.ModelSha256 },
    });
    var embedder = new E5TextEmbedder(tokenizer, encoder);
    Console.WriteLine($"session: {string.Join(",", encoder.Session.Diagnostics.SessionProviders)}, model sha256 {encoder.Session.Diagnostics.ModelSha256}");

    float[][] single = [.. cases.Select(c => embedder.Embed(c.Text))];
    float[][] batch = embedder.EmbedBatch([.. cases.Select(c => c.Text)]);

    using var generator = E5EmbeddingGenerator.Create(files);    // its own session; fixture texts carry their prefixes
    GeneratedEmbeddings<Embedding<float>> generated = await generator.GenerateAsync(cases.Select(c => c.Text));
    float[][] adapter = [.. generated.Select(e => e.Vector.ToArray())];

    Compare("typed API, one text per call vs protected", single, cases.Select(c => c.Embedding).ToArray(), embeddingBound);
    Compare("typed API, one batch of 46 vs protected", batch, cases.Select(c => c.Embedding).ToArray(), embeddingBound);
    Compare("typed API, batch vs own single rows", batch, single, batchBound);
    Compare("Extensions.AI generator vs protected", adapter, cases.Select(c => c.Embedding).ToArray(), embeddingBound);
    Compare("Extensions.AI generator vs typed batch (report)", adapter, batch, double.PositiveInfinity);
    long tokens = cases.Sum(c => (long)c.Ids.Length);
    Check(generated.Usage?.InputTokenCount == tokens, $"Extensions.AI usage: {generated.Usage?.InputTokenCount} input tokens (fixture ids: {tokens})");

    // ML.NET: the same embedder in a pipeline, row-wise (equal to one text per call) and with look-ahead batching.
    var ml = new MLContext(seed: 1);
    IDataView view = ml.Data.LoadFromEnumerable(cases.Select(c => new TextRow { Text = c.Text }).ToList());
    using var rowWise = ml.Transforms.ApplyE5Embedding(embedder, "Embedding", "Text").Fit(view);
    float[][] mlnet = [.. rowWise.Transform(view).GetColumn<float[]>("Embedding")];
    using var batched = ml.Transforms.ApplyE5Embedding(embedder, "Embedding", "Text", new E5EmbeddingOptions { BatchSize = 46 }).Fit(view);
    float[][] mlnetBatched = [.. batched.Transform(view).GetColumn<float[]>("Embedding")];
    Compare("ML.NET transform, row-wise, vs typed API one per call (exact)", mlnet, single, 0);
    Compare("ML.NET transform vs protected", mlnet, cases.Select(c => c.Embedding).ToArray(), embeddingBound);
    Compare("ML.NET transform, look-ahead batch of 46, vs typed batch (exact)", mlnetBatched, batch, 0);
    bool saveRejected = false;
    try
    {
        ml.Model.Save(rowWise, view.Schema, new MemoryStream());
    }
    catch (Exception ex) when (ex is NotSupportedException || ex.InnerException is NotSupportedException)
    {
        saveRejected = true;
    }

    Check(saveRejected, "ML.NET Save of the e5 transformer is rejected (persistence unsupported)");
}

// No other tokenizer implementation in the process: no Microsoft.ML.Tokenizers assembly, no native tokenizers library.
string[] foreignManaged = [.. AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetName().Name ?? "")
    .Where(n => n.Contains("Tokenizer", StringComparison.OrdinalIgnoreCase) && n != "Starlings.Tokenizers")];
Check(foreignManaged.Length == 0, $"no other managed tokenizer assembly loaded ({string.Join(", ", foreignManaged)})");
if (File.Exists("/proc/self/maps"))
{
    string[] foreignNative = [.. File.ReadAllLines("/proc/self/maps").Select(l => l.Split(' ', StringSplitOptions.RemoveEmptyEntries).LastOrDefault() ?? "")
        .Where(p => p.Contains("tokenizers", StringComparison.OrdinalIgnoreCase) && !p.EndsWith("Starlings.Tokenizers.dll", StringComparison.Ordinal))
        .Distinct()];
    Check(foreignNative.Length == 0, $"no native tokenizers library mapped ({string.Join(", ", foreignNative)})");
}

Console.WriteLine(failures == 0 ? "RESULT: PASS" : $"RESULT: FAIL ({failures})");
return failures == 0 ? 0 : 1;

void Compare(string what, float[][] actual, float[][] expected, double bound)
{
    double max = 0, sum = 0, minCos = 1;
    long count = 0;
    bool finite = true;
    for (int r = 0; r < actual.Length; r++)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < actual[r].Length; i++)
        {
            finite &= float.IsFinite(actual[r][i]);
            double d = Math.Abs((double)actual[r][i] - expected[r][i]);
            max = Math.Max(max, d);
            sum += d;
            count++;
            dot += (double)actual[r][i] * expected[r][i];
            na += (double)actual[r][i] * actual[r][i];
            nb += (double)expected[r][i] * expected[r][i];
        }

        minCos = Math.Min(minCos, dot / Math.Sqrt(na * nb));
    }

    string boundText = double.IsPositiveInfinity(bound) ? "report only" : $"bound {bound:G3}";
    Check(finite && actual.Length == expected.Length && max <= bound,
        $"{what}: {actual.Length} rows, max abs {max:G3}, mean abs {sum / count:G3}, min cosine {minCos:F14} ({boundText}){(finite ? "" : ", NaN/Inf")}");
}

/// <summary>Input row of the ML.NET check.</summary>
internal sealed class TextRow
{
    public string Text { get; set; } = string.Empty;
}

/// <summary>Records transferred files; called synchronously by the store.</summary>
internal sealed class FileProgress : IProgress<ModelAssetProgress>
{
    public HashSet<string> Downloaded { get; } = [];

    public void Report(ModelAssetProgress value)
    {
        if (value.Stage == ModelAssetStage.Downloading)
        {
            Downloaded.Add(value.Path);
        }
    }
}
