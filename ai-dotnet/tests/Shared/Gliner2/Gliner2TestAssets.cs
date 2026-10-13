using System.Text;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.Gliner2;

namespace Starlings.TestSupport;

/// <summary>
/// Locates the repository, the GLiNER2 fixtures (read only), the tokenizer and the converted ONNX graph (neither is in
/// Git). Tests that need an absent asset skip with the reason, so CI without models stays green.
/// </summary>
internal static class Gliner2TestAssets
{
    /// <summary>Optional: a directory holding <c>tokenizer.json</c> and <c>onnx/gliner2-entities-fp32.onnx</c>.</summary>
    public const string SnapshotEnvironmentVariable = "STARLING_GLINER2_SNAPSHOT";

    /// <summary>
    /// Optional: the directory of the extraction fixtures (<c>extraction.jsonl</c>). Default: the profile's protected
    /// fixture directory (accepted as development fixtures on 2026-10-10, PR #19); override only to test a
    /// read-only copy elsewhere.
    /// </summary>
    public const string FixturesEnvironmentVariable = "STARLING_GLINER2_FIXTURES";

    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    public static string ProtectedRoot => Path.Combine(RepositoryRoot, "model-pipeline", "protected", "gliner2-base-v1");

    /// <summary>Accepted tokenizer fixtures (on main since PR #4).</summary>
    public static string TokenizerFixturePath => Path.Combine(ProtectedRoot, "tokenizer", "gliner2-tokenizer.jsonl");

    public static string ExtractionFixtureDirectory =>
        Environment.GetEnvironmentVariable(FixturesEnvironmentVariable) is { Length: > 0 } env ? env : Path.Combine(ProtectedRoot, "onnx-fp32-cpu");

    public static string ManifestPath => Path.Combine(RepositoryRoot, "model-pipeline", "registry", "gliner2-base-v1", "onnx-fp32-cpu", "manifest.yaml");

    /// <summary>Cache root: <c>$STARLING_MODELS</c>, else <c>&lt;repo&gt;/.models</c>.</summary>
    public static string ModelsRoot =>
        Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(RepositoryRoot, ".models");

    /// <summary>The extraction fixture file, or null with the reason it is unavailable.</summary>
    public static (string? Path, string Reason) FindExtractionFixtures()
    {
        string path = Path.Combine(ExtractionFixtureDirectory, "extraction.jsonl");
        return File.Exists(path)
            ? (path, string.Empty)
            : (null, $"GLiNER2 extraction fixtures not found at {path}: the protected fixtures (PR #19) are missing from this checkout. " +
                     $"Set {FixturesEnvironmentVariable} to a read-only copy of them.");
    }

    /// <summary>The pinned tokenizer.json, or null with the reason it is unavailable.</summary>
    public static (string? Path, string Reason) FindTokenizer()
    {
        var candidates = new List<string>();
        if (Environment.GetEnvironmentVariable(SnapshotEnvironmentVariable) is { Length: > 0 } snapshot)
        {
            candidates.Add(Path.Combine(snapshot, Gliner2ModelAssets.TokenizerFile));
        }

        candidates.Add(Path.Combine(ModelsRoot, Gliner2ModelAssets.Repository, Gliner2ModelAssets.Revision, Gliner2ModelAssets.TokenizerFile));
        foreach (string path in candidates)
        {
            if (File.Exists(path))
            {
                return (path, string.Empty);
            }
        }

        return (null, $"tokenizer.json of {Gliner2ModelAssets.Repository}@{Gliner2ModelAssets.Revision} not found ({string.Join(", ", candidates)}); run tools/fetch-tokenizer-assets.sh.");
    }

    /// <summary>
    /// The model files through the acquisition library (offline; never downloads), or null with the reason. Files are
    /// checked by size here; the session verifies the graph's SHA-256 itself.
    /// </summary>
    public static (Gliner2ModelFiles? Files, string Reason) FindModel()
    {
        string? snapshot = Environment.GetEnvironmentVariable(SnapshotEnvironmentVariable);
        try
        {
            var files = string.IsNullOrEmpty(snapshot)
                ? Gliner2ModelAssets.OpenLocal(new ModelAcquisitionOptions { CacheDirectory = ModelsRoot, Verification = CacheVerification.Size })
                : Gliner2ModelAssets.OpenDirectory(snapshot, CacheVerification.Size);
            return (files, string.Empty);
        }
        catch (ModelAssetMissingException ex)
        {
            return (null, ex.Message + $" Alternatively set {SnapshotEnvironmentVariable} to a directory with tokenizer.json and {Gliner2ModelAssets.ModelFile} " +
                          "(export: model-pipeline/reference/gliner2/export_gliner2_onnx.py, then release_gliner2_onnx.py).");
        }
    }

    // Split on LF only: fixture text contains U+2028 and U+0085, which other line readers may treat as breaks.
    public static List<JsonElement> ReadJsonLines(string path) =>
        File.ReadAllText(path, Encoding.UTF8).Split('\n')
            .Where(l => l.Trim().Length > 0)
            .Select(l => JsonDocument.Parse(l).RootElement.Clone())
            .ToList();

    private static string FindRepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "protected")) &&
                Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        throw new InvalidOperationException("Repository root not found above " + AppContext.BaseDirectory);
    }
}

/// <summary>
/// Binding tolerances of a GLiNER2 profile, read from its protected <c>tolerances.yaml</c> (set by the maintainer on
/// 2026-10-10, PR #21; plan 10.1). Read only; a missing key fails the test rather than defaulting.
/// </summary>
internal sealed class Gliner2Tolerances
{
    private readonly Dictionary<string, string> _values;

    private Gliner2Tolerances(string source, Dictionary<string, string> values)
    {
        Source = source;
        _values = values;
    }

    public string Source { get; }

    public double CountLogitsMaxAbs => Number("float.count_logits_max_abs");

    /// <summary>span_logits over valid spans: |d| &lt;= bound * max(1, |ref|).</summary>
    public double SpanLogitsRelative => Number("float.span_logits_relative");

    public double ScoresMaxAbs => Number("float.scores_max_abs");

    public double ConfidenceMaxAbs => Number("float.confidence_max_abs");

    public double PaddedBatchVsSingleRelative => Number("float.padded_batch_vs_single_span_logits_relative");

    /// <summary>Order may differ only between items whose reference confidences differ by at most this.</summary>
    public double OrderTieWindow => Number("entities.order_swap_max_confidence_delta");

    /// <summary>A span may flip only when its reference score is within this of the threshold.</summary>
    public double ThresholdFlipWindow => Number("entities.threshold_flip_max_score_margin");

    /// <summary>Raw value of a flattened key (<c>section.key</c>).</summary>
    public string this[string key] => _values.TryGetValue(key, out var v) ? v : throw new KeyNotFoundException($"{key} is not in {Source}.");

    public static Gliner2Tolerances Load(string profile)
    {
        string path = Path.Combine(Gliner2TestAssets.ProtectedRoot, profile, "tolerances.yaml");
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        string section = string.Empty;
        foreach (string raw in File.ReadAllLines(path))
        {
            string line = raw.Split('#')[0].TrimEnd();
            if (line.Trim().Length == 0)
            {
                continue;
            }

            int colon = line.IndexOf(':', StringComparison.Ordinal);
            string key = line[..colon].Trim();
            string value = line[(colon + 1)..].Trim();
            if (!char.IsWhiteSpace(line[0]))
            {
                section = value.Length == 0 ? key : string.Empty;
                values[key] = value;
            }
            else
            {
                values[$"{section}.{key}"] = value;
            }
        }

        return new Gliner2Tolerances($"model-pipeline/protected/gliner2-base-v1/{profile}/tolerances.yaml", values);
    }

    private double Number(string key) => double.Parse(this[key], System.Globalization.CultureInfo.InvariantCulture);
}

/// <summary>One upstream entity: label, text, offsets in Unicode scalar values of the normalized text, confidence.</summary>
public sealed record Gliner2RefEntity(string Label, string Text, int Start, int End, double Confidence)
{
    public string Key => $"{Label}\u0001{Text}\u0001{Start}\u0001{End}";
}

/// <summary>One case of the protected <c>extraction.jsonl</c>.</summary>
public sealed record Gliner2Case(
    string Id,
    string Category,
    string Text,
    IReadOnlyList<string> LabelsRequested,
    IReadOnlyDictionary<string, string>? Descriptions,
    float Threshold,
    int? MaxLen,
    string NormalizedText,
    IReadOnlyList<string> Words,
    int[] StartMap,
    int[] EndMap,
    IReadOnlyList<string> SchemaTokens,
    IReadOnlyList<string> Labels,
    long[] InputIds,
    long[] WordIndex,
    long[] SchemaIndex,
    float[] CountLogits,
    bool GateOpen,
    float[] SpanLogitsValid,
    bool OutputEmpty,
    IReadOnlyList<Gliner2RefEntity> Entities)
{
    /// <summary>Why the case is outside the C# contract, or null when it is in.</summary>
    public string? OutOfContract =>
        MaxLen is not null ? "max_len word cap (unsupported)"
        : InputIds.Length > 512 ? $"{InputIds.Length} tokens > 512 (rejected by contract)"
        : null;

    /// <summary>Reference score of a valid span (sigmoid of the reference logit, float32).</summary>
    public double ReferenceScore(int labelIndex, int startWord, int width)
    {
        int n = Words.Count;
        int index = 0;
        for (int label = 0; label < labelIndex; label++)
        {
            index += ValidPerLabel(n);
        }

        for (int s = 0; s < startWord; s++)
        {
            index += Math.Min(8, n - s);
        }

        index += width;
        float logit = SpanLogitsValid[index];
        return 1f / (1f + MathF.Exp(-logit));
    }

    private static int ValidPerLabel(int n)
    {
        int count = 0;
        for (int s = 0; s < n; s++)
        {
            count += Math.Min(8, n - s);
        }

        return count;
    }
}

/// <summary>Read-only views of the GLiNER2 fixtures.</summary>
internal static class Gliner2Fixtures
{
    public static IReadOnlyList<Gliner2Case> LoadCases(string path) => Gliner2TestAssets.ReadJsonLines(path).Select(Parse).ToList();

    private static Gliner2Case Parse(JsonElement r)
    {
        var entities = new List<Gliner2RefEntity>();
        bool empty = true;
        if (r.GetProperty("output").TryGetProperty("entities", out var byLabel))
        {
            foreach (var label in byLabel.EnumerateObject())
            {
                empty = false;
                foreach (var e in label.Value.EnumerateArray())
                {
                    entities.Add(new Gliner2RefEntity(
                        label.Name, e.GetProperty("text").GetString()!, e.GetProperty("start").GetInt32(), e.GetProperty("end").GetInt32(), e.GetProperty("confidence").GetDouble()));
                }
            }
        }

        Dictionary<string, string>? descriptions = null;
        if (r.GetProperty("descriptions").ValueKind == JsonValueKind.Object)
        {
            descriptions = r.GetProperty("descriptions").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!, StringComparer.Ordinal);
        }

        return new Gliner2Case(
            r.GetProperty("id").GetString()!,
            r.GetProperty("category").GetString()!,
            r.GetProperty("text").GetString()!,
            Strings(r.GetProperty("labels_requested")),
            descriptions,
            (float)r.GetProperty("threshold").GetDouble(),
            r.GetProperty("max_len").ValueKind == JsonValueKind.Number ? r.GetProperty("max_len").GetInt32() : null,
            r.GetProperty("normalized_text").GetString()!,
            Strings(r.GetProperty("words")),
            r.GetProperty("start_map").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            r.GetProperty("end_map").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            Strings(r.GetProperty("schema_tokens")),
            Strings(r.GetProperty("labels")),
            Longs(r.GetProperty("input_ids")),
            Longs(r.GetProperty("word_index")),
            Longs(r.GetProperty("schema_index")),
            Floats(r.GetProperty("count_logits")),
            r.GetProperty("gate_open").GetBoolean(),
            Floats(r.GetProperty("span_logits_valid")),
            empty,
            entities);
    }

    private static List<string> Strings(JsonElement a) => a.EnumerateArray().Select(e => e.GetString()!).ToList();

    private static long[] Longs(JsonElement a) => a.EnumerateArray().Select(e => e.GetInt64()).ToArray();

    private static float[] Floats(JsonElement a) => a.EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
}
