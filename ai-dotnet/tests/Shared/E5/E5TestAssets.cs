using System.Globalization;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.E5;

namespace Starlings.TestSupport;

/// <summary>
/// Locates the repository, the protected e5 fixtures (read only) and the e5 ONNX model, which is not in Git.
/// </summary>
internal static class E5TestAssets
{
    /// <summary>Optional: a snapshot directory holding <c>onnx/model.onnx</c> and <c>tokenizer.json</c> of the pinned commit.</summary>
    public const string SnapshotEnvironmentVariable = "STARLING_E5_SNAPSHOT";

    public const string Revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3";

    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    public static string CpuFixtureDirectory => Path.Combine(RepositoryRoot, "model-pipeline", "protected", "multilingual-e5-small", "onnx-fp32-cpu");

    public static string CudaFixtureDirectory => Path.Combine(RepositoryRoot, "model-pipeline", "protected", "multilingual-e5-small", "onnx-fp32-cuda");

    public static string ManifestPath(string profile) => Path.Combine(RepositoryRoot, "model-pipeline", "registry", "multilingual-e5-small", profile, "manifest.yaml");

    /// <summary>
    /// Cache root used by the tests: <c>$STARLING_MODELS</c>, else <c>&lt;repo&gt;/.models</c> (the layout shared with
    /// <c>tools/fetch-tokenizer-assets.sh</c> and <c>ai-dotnet/tools/fetch-e5-model.sh</c>).
    /// </summary>
    public static string ModelsRoot =>
        Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(RepositoryRoot, ".models");

    /// <summary>SHA-256 of <c>onnx/model.onnx</c>, read from the CPU profile manifest.</summary>
    public static string ModelSha256 { get; } = ReadManifestSha256("onnx-fp32-cpu", "onnx/model.onnx");

    /// <summary>
    /// The e5 files through the acquisition library (offline; never downloads), or null with the reason they are unavailable.
    /// Files are checked by size here; the session verifies the model's SHA-256 itself. A present file of the wrong size
    /// throws <see cref="ModelAssetIntegrityException"/> rather than skipping.
    /// </summary>
    public static (E5ModelFiles? Files, string Reason) FindFiles()
    {
        string? snapshot = Environment.GetEnvironmentVariable(SnapshotEnvironmentVariable);
        try
        {
            var files = string.IsNullOrEmpty(snapshot)
                ? E5ModelAssets.OpenLocal(new ModelAcquisitionOptions { CacheDirectory = ModelsRoot, Verification = CacheVerification.Size })
                : E5ModelAssets.OpenDirectory(snapshot, CacheVerification.Size);
            return (files, string.Empty);
        }
        catch (ModelAssetMissingException ex)
        {
            return (null, ex.Message + $" Alternatively set {SnapshotEnvironmentVariable} to an existing snapshot directory, or run ai-dotnet/tools/fetch-e5-model.sh.");
        }
    }

    /// <summary>Model path, or null with the reason it is unavailable.</summary>
    public static (string? Path, string Reason) FindModel()
    {
        var (files, reason) = FindFiles();
        return (files?.ModelPath, reason);
    }

    /// <summary>Reads <c>sha256</c> for <paramref name="file"/> under <c>origin.files</c> in a profile manifest.</summary>
    public static string ReadManifestSha256(string profile, string file)
    {
        var lines = File.ReadAllLines(ManifestPath(profile));
        for (int i = 0; i < lines.Length - 1; i++)
        {
            if (lines[i].Trim() == $"- path: {file}")
            {
                string next = lines[i + 1].Trim();
                if (next.StartsWith("sha256:", StringComparison.Ordinal))
                {
                    return next["sha256:".Length..].Trim().Trim('"');
                }
            }
        }

        throw new InvalidOperationException($"No sha256 for {file} in {ManifestPath(profile)}.");
    }

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

/// <summary>One single-text case from <c>embeddings.jsonl</c>.</summary>
internal sealed record E5Case(string Id, long[] InputIds, float[] Embedding);

/// <summary>One row of <c>tokenizer-batch.json</c>: right-padded ids and mask.</summary>
internal sealed record E5BatchRow(string Id, long[] Ids, long[] AttentionMask);

/// <summary>Read-only views of the protected fixtures.</summary>
internal static class E5Fixtures
{
    private static readonly Lazy<IReadOnlyList<E5Case>> s_cases = new(LoadCases);
    private static readonly Lazy<(long PadId, IReadOnlyList<E5BatchRow> Rows)> s_batch = new(LoadBatch);
    private static readonly Lazy<IReadOnlyDictionary<string, string>> s_texts = new(LoadTexts);

    public static IReadOnlyList<E5Case> Cases => s_cases.Value;

    public static long BatchPadId => s_batch.Value.PadId;

    public static IReadOnlyList<E5BatchRow> BatchRows => s_batch.Value.Rows;

    /// <summary>Case id to source text, from <c>tokenizer.jsonl</c>.</summary>
    public static IReadOnlyDictionary<string, string> Texts => s_texts.Value;

    private static List<E5Case> LoadCases()
    {
        var result = new List<E5Case>();
        foreach (var line in File.ReadLines(Path.Combine(E5TestAssets.CpuFixtureDirectory, "embeddings.jsonl")))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            result.Add(new E5Case(
                root.GetProperty("id").GetString()!,
                root.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
                root.GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle()).ToArray()));
        }

        return result;
    }

    private static (long, IReadOnlyList<E5BatchRow>) LoadBatch()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(E5TestAssets.CpuFixtureDirectory, "tokenizer-batch.json")));
        var root = doc.RootElement;
        var padding = root.GetProperty("padding");
        if (padding.GetProperty("side").GetString() != "right")
        {
            throw new InvalidOperationException("tokenizer-batch.json is not right-padded.");
        }

        var rows = root.GetProperty("rows").EnumerateArray().Select(r => new E5BatchRow(
            r.GetProperty("id").GetString()!,
            r.GetProperty("ids").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
            r.GetProperty("attention_mask").EnumerateArray().Select(e => e.GetInt64()).ToArray())).ToList();
        return (padding.GetProperty("pad_id").GetInt64(), rows);
    }

    private static Dictionary<string, string> LoadTexts()
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var line in File.ReadLines(Path.Combine(E5TestAssets.CpuFixtureDirectory, "tokenizer.jsonl")))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            using var doc = JsonDocument.Parse(line);
            result[doc.RootElement.GetProperty("id").GetString()!] = doc.RootElement.GetProperty("text").GetString()!;
        }

        return result;
    }
}

/// <summary>Binding tolerances read from a protected <c>tolerances.yaml</c> (never edited by tests).</summary>
internal sealed record E5Tolerances(double EmbeddingMaxAbs, double PaddedBatchVsSingleMaxAbs, string Source)
{
    public static E5Tolerances Load(string fixtureDirectory)
    {
        string path = Path.Combine(fixtureDirectory, "tolerances.yaml");
        var values = ReadSection(File.ReadAllLines(path), "float");
        return new E5Tolerances(
            Require(values, "embedding_max_abs", path),
            Require(values, "padded_batch_vs_single_embedding_max_abs", path),
            path);
    }

    /// <summary>Reads <c>key: number</c> pairs directly under a top-level <c>section:</c> of a flat YAML file.</summary>
    private static Dictionary<string, string> ReadSection(string[] lines, string section)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        bool inside = false;
        foreach (var raw in lines)
        {
            string line = raw.Split('#')[0].TrimEnd();
            if (line.Length == 0)
            {
                continue;
            }

            if (!char.IsWhiteSpace(line[0]))
            {
                inside = line == section + ":";
                continue;
            }

            int colon = line.IndexOf(':', StringComparison.Ordinal);
            if (inside && colon > 0)
            {
                result[line[..colon].Trim()] = line[(colon + 1)..].Trim();
            }
        }

        return result;
    }

    private static double Require(Dictionary<string, string> values, string key, string path) =>
        values.TryGetValue(key, out var text) && double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v)
            ? v
            : throw new InvalidOperationException($"{path}: float.{key} missing or not a number.");
}

/// <summary>Absolute-error summary, as the M0 reports compute it.</summary>
internal sealed class ErrorStats
{
    private readonly List<double> _abs = [];
    private double _minCosine = double.PositiveInfinity;
    private bool _nonFinite;

    public int Count => _abs.Count;

    public double Max => _abs.Count == 0 ? 0 : _abs.Max();

    public double Mean => _abs.Count == 0 ? 0 : _abs.Average();

    public double MinCosine => _minCosine;

    public bool NonFinite => _nonFinite;

    public double P99
    {
        get
        {
            if (_abs.Count == 0)
            {
                return 0;
            }

            var sorted = _abs.Order().ToArray();
            double pos = (sorted.Length - 1) * 0.99;
            int lo = (int)Math.Floor(pos);
            int hi = (int)Math.Ceiling(pos);
            return sorted[lo] + ((sorted[hi] - sorted[lo]) * (pos - lo));
        }
    }

    public void Add(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        if (actual.Length != expected.Length)
        {
            throw new ArgumentException("Length mismatch.");
        }

        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            _nonFinite |= !float.IsFinite(actual[i]);
            _abs.Add(Math.Abs((double)actual[i] - expected[i]));
            dot += (double)actual[i] * expected[i];
            na += (double)actual[i] * actual[i];
            nb += (double)expected[i] * expected[i];
        }

        _minCosine = Math.Min(_minCosine, dot / (Math.Sqrt(na) * Math.Sqrt(nb)));
    }

    public override string ToString() => string.Create(
        CultureInfo.InvariantCulture,
        $"max {Max:E3}, mean {Mean:E3}, p99 {P99:E3}, min cosine {MinCosine:F14}, values {Count}, non-finite {NonFinite}");
}
