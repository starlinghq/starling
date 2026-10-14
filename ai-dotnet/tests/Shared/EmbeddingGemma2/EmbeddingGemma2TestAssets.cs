using System.Globalization;
using System.Text;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.EmbeddingGemma2;

namespace Starlings.TestSupport;

/// <summary>
/// Locates the repository, the EmbeddingGemma 2 fixtures (read only) and the text-path graph (not in Git). Tests that
/// need an absent asset skip with the reason, so CI without models stays green.
/// </summary>
internal static class EmbeddingGemma2TestAssets
{
    /// <summary>
    /// Optional: a directory holding <c>tokenizer.json</c>, <c>config_sentence_transformers.json</c> and
    /// <c>onnx/embeddinggemma2-text-fp32.onnx</c>.
    /// </summary>
    public const string SnapshotEnvironmentVariable = "STARLING_EMBEDDINGGEMMA2_SNAPSHOT";

    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    public static string ProtectedRoot => Path.Combine(RepositoryRoot, "model-pipeline", "protected", "embeddinggemma-2");

    /// <summary>Accepted development fixtures (PR #29): tokenizer.jsonl, embeddings.jsonl, tokenizer-batch.json.</summary>
    public static string FixtureDirectory => Path.Combine(ProtectedRoot, "onnx-fp32-cpu");

    /// <summary>Cache root: <c>$STARLING_MODELS</c>, else <c>&lt;repo&gt;/.models</c>.</summary>
    public static string ModelsRoot =>
        Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(RepositoryRoot, ".models");

    /// <summary>
    /// The model files through the acquisition library (offline; never downloads), or null with the reason. Files are
    /// checked by size here; the session verifies the graph's SHA-256 itself.
    /// </summary>
    public static (EmbeddingGemma2ModelFiles? Files, string Reason) FindModel()
    {
        string? snapshot = Environment.GetEnvironmentVariable(SnapshotEnvironmentVariable);
        try
        {
            var files = string.IsNullOrEmpty(snapshot)
                ? EmbeddingGemma2ModelAssets.OpenLocal(new ModelAcquisitionOptions { CacheDirectory = ModelsRoot, Verification = CacheVerification.Size })
                : EmbeddingGemma2ModelAssets.OpenDirectory(snapshot, CacheVerification.Size);
            return (files, string.Empty);
        }
        catch (ModelAssetMissingException ex)
        {
            return (null, ex.Message + $" Alternatively set {SnapshotEnvironmentVariable} to a directory with tokenizer.json, config_sentence_transformers.json and {EmbeddingGemma2ModelAssets.ModelFile} " +
                          "(export: model-pipeline/reference/embeddinggemma2/export_embeddinggemma2_onnx.py, then release_embeddinggemma2_onnx.py).");
        }
    }

    /// <summary>
    /// The pinned tokenizer.json (from the snapshot directory if set, else the cache that
    /// <c>tools/fetch-tokenizer-assets.sh</c> fills), or null with the reason it is unavailable.
    /// </summary>
    public static (string? Path, string Reason) FindTokenizer()
    {
        var candidates = new List<string>();
        if (Environment.GetEnvironmentVariable(SnapshotEnvironmentVariable) is { Length: > 0 } snapshot)
        {
            candidates.Add(Path.Combine(snapshot, EmbeddingGemma2ModelAssets.TokenizerFile));
        }

        candidates.Add(Path.Combine(ModelsRoot, EmbeddingGemma2ModelAssets.Repository, EmbeddingGemma2ModelAssets.Revision, EmbeddingGemma2ModelAssets.TokenizerFile));
        foreach (string path in candidates)
        {
            if (File.Exists(path))
            {
                return (path, string.Empty);
            }
        }

        return (null, $"tokenizer.json of {EmbeddingGemma2ModelAssets.Repository}@{EmbeddingGemma2ModelAssets.Revision} not found ({string.Join(", ", candidates)}); run tools/fetch-tokenizer-assets.sh.");
    }

    private static readonly Lazy<EmbeddingGemma2Tokenizer?> SharedTokenizer = new(() =>
        FindTokenizer().Path is { } path ? EmbeddingGemma2Tokenizer.FromFile(path) : null);

    /// <summary>The owned tokenizer on the pinned file (loaded once per process); skips the calling test when absent.</summary>
    public static EmbeddingGemma2Tokenizer Tokenizer
    {
        get
        {
            var tokenizer = SharedTokenizer.Value;
            Assert.SkipWhen(tokenizer is null, FindTokenizer().Reason);
            return tokenizer!;
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
/// Binding tolerances of an EmbeddingGemma 2 profile, read from its protected <c>tolerances.yaml</c> (set by the
/// maintainer on 2026-10-10; plan 10.1). Read only; a missing key fails the test rather than defaulting.
/// </summary>
internal sealed class EmbeddingGemma2Tolerances
{
    private readonly Dictionary<string, string> _values;

    private EmbeddingGemma2Tolerances(string source, Dictionary<string, string> values)
    {
        Source = source;
        _values = values;
    }

    public string Source { get; }

    public double EmbeddingMaxAbs => Number("float.embedding_max_abs");

    public double PaddedBatchVsSingleMaxAbs => Number("float.padded_batch_vs_single_embedding_max_abs");

    /// <summary>Minimum cosine (float64) of an embedding against its reference.</summary>
    public double MinCosine => Number("float.min_cosine");

    public double CSharpVsPythonMaxAbs => Number("csharp_vs_python_same_machine_and_provider.embedding_max_abs");

    /// <summary>Image profiles: per image, max |soft token diff| / max |reference soft token| (set 2026-10-11).</summary>
    public double SoftTokensRowRelative => Number("float.soft_tokens_row_relative");

    /// <summary>Image profiles: per row, max |token embedding diff| / max |reference token embedding|.</summary>
    public double TokenEmbeddingsRowRelative => Number("float.token_embeddings_row_relative");

    /// <summary>Image profiles: a discrete check that must be <c>exact</c> (token_ids, resized_pixels, pixel_values, image_positions).</summary>
    public bool IsExact(string discreteKey) => this[$"discrete.{discreteKey}"] == "exact";

    /// <summary>Image profiles: the SkiaSharp decoder must reproduce PIL's decoded RGB exactly.</summary>
    public bool DecoderSkiaSharpExact => this["decoder_skiasharp.decoded_rgb_vs_pil"] == "exact";

    /// <summary>NaN or Inf fails the profile.</summary>
    public bool NanOrInfFails => this["float.nan_or_inf"] == "fail";

    /// <summary>Whether the protected file has a flattened key (for a bound the maintainer has not committed yet).</summary>
    public bool Contains(string key) => _values.ContainsKey(key);

    /// <summary>Raw value of a flattened key (<c>section.key</c>).</summary>
    public string this[string key] => _values.TryGetValue(key, out var v) ? v : throw new KeyNotFoundException($"{key} is not in {Source}.");

    public static EmbeddingGemma2Tolerances Load(string profile)
    {
        string path = Path.Combine(EmbeddingGemma2TestAssets.ProtectedRoot, profile, "tolerances.yaml");
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

        return new EmbeddingGemma2Tolerances($"model-pipeline/protected/embeddinggemma-2/{profile}/tolerances.yaml", values);
    }

    private double Number(string key) => double.Parse(this[key], CultureInfo.InvariantCulture);
}

/// <summary>One row of the protected <c>embeddings.jsonl</c>: a text under one prompt setting.</summary>
public sealed record EmbeddingGemma2Row(string Id, string Prompt, string PromptText, string Text, long[] InputIds, float[] Embedding)
{
    /// <summary>Unique key: the case id, plus <c>@prompt</c> for the query and document rows.</summary>
    public string Key => Prompt == "none" ? Id : $"{Id}@{Prompt}";

    /// <summary>The prompt as the package names it.</summary>
    public EmbeddingGemma2Prompt PackagePrompt => EmbeddingGemma2Prompt.FromName(Prompt);
}

/// <summary>Read-only views of the EmbeddingGemma 2 fixtures.</summary>
internal static class EmbeddingGemma2Fixtures
{
    public static IReadOnlyList<EmbeddingGemma2Row> LoadRows() => LoadRows(Path.Combine(EmbeddingGemma2TestAssets.FixtureDirectory, "embeddings.jsonl"));

    /// <summary>
    /// The task-prompt rows (capability <c>prompts_task</c>): four public texts x the 20 prompt names of
    /// <c>config_sentence_transformers.json</c>, from <c>run_embeddinggemma2_task_prompts.py</c>. Reads the protected copy
    /// when it exists and the proposal under <c>model-pipeline/proposed/</c> otherwise.
    /// </summary>
    public static IReadOnlyList<EmbeddingGemma2Row> LoadTaskPromptRows()
    {
        string accepted = Path.Combine(EmbeddingGemma2TestAssets.ProtectedRoot, "task-prompts", "task-prompt-embeddings.jsonl");
        string proposed = Path.Combine(EmbeddingGemma2TestAssets.RepositoryRoot, "model-pipeline", "proposed", "embeddinggemma-2", "task-prompts", "task-prompt-embeddings.jsonl");
        return LoadRows(File.Exists(accepted) ? accepted : proposed);
    }

    private static List<EmbeddingGemma2Row> LoadRows(string path) =>
        EmbeddingGemma2TestAssets.ReadJsonLines(path)
            .Select(r => new EmbeddingGemma2Row(
                r.GetProperty("id").GetString()!,
                r.GetProperty("prompt").GetString()!,
                r.GetProperty("prompt_text").GetString()!,
                r.GetProperty("text").GetString()!,
                r.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
                r.GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle()).ToArray()))
            .ToList();

    /// <summary>The protected right-padded batch (pad id 0) of six cases: id, ids and attention mask per row.</summary>
    public static IReadOnlyList<(string Id, long[] Ids, long[] Mask)> LoadPaddedBatch()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(EmbeddingGemma2TestAssets.FixtureDirectory, "tokenizer-batch.json")));
        return doc.RootElement.GetProperty("rows").EnumerateArray()
            .Select(r => (
                r.GetProperty("id").GetString()!,
                r.GetProperty("ids").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
                r.GetProperty("attention_mask").EnumerateArray().Select(e => e.GetInt64()).ToArray()))
            .ToList();
    }
}
