using System.Text;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.Qwen35;
using Starlings.Tokenizers;

namespace Starlings.TestSupport;

/// <summary>
/// Locates the Qwen3.5 tokenizer, the reference generation fixture (read only) and the built ONNX Runtime GenAI bundle
/// (not in Git). Tests that need an absent asset skip with the reason, so CI without models stays green.
/// </summary>
internal static class Qwen35TestAssets
{
    /// <summary>Optional: a directory in the snapshot layout (<c>chat_template.jinja</c>, <c>transformers-5.18.0/</c>, <c>ortgenai-*/</c>).</summary>
    public const string SnapshotEnvironmentVariable = "STARLING_QWEN35_SNAPSHOT";

    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    /// <summary>Cache root: <c>$STARLING_MODELS</c>, else <c>&lt;repo&gt;/.models</c> (where tools/fetch-tokenizer-assets.sh writes).</summary>
    public static string ModelsRoot =>
        Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(RepositoryRoot, ".models");

    public static string PromptsPath => Path.Combine(RepositoryRoot, "model-pipeline", "reference", "generation", "prompts-v1.jsonl");

    public static string ManifestDirectory => Path.Combine(RepositoryRoot, "model-pipeline", "registry", "qwen3.5-0.8b");

    /// <summary>
    /// The reference generation fixture: the protected copy once accepted, else the copy staged with the tests
    /// (byte-identical; proposed in a separate pull request).
    /// </summary>
    public static string FixturePath
    {
        get
        {
            string accepted = Path.Combine(RepositoryRoot, "model-pipeline", "protected", "qwen3.5-0.8b", "reference", "generation.jsonl");
            return File.Exists(accepted) ? accepted : Path.Combine(AppContext.BaseDirectory, "TestData", "generation-v1", "generation.jsonl");
        }
    }

    private static readonly Lazy<(Tokenizer? Tokenizer, string Reason)> SharedTokenizer = new(LoadTokenizer);

    /// <summary>The qwen35-bpe-chat tokenizer, or null with the reason.</summary>
    public static (Tokenizer? Tokenizer, string Reason) Tokenizer => SharedTokenizer.Value;

    /// <summary>The model files of a variant (size-verified, offline), or null with the reason.</summary>
    public static (Qwen35ModelFiles? Files, string Reason) FindModel(Qwen35Variant variant)
    {
        string? snapshot = Environment.GetEnvironmentVariable(SnapshotEnvironmentVariable);
        try
        {
            var files = string.IsNullOrEmpty(snapshot)
                ? Qwen35ModelAssets.OpenLocal(variant, new ModelAcquisitionOptions { CacheDirectory = ModelsRoot, Verification = CacheVerification.Size })
                : Qwen35ModelAssets.OpenDirectory(snapshot, variant, CacheVerification.Size);
            return (files, string.Empty);
        }
        catch (ModelAcquisitionException ex)
        {
            return (null, ex.Message + $" Alternatively set {SnapshotEnvironmentVariable} to a directory built by " +
                          "model-pipeline/reference/generation/build_ortgenai_bundle.py.");
        }
    }

    public static IReadOnlyList<GenerationCase> LoadCases() => ReadJsonLines(FixturePath).Select(GenerationCase.Parse).ToList();

    // Split on LF only: prompt text may contain U+2028 and similar.
    public static List<JsonElement> ReadJsonLines(string path) =>
        File.ReadAllText(path, Encoding.UTF8).Split('\n')
            .Where(l => l.Trim().Length > 0)
            .Select(l => JsonDocument.Parse(l).RootElement.Clone())
            .ToList();

    private static (Tokenizer?, string) LoadTokenizer()
    {
        var candidates = new List<string>();
        if (Environment.GetEnvironmentVariable(SnapshotEnvironmentVariable) is { Length: > 0 } snapshot)
        {
            candidates.Add(Path.Combine(snapshot, Qwen35ModelAssets.TokenizerFile));
        }

        candidates.Add(Path.Combine(ModelsRoot, Qwen35ModelAssets.Repository, Qwen35ModelAssets.Revision, Qwen35ModelAssets.TokenizerFile));
        foreach (string path in candidates)
        {
            if (File.Exists(path))
            {
                return (Tokenizers.Tokenizer.FromFile(path), string.Empty);
            }
        }

        return (null, $"qwen35-bpe-chat tokenizer.json not found ({string.Join(", ", candidates)}); run tools/fetch-tokenizer-assets.sh.");
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

/// <summary>One prompt of the reference generation fixture.</summary>
internal sealed record GenerationCase(
    string Id,
    IReadOnlyList<Qwen35Message>? Messages,
    bool HasTools,
    bool EnableThinking,
    int MaxNewTokens,
    string Rendered,
    int[] PromptIds,
    int[] OutputIds,
    string FinishReason,
    float[] StepMargin)
{
    public static GenerationCase Parse(JsonElement r)
    {
        bool tools = r.TryGetProperty("tools", out var t) && t.ValueKind == JsonValueKind.Array;
        bool thinking = r.TryGetProperty("chat_template_kwargs", out var k) && k.ValueKind == JsonValueKind.Object &&
            k.TryGetProperty("enable_thinking", out var e) && e.GetBoolean();
        var messages = r.GetProperty("messages").EnumerateArray().Select(m => new Qwen35Message(
            m.GetProperty("role").GetString() switch
            {
                "system" => Qwen35Role.System,
                "user" => Qwen35Role.User,
                "assistant" => Qwen35Role.Assistant,
                var other => throw new NotSupportedException(other),
            },
            m.GetProperty("content").GetString()!)).ToList();
        return new GenerationCase(
            r.GetProperty("id").GetString()!,
            messages,
            tools,
            thinking,
            r.GetProperty("max_new_tokens").GetInt32(),
            r.GetProperty("rendered").GetString()!,
            r.GetProperty("prompt_ids").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
            r.GetProperty("output_ids").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
            r.GetProperty("finish_reason").GetString()!,
            r.GetProperty("step_margin").EnumerateArray().Select(x => (float)x.GetDouble()).ToArray());
    }

    /// <summary>Reference completion ids without the stop token.</summary>
    public int[] ContentIds => FinishReason == "eos" ? OutputIds[..^1] : OutputIds;
}
