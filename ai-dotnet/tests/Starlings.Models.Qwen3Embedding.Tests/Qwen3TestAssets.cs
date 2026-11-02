using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.Qwen3Embedding;

namespace Starlings.Models.Qwen3Embedding.Tests;

/// <summary>
/// Locates the pinned tokenizers (tools/fetch-tokenizer-assets.sh) and the released graph directories (built locally
/// with model-pipeline/reference/qwen3-embedding/, never in Git). Tests that need an absent asset skip with the reason.
/// </summary>
internal static class Qwen3TestAssets
{
    /// <summary>The directory release_qwen3_onnx.py wrote for the embedding model.</summary>
    public const string EmbeddingReleaseVariable = "STARLING_QWEN3_EMBEDDING_RELEASE";

    /// <summary>The directory release_qwen3_onnx.py wrote for the reranker.</summary>
    public const string RerankerReleaseVariable = "STARLING_QWEN3_RERANKER_RELEASE";

    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    public static string ModelsRoot =>
        Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(RepositoryRoot, ".models");

    /// <summary>The released directory of a profile, or null with the reason.</summary>
    public static (string? Directory, string Reason) Release(Qwen3EmbeddingProfile profile)
    {
        string variable = profile.Task == Qwen3EmbeddingTask.Embedding ? EmbeddingReleaseVariable : RerankerReleaseVariable;
        return Environment.GetEnvironmentVariable(variable) is { Length: > 0 } dir && Directory.Exists(dir)
            ? (dir, string.Empty)
            : (null, $"Set {variable} to a directory built by model-pipeline/reference/qwen3-embedding/release_qwen3_onnx.py.");
    }

    /// <summary>The tokenization of a profile from its pinned tokenizer.json, or null with the reason.</summary>
    public static (Qwen3Tokenization? Tokenization, string Reason) Tokenization(Qwen3EmbeddingProfile profile)
    {
        var candidates = new List<string>();
        if (Release(profile).Directory is { } release)
        {
            candidates.Add(Path.Combine(release, Qwen3EmbeddingProfile.TokenizerFile));
        }

        candidates.Add(Path.Combine(ModelsRoot, profile.Repository, profile.Revision, Qwen3EmbeddingProfile.TokenizerFile));
        if (profile.Task == Qwen3EmbeddingTask.Reranking)
        {
            // Byte-identical file (same SHA-256), which tools/fetch-tokenizer-assets.sh already fetches.
            candidates.Add(Path.Combine(ModelsRoot, "Qwen", "Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", Qwen3EmbeddingProfile.TokenizerFile));
        }

        foreach (string path in candidates)
        {
            if (File.Exists(path))
            {
                return (Qwen3Tokenization.Open(profile, path), string.Empty);
            }
        }

        return (null, $"tokenizer.json of {profile.Repository} not found ({string.Join(", ", candidates)}); run tools/fetch-tokenizer-assets.sh.");
    }

    public static JsonElement Reference(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", name))).RootElement.Clone();

    public static long[] Longs(JsonElement array) => array.EnumerateArray().Select(e => e.GetInt64()).ToArray();

    private static string FindRepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline")) && Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException("Repository root not found.");
    }
}
