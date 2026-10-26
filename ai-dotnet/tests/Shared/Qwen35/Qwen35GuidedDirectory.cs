using System.Security.Cryptography;
using Starlings.Models.Qwen35;

namespace Starlings.TestSupport;

/// <summary>
/// A temporary variant directory that adds ONNX Runtime GenAI's guidance files to the published bundle: links to the
/// variant's files, a copy of the bundle's <c>transformers-5.18.0/tokenizer.json</c>, and the checkpoint's
/// <c>tokenizer_config.json</c> from <see cref="TokenizerConfigEnvironmentVariable"/> (not part of the bundle; verified by
/// SHA-256). The published bundle itself has neither file in its variant directories.
/// </summary>
internal sealed class Qwen35GuidedDirectory : IDisposable
{
    /// <summary>Path of the checkpoint's <c>tokenizer_config.json</c> at <see cref="Qwen35ModelAssets.Revision"/>.</summary>
    public const string TokenizerConfigEnvironmentVariable = "STARLING_QWEN35_TOKENIZER_CONFIG";

    /// <summary>SHA-256 of <c>tokenizer_config.json</c> at <see cref="Qwen35ModelAssets.Revision"/> (corvid spike S5).</summary>
    public const string TokenizerConfigSha256 = "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c";

    private Qwen35GuidedDirectory(Qwen35ModelFiles files) => Files = files;

    /// <summary>The variant's files with <see cref="Qwen35ModelFiles.ModelDirectory"/> pointing at the temporary directory.</summary>
    public Qwen35ModelFiles Files { get; }

    /// <summary>Builds the directory, or returns null with the reason (no model, or no verified <c>tokenizer_config.json</c>).</summary>
    public static Qwen35GuidedDirectory? TryCreate(Qwen35Variant variant, out string reason)
    {
        var (files, missing) = Qwen35TestAssets.FindModel(variant);
        if (files is null)
        {
            reason = missing;
            return null;
        }

        string? config = Environment.GetEnvironmentVariable(TokenizerConfigEnvironmentVariable);
        if (string.IsNullOrEmpty(config) || !File.Exists(config))
        {
            reason = $"Set {TokenizerConfigEnvironmentVariable} to tokenizer_config.json of {Qwen35ModelAssets.Repository}@{Qwen35ModelAssets.Revision}.";
            return null;
        }

        string sha = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(config)));
        if (sha != TokenizerConfigSha256)
        {
            reason = $"{config} has SHA-256 {sha}, not {TokenizerConfigSha256}.";
            return null;
        }

        string directory = Directory.CreateTempSubdirectory("starling-qwen35-guided-").FullName;
        foreach (string file in Directory.EnumerateFiles(files.ModelDirectory))
        {
            File.CreateSymbolicLink(Path.Combine(directory, Path.GetFileName(file)), file);
        }

        File.Copy(files.TokenizerPath, Path.Combine(directory, "tokenizer.json"));
        File.Copy(config, Path.Combine(directory, "tokenizer_config.json"));
        reason = string.Empty;
        return new Qwen35GuidedDirectory(files with { ModelDirectory = directory });
    }

    public void Dispose() => Directory.Delete(Files.ModelDirectory, recursive: true);
}
