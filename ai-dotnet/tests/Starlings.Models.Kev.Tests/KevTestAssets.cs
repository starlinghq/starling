using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.Kev;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// Locates the test data (in Git), the pinned tokenizer and the Kev bundles (neither is in Git). Tests that need an
/// absent asset skip with the reason, so CI without models stays green.
/// </summary>
internal static class KevTestAssets
{
    /// <summary>Directory of a verified Kev-0.8B bundle (model.onnx, model.onnx.data, head.safetensors, head.json, tokenizer.json).</summary>
    public const string Kev08BBundleEnvironmentVariable = "STARLING_KEV_08B_BUNDLE";

    /// <summary>Directory of a verified Kev-4B bundle.</summary>
    public const string Kev4BBundleEnvironmentVariable = "STARLING_KEV_4B_BUNDLE";

    public static string DataDirectory => Path.Combine(AppContext.BaseDirectory, "Data");

    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    /// <summary>Cache root: <c>$STARLING_MODELS</c>, else <c>&lt;repo&gt;/.models</c> (tools/fetch-tokenizer-assets.sh).</summary>
    public static string ModelsRoot =>
        Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(RepositoryRoot, ".models");

    public static JsonElement ReadData(string name)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(DataDirectory, name)));
        return doc.RootElement.Clone();
    }

    /// <summary>The pinned tokenizer.json of either Kev repository (the same bytes), or null with the reason.</summary>
    public static (string? Path, string Reason) FindTokenizer()
    {
        var candidates = KevModelProfile.All
            .Select(p => Path.Combine(ModelsRoot, p.Repository, p.Revision, KevModelAssets.TokenizerFile))
            .Concat(new[] { Kev08BBundleEnvironmentVariable, Kev4BBundleEnvironmentVariable }
                .Select(Environment.GetEnvironmentVariable)
                .Where(d => !string.IsNullOrEmpty(d))
                .Select(d => Path.Combine(d!, KevModelAssets.TokenizerFile)))
            .ToList();
        string? found = candidates.FirstOrDefault(File.Exists);
        return found is not null
            ? (found, string.Empty)
            : (null, $"Kev tokenizer.json not found ({string.Join(", ", candidates)}); run tools/fetch-tokenizer-assets.sh.");
    }

    /// <summary>A bundle verified by size against the profile's lock, or null with the reason.</summary>
    public static (KevModelFiles? Files, string Reason) FindBundle(KevModel model)
    {
        string variable = model == KevModel.Kev08B ? Kev08BBundleEnvironmentVariable : Kev4BBundleEnvironmentVariable;
        string? directory = Environment.GetEnvironmentVariable(variable);
        if (string.IsNullOrEmpty(directory))
        {
            return (null, $"{variable} is not set: point it at a {KevModelProfile.For(model).Id} bundle to run the model tests.");
        }

        try
        {
            return (KevModelAssets.OpenDirectory(model, directory, CacheVerification.Size), string.Empty);
        }
        catch (ModelAcquisitionException e)
        {
            return (null, $"{variable}={directory} is not a valid bundle: {e.Message}");
        }
    }

    private static string FindRepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline")) && Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException("Repository root not found above the test binaries.");
    }
}
