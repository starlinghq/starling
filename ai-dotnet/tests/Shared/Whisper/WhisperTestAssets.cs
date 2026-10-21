using Starlings.Acquisition;
using Starlings.Models.Whisper;

namespace Starlings.TestSupport;

/// <summary>
/// Locates the repository and the whisper snapshot (model graphs + golden fixtures). The snapshot is not in Git;
/// tests skip when it is absent, so CI without the model stays green.
/// </summary>
internal static class WhisperTestAssets
{
    /// <summary>Optional: a snapshot directory holding <c>onnx/encoder_model_fp16.onnx</c>, <c>onnx/decoder_model_fp16.onnx</c>, <c>tokenizer.json</c> and the <c>golden-*</c> fixtures.</summary>
    public const string SnapshotEnvironmentVariable = "STARLING_WHISPER_SNAPSHOT";

    public const string SnapshotName = "whisper-large-v3-turbo";

    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    public static string ModelsRoot =>
        Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(RepositoryRoot, ".models");

    public static string DefaultSnapshotDirectory => Path.Combine(ModelsRoot, "snapshots", SnapshotName);

    /// <summary>Snapshot directory, or null with the reason it is unavailable.</summary>
    public static (string? Dir, string Reason) FindSnapshotDirectory()
    {
        string? snap = Environment.GetEnvironmentVariable(SnapshotEnvironmentVariable);
        string dir = string.IsNullOrEmpty(snap) ? DefaultSnapshotDirectory : snap;
        if (!Directory.Exists(dir))
        {
            return (null, $"{dir} does not exist. Set {SnapshotEnvironmentVariable} to a whisper snapshot directory.");
        }

        return (dir, string.Empty);
    }

    /// <summary>A golden fixture path inside the snapshot, or null with the reason it is missing.</summary>
    public static (string? Path, string Reason) FindGoldenFile(string name)
    {
        var (dir, reason) = FindSnapshotDirectory();
        if (dir is null)
        {
            return (null, reason);
        }

        string path = Path.Combine(dir, name);
        return File.Exists(path) ? (path, string.Empty) : (null, $"{path} does not exist.");
    }

    /// <summary>Whisper model files (graphs + tokenizer), verified by size against the lock, or null with the reason unavailable.</summary>
    public static (WhisperModelFiles? Files, string Reason) FindModel()
    {
        var (dir, reason) = FindSnapshotDirectory();
        if (dir is null)
        {
            return (null, reason);
        }

        try
        {
            return (WhisperModelAssets.OpenDirectory(dir, CacheVerification.Size), string.Empty);
        }
        catch (ModelAssetMissingException ex)
        {
            return (null, ex.Message);
        }
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
