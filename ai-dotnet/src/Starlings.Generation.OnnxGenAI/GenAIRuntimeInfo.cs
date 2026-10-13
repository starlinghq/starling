using System.Text.Json;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>Process-wide facts about the ONNX Runtime GenAI binding.</summary>
/// <remarks>
/// The GenAI managed assembly carries no version (its assembly version is 0.0.0.0) and the native library exports no
/// version function, so the versions are read from the application's dependency manifest (<c>.deps.json</c>), which
/// lists the NuGet packages it was built with. In hosts without one (single-file or AOT publishing) they are null.
/// </remarks>
public static class GenAIRuntimeInfo
{
    /// <summary>The ONNX Runtime GenAI version this package was built and qualified with; builder and runtime must match it.</summary>
    public const string PinnedVersion = "0.17.1";

    private static readonly Lazy<(string? Managed, string? Native)> Packages = new(ReadPackages);

    /// <summary>Version of <c>Microsoft.ML.OnnxRuntimeGenAI.Managed</c> in the dependency manifest, or null when unknown.</summary>
    public static string? ManagedVersion => Packages.Value.Managed;

    /// <summary>
    /// The native package in the dependency manifest (<c>Microsoft.ML.OnnxRuntimeGenAI/0.17.1</c> or
    /// <c>Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1</c>), or null when unknown or absent.
    /// </summary>
    public static string? NativePackage => Packages.Value.Native;

    /// <summary>Throws when the dependency manifest names GenAI packages of another version than <see cref="PinnedVersion"/>.</summary>
    internal static void CheckVersions()
    {
        var (managed, native) = Packages.Value;
        string? nativeVersion = native?[(native.IndexOf('/', StringComparison.Ordinal) + 1)..];
        if ((managed is not null && managed != PinnedVersion) || (nativeVersion is not null && nativeVersion != PinnedVersion))
        {
            throw new GenAIRuntimeUnavailableException(
                $"ONNX Runtime GenAI packages do not match the qualified version {PinnedVersion}: managed {managed ?? "unknown"}, native {native ?? "unknown"}. " +
                "Reference Microsoft.ML.OnnxRuntimeGenAI or Microsoft.ML.OnnxRuntimeGenAI.Cuda at exactly that version.");
        }
    }

    private static (string?, string?) ReadPackages()
    {
        string? managed = null;
        string? native = null;
        try
        {
            if (AppContext.GetData("APP_CONTEXT_DEPS_FILES") is not string files)
            {
                return (null, null);
            }

            foreach (string path in files.Split(';', StringSplitOptions.RemoveEmptyEntries))
            {
                if (!File.Exists(path))
                {
                    continue;
                }

                using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
                if (!doc.RootElement.TryGetProperty("libraries", out var libraries))
                {
                    continue;
                }

                foreach (var library in libraries.EnumerateObject())
                {
                    string name = library.Name;
                    int slash = name.IndexOf('/', StringComparison.Ordinal);
                    if (slash < 0)
                    {
                        continue;
                    }

                    string id = name[..slash];
                    if (id.Equals("Microsoft.ML.OnnxRuntimeGenAI.Managed", StringComparison.OrdinalIgnoreCase))
                    {
                        managed = name[(slash + 1)..];
                    }
                    else if (id.Equals("Microsoft.ML.OnnxRuntimeGenAI", StringComparison.OrdinalIgnoreCase) ||
                             id.Equals("Microsoft.ML.OnnxRuntimeGenAI.Cuda", StringComparison.OrdinalIgnoreCase))
                    {
                        native = name;
                    }
                }
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            return (managed, native);
        }

        return (managed, native);
    }
}
