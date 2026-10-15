using System.Security.Cryptography;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Starlings.Benchmarks;

/// <summary>
/// The native ONNX Runtime and ONNX Runtime GenAI libraries this process actually mapped, each with its SHA-256
/// (<c>environment.native_library_digests</c>). Two runs can report the same ONNX Runtime version and still load
/// different native builds (NuGet against pip); the digests make "same backend" checkable. Read from
/// <c>/proc/self/maps</c> on Linux; elsewhere the value is the string <c>"unavailable"</c>.
/// </summary>
internal static partial class NativeLibraryDigests
{
    public const string Unavailable = "unavailable";

    /// <summary>The environment value: an array of <c>{path, sha256}</c> on Linux, otherwise <c>"unavailable"</c>.</summary>
    public static JsonNode Collect()
    {
        if (!OperatingSystem.IsLinux())
        {
            return JsonValue.Create(Unavailable);
        }

        string maps;
        try
        {
            maps = File.ReadAllText("/proc/self/maps");
        }
        catch (IOException)
        {
            return JsonValue.Create(Unavailable);
        }
        catch (UnauthorizedAccessException)
        {
            return JsonValue.Create(Unavailable);
        }

        return ToJson(Paths(maps), Sha256OrNull);
    }

    /// <summary>The distinct, sorted absolute paths of ONNX Runtime and GenAI shared objects in a maps listing.</summary>
    public static List<string> Paths(string maps) =>
        maps.Split('\n')
            .Select(PathOf)
            .Where(p => p is not null && IsOnnxRuntimeLibrary(p))
            .Select(p => p!)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToList();

    /// <summary>True for an ONNX Runtime or GenAI shared object (for example <c>libonnxruntime.so</c>,
    /// <c>libonnxruntime.so.1.30.0</c>, <c>libonnxruntime-genai.so</c>, <c>libonnxruntime_providers_cuda.so</c>,
    /// <c>onnxruntime_pybind11_state.cpython-313-x86_64-linux-gnu.so</c>); not other files of the same name stem.</summary>
    public static bool IsOnnxRuntimeLibrary(string path) => LibraryName().IsMatch(Path.GetFileName(path));

    public static JsonArray ToJson(IEnumerable<string> paths, Func<string, string?> sha256) =>
        new(paths.Select(p => (JsonNode)new JsonObject { ["path"] = p, ["sha256"] = sha256(p) }).ToArray());

    private static string? PathOf(string line)
    {
        // address perms offset dev inode [pathname]; the pathname may contain spaces, so take everything from the
        // first '/' after the inode column.
        string[] fields = line.Split(' ', 6, StringSplitOptions.RemoveEmptyEntries);
        if (fields.Length < 6)
        {
            return null;
        }

        string path = fields[5].Trim();
        return path.StartsWith('/') ? path : null;
    }

    public static string? Sha256OrNull(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            return Convert.ToHexStringLower(SHA256.HashData(stream));
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
    }

    [GeneratedRegex(@"^(lib)?onnxruntime[\w.-]*\.so(\.\d+)*$", RegexOptions.CultureInvariant)]
    private static partial Regex LibraryName();
}
