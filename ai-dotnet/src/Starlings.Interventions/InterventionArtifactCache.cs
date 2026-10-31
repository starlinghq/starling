using System.Text.RegularExpressions;

namespace Starlings.Interventions;

/// <summary>
/// Read side of the derived-artifact cache, <c>&lt;cache&gt;/&lt;model&gt;/interventions/&lt;key&gt;/</c>. Entries are
/// written (compiled, then published with one directory rename) by the Python compiler,
/// <c>model_pipeline.interventions.cache</c>; this class locates and verifies them. Experimental. Layout and key:
/// <c>docs/interventions/cache.md</c>.
/// </summary>
public static partial class InterventionArtifactCache
{
    /// <summary>Name of the directory under <c>&lt;cache&gt;/&lt;model&gt;</c> that holds the entries.</summary>
    public const string DirectoryName = "interventions";

    /// <summary>Whether <paramref name="model"/> is a valid cache model name (<c>[a-z0-9][a-z0-9._-]</c>, at most 64).</summary>
    public static bool IsModelName(string? model) => model is not null && ModelNamePattern().IsMatch(model);

    /// <summary>The directory that holds the entries of <paramref name="model"/>.</summary>
    /// <exception cref="ArgumentException">The model name is invalid.</exception>
    public static string GetModelDirectory(string cacheRoot, string model)
    {
        ArgumentException.ThrowIfNullOrEmpty(cacheRoot);
        if (!IsModelName(model))
        {
            throw new ArgumentException($"'{model}' is not a cache model name ([a-z0-9][a-z0-9._-], at most 64).", nameof(model));
        }

        return Path.Combine(Path.GetFullPath(cacheRoot), model, DirectoryName);
    }

    /// <summary>The directory of the entry with cache key <paramref name="cacheKey"/> (it may not exist).</summary>
    /// <exception cref="ArgumentException">The model name or the key is invalid.</exception>
    public static string GetEntryDirectory(string cacheRoot, string model, string cacheKey)
    {
        InterventionHashes.RequireSha256(cacheKey, nameof(cacheKey));
        return Path.Combine(GetModelDirectory(cacheRoot, model), cacheKey);
    }

    /// <summary>
    /// The published entry directories of <paramref name="model"/> (names that are cache keys), in ordinal order.
    /// Temporary build directories and lock files are not listed. Entries are not opened or verified.
    /// </summary>
    public static IReadOnlyList<string> ListEntries(string cacheRoot, string model)
    {
        string directory = GetModelDirectory(cacheRoot, model);
        if (!System.IO.Directory.Exists(directory))
        {
            return [];
        }

        return [.. new DirectoryInfo(directory).EnumerateDirectories()
            .Where(d => (d.Attributes & FileAttributes.ReparsePoint) == 0 && InterventionHashes.IsSha256(d.Name))
            .Select(d => d.FullName)
            .Order(StringComparer.Ordinal)];
    }

    /// <summary>
    /// Opens and verifies the entry with cache key <paramref name="cacheKey"/>: <see cref="InterventionArtifact.Open"/>,
    /// and the key recomputed from its manifest must equal <paramref name="cacheKey"/> (the directory name).
    /// </summary>
    /// <exception cref="DirectoryNotFoundException">No entry has this key.</exception>
    /// <exception cref="InterventionMismatchException">The entry's manifest gives another key, or a file does not match.</exception>
    /// <exception cref="InterventionContractException">The manifest or profile violates the contract.</exception>
    public static InterventionArtifact Open(
        string cacheRoot,
        string model,
        string cacheKey,
        InterventionArtifactVerification verification = InterventionArtifactVerification.Fast,
        string? baseDirectory = null)
    {
        string directory = GetEntryDirectory(cacheRoot, model, cacheKey);
        return OpenEntry(directory, verification, baseDirectory);
    }

    /// <summary>
    /// Opens and verifies a cache entry directory (whose name is its cache key): <see cref="InterventionArtifact.Open"/>,
    /// and the key recomputed from its manifest must equal the directory name.
    /// </summary>
    /// <exception cref="DirectoryNotFoundException">The directory does not exist.</exception>
    /// <exception cref="InterventionMismatchException">The name is not the manifest's key, or a file does not match.</exception>
    public static InterventionArtifact OpenEntry(
        string entryDirectory,
        InterventionArtifactVerification verification = InterventionArtifactVerification.Fast,
        string? baseDirectory = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(entryDirectory);
        string full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(entryDirectory));
        if (!System.IO.Directory.Exists(full))
        {
            throw new DirectoryNotFoundException($"Intervention cache entry not found: {full}");
        }

        string name = Path.GetFileName(full);
        if (!InterventionHashes.IsSha256(name))
        {
            throw new InterventionMismatchException($"{full} is not a cache entry: its name is not a cache key.");
        }

        InterventionArtifact artifact = InterventionArtifact.Open(full, verification, baseDirectory);
        if (!string.Equals(artifact.CacheKey, name, StringComparison.Ordinal))
        {
            throw new InterventionMismatchException($"Cache entry {full} holds the artifact of cache key {artifact.CacheKey}.");
        }

        return artifact;
    }

    [GeneratedRegex("^[a-z0-9][a-z0-9._-]{0,63}\\z", RegexOptions.CultureInvariant)]
    private static partial Regex ModelNamePattern();
}
