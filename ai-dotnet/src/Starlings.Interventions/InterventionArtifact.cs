namespace Starlings.Interventions;

/// <summary>How much of a derived artifact <see cref="InterventionArtifact.Open"/> checks.</summary>
public enum InterventionArtifactVerification
{
    /// <summary>
    /// For every load. The manifest parses; <c>profile.json</c> has the manifest's profile SHA-256 and matches it, and
    /// its directions files verify; the derived graph has the recorded size and SHA-256; every other derived file (the
    /// hard-linked external weights) exists with the recorded size. The weights are not hashed, so a weights file
    /// rewritten in place with the same size is not detected at this level.
    /// </summary>
    Fast,

    /// <summary>
    /// <see cref="Fast"/>, plus the SHA-256 of every derived file (this reads the whole weights file) and, when a base
    /// directory is given, of every base artifact under it.
    /// </summary>
    Full,
}

/// <summary>
/// A derived intervention artifact directory opened and verified (a cache entry, or any directory with the same layout:
/// manifest, derived graph, linked data, <c>profile.json</c> and its directions files):
/// the compiler manifest, the compiled profile, and the derived graph path. Open it once at load; it is immutable.
/// The layout is specified in <c>docs/interventions/cache.md</c>.
/// </summary>
public sealed class InterventionArtifact
{
    /// <summary>File name of the compiler manifest in a derived artifact directory.</summary>
    public const string ManifestFileName = "compiler-manifest.json";

    /// <summary>File name of the compiled profile (stored byte for byte) in a derived artifact directory.</summary>
    public const string ProfileFileName = "profile.json";

    private InterventionArtifact(string directory, InterventionManifest manifest, InterventionProfileFile profile, string cacheKey)
    {
        Directory = directory;
        Manifest = manifest;
        Profile = profile;
        CacheKey = cacheKey;
        GraphPath = Internal.StrictJson.Resolve(directory, manifest.DerivedGraph.Path);
    }

    /// <summary>Full path of the derived artifact directory.</summary>
    public string Directory { get; }

    /// <summary>The verified compiler manifest.</summary>
    public InterventionManifest Manifest { get; }

    /// <summary>The compiled profile, read from the artifact directory.</summary>
    public InterventionProfileFile Profile { get; }

    /// <summary>The cache key recomputed from <see cref="Manifest"/> (<see cref="InterventionCacheKey"/>).</summary>
    public string CacheKey { get; }

    /// <summary>Full path of the derived <c>model.onnx</c>, to load with the ordinary ONNX Runtime session API.</summary>
    public string GraphPath { get; }

    /// <summary>
    /// Opens and verifies the derived artifact in <paramref name="directory"/>. With <paramref name="baseDirectory"/>
    /// (the base model directory the manifest's <c>base_artifacts</c> paths are relative to) and
    /// <see cref="InterventionArtifactVerification.Full"/>, the base artifacts are hashed too. Call
    /// <see cref="InterventionManifest.VerifyBaseModel(string)"/> on <see cref="Manifest"/> with the base graph SHA-256
    /// the application expects before creating a session.
    /// </summary>
    /// <exception cref="InterventionContractException">The manifest or profile violates the contract.</exception>
    /// <exception cref="InterventionMismatchException">A file, the profile or a hash does not match the manifest.</exception>
    /// <exception cref="FileNotFoundException">The manifest, the profile or a recorded file is missing.</exception>
    public static InterventionArtifact Open(
        string directory,
        InterventionArtifactVerification verification = InterventionArtifactVerification.Fast,
        string? baseDirectory = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        if (!Enum.IsDefined(verification))
        {
            throw new ArgumentOutOfRangeException(nameof(verification));
        }

        string full = Path.GetFullPath(directory);
        InterventionManifest manifest = InterventionManifest.Load(Path.Combine(full, ManifestFileName));
        InterventionProfileFile profile = InterventionProfileFile.Load(Path.Combine(full, ProfileFileName));
        manifest.VerifyProfile(profile);
        profile.VerifyDirections();
        manifest.DerivedGraph.Verify(full);
        foreach (InterventionArtifactFile file in manifest.DerivedFiles)
        {
            if (verification == InterventionArtifactVerification.Full)
            {
                file.Verify(full);
                continue;
            }

            string path = Internal.StrictJson.Resolve(full, file.Path);
            FileInfo info = new(path);
            if (!info.Exists)
            {
                throw new FileNotFoundException($"Intervention artifact not found: {path}", path);
            }

            if (info.Length != file.Size)
            {
                throw new InterventionMismatchException($"{path}: size {info.Length} bytes, expected {file.Size}.");
            }
        }

        if (verification == InterventionArtifactVerification.Full && baseDirectory is not null)
        {
            manifest.VerifyBaseArtifacts(baseDirectory);
        }

        return new InterventionArtifact(full, manifest, profile, InterventionCacheKey.Compute(manifest));
    }
}
