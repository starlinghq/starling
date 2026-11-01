namespace Starlings.Interventions;

/// <summary>How much of a derived artifact <see cref="InterventionArtifact.Open"/> checks.</summary>
public enum InterventionArtifactVerification
{
    /// <summary>
    /// For every load. The manifest parses; <c>profile.json</c> has the manifest's profile SHA-256 and matches it, and
    /// its directions files verify; the derived graph has the recorded size and SHA-256; every other derived file (the
    /// hard-linked external weights) is recorded with its base artifact's SHA-256 and size, exists with that size, and
    /// still has the <see cref="InterventionArtifactFile.Identity"/> the manifest records (an entry without one fails):
    /// on Linux device, inode and modification time (and, with a base directory, the base file is that same inode);
    /// on other platforms the modification time only, at 100 ns precision, and no base-directory inode check. The
    /// weights are not hashed, so a rewrite in place that keeps the size and restores the modification time is not
    /// detected at this level.
    /// </summary>
    Fast,

    /// <summary>
    /// Instead of the identity check of <see cref="Fast"/>: the SHA-256 of every derived file (this reads the whole
    /// weights file) and, when a base directory is given, of every base artifact under it. A copied or restored entry
    /// (new inodes) fails <see cref="Fast"/> and can still pass <see cref="Full"/>.
    /// </summary>
    Full,
}

/// <summary>
/// A derived intervention artifact directory opened and verified (a cache entry, or any directory with the same layout:
/// manifest, derived graph, linked data, <c>profile.json</c> and its directions files, or for a schema version 2
/// manifest <c>profiles/&lt;i&gt;/profile.json</c> and theirs): the compiler manifest, the compiled profiles, and the
/// derived graph path. Open it once at load; it is immutable.
/// The layout is specified in <c>docs/interventions/cache.md</c>.
/// </summary>
public sealed class InterventionArtifact
{
    /// <summary>File name of the compiler manifest in a derived artifact directory.</summary>
    public const string ManifestFileName = "compiler-manifest.json";

    /// <summary>File name of the compiled profile (stored byte for byte) in a derived artifact directory.</summary>
    public const string ProfileFileName = "profile.json";

    /// <summary>Directory of the profiles of a schema version 2 entry: <c>profiles/&lt;strength index&gt;/profile.json</c>.</summary>
    public const string ProfilesDirectoryName = "profiles";

    private InterventionArtifact(string directory, InterventionManifest manifest, InterventionProfileFile[] profiles, string cacheKey)
    {
        Directory = directory;
        Manifest = manifest;
        Profiles = profiles;
        CacheKey = cacheKey;
        GraphPath = Internal.StrictJson.Resolve(directory, manifest.DerivedGraph.Path);
    }

    /// <summary>Full path of the derived artifact directory.</summary>
    public string Directory { get; }

    /// <summary>The verified compiler manifest.</summary>
    public InterventionManifest Manifest { get; }

    /// <summary>The compiled profile, read from the artifact directory, when the graph carries exactly one.</summary>
    /// <exception cref="InvalidOperationException">The graph carries several profiles; use <see cref="Profiles"/>.</exception>
    public InterventionProfileFile Profile => Profiles.Count == 1
        ? Profiles[0]
        : throw new InvalidOperationException($"The artifact carries {Profiles.Count} profiles; use Profiles.");

    /// <summary>The compiled profiles in strength-vector order (<see cref="InterventionManifest.Profiles"/>), read from the artifact directory.</summary>
    public IReadOnlyList<InterventionProfileFile> Profiles { get; }

    /// <summary>
    /// Path of profile <paramref name="index"/> relative to an artifact directory of <paramref name="manifest"/>:
    /// <c>profile.json</c> (schema version 1) or <c>profiles/&lt;index&gt;/profile.json</c> (schema version 2, so two
    /// profiles' directions files never collide).
    /// </summary>
    public static string ProfilePath(InterventionManifest manifest, int index)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, manifest.Profiles.Count);
        return manifest.SchemaVersion == InterventionManifest.MultiProfileSchemaVersion
            ? $"{ProfilesDirectoryName}/{index.ToString(System.Globalization.CultureInfo.InvariantCulture)}/{ProfileFileName}"
            : ProfileFileName;
    }

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
        InterventionProfileFile[] profiles = new InterventionProfileFile[manifest.Profiles.Count];
        for (int i = 0; i < profiles.Length; i++)
        {
            profiles[i] = InterventionProfileFile.Load(Internal.StrictJson.Resolve(full, ProfilePath(manifest, i)));
        }

        manifest.VerifyProfiles(profiles);
        foreach (InterventionProfileFile profile in profiles)
        {
            profile.VerifyDirections();
        }

        manifest.DerivedGraph.Verify(full);
        VerifyLinkedFiles(manifest);
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

            VerifyIdentity(path, file, manifest.BaseGraph.Path, baseDirectory);
        }

        if (verification == InterventionArtifactVerification.Full && baseDirectory is not null)
        {
            manifest.VerifyBaseArtifacts(baseDirectory);
        }

        return new InterventionArtifact(full, manifest, profiles, InterventionCacheKey.Compute(manifest));
    }

    /// <summary>
    /// Fast check of a linked file against the identity the compiler recorded: a replaced link, a restored or copied
    /// entry, or a weights file rewritten in place (new modification time) fails here without hashing the weights.
    /// </summary>
    private static void VerifyIdentity(string path, InterventionArtifactFile file, string baseGraphPath, string? baseDirectory)
    {
        InterventionFileIdentity recorded = file.Identity ?? throw new InterventionMismatchException(
            $"The manifest records no identity (device, inode, mtime) for {path}, so Fast verification cannot tie it to the bytes that were hashed; verify with Full, or evict and recompile the entry.");
        InterventionFileIdentity? actual = Internal.FileIdentityProbe.TryGet(path);
        if (actual is null)
        {
            long ticks = Math.DivRem(recorded.ModifiedNanoseconds, 100, out long remainder) - (remainder < 0 ? 1 : 0);
            if (Internal.FileIdentityProbe.ModifiedNanoseconds(path) != ticks * 100)
            {
                throw new InterventionMismatchException(
                    $"{path} was modified since the entry was written: the linked weights may have been rewritten in place; verify with Full, or evict the entry.");
            }

            return;
        }

        if (actual.Device != recorded.Device || actual.Inode != recorded.Inode)
        {
            throw new InterventionMismatchException(
                $"{path} is now device {actual.Device} inode {actual.Inode}; the manifest records device {recorded.Device} inode {recorded.Inode} (the file was replaced, or the entry copied or restored); verify with Full, or evict the entry.");
        }

        if (actual.ModifiedNanoseconds != recorded.ModifiedNanoseconds)
        {
            throw new InterventionMismatchException(
                $"{path} was modified (mtime {actual.ModifiedNanoseconds} ns, recorded {recorded.ModifiedNanoseconds} ns): the linked weights may have been rewritten in place; verify with Full, or evict the entry.");
        }

        if (baseDirectory is not null)
        {
            int slash = baseGraphPath.LastIndexOf('/');
            string basePath = slash < 0 ? file.Path : baseGraphPath[..(slash + 1)] + file.Path;
            string original = Internal.StrictJson.Resolve(Path.GetFullPath(baseDirectory), basePath);
            InterventionFileIdentity? source = File.Exists(original) ? Internal.FileIdentityProbe.TryGet(original) : null;
            if (source is null || source.Device != actual.Device || source.Inode != actual.Inode)
            {
                throw new InterventionMismatchException(
                    $"{path} is no longer a hard link to {original} (the base file was replaced or removed); verify with Full, or evict the entry.");
            }
        }
    }

    /// <summary>
    /// In a derived artifact directory the derived files are exactly the base graph's external data files (hard links), at
    /// their location relative to the base graph's directory, with the base artifact's SHA-256 and size. The cache key
    /// covers <c>base_artifacts</c> but not <c>derived_files</c>, so without this rule a manifest could name the base
    /// weights in its key while recording, and verifying, other bytes for the file ONNX Runtime reads.
    /// </summary>
    private static void VerifyLinkedFiles(InterventionManifest manifest)
    {
        string graphPath = manifest.BaseGraph.Path;
        int slash = graphPath.LastIndexOf('/');
        string prefix = slash < 0 ? string.Empty : graphPath[..(slash + 1)];
        Dictionary<string, InterventionArtifactFile> external = new(StringComparer.Ordinal);
        foreach (InterventionBaseArtifact artifact in manifest.BaseArtifacts)
        {
            if (artifact.Role == InterventionArtifactRole.ExternalData)
            {
                external[artifact.File.Path] = artifact.File;
            }
        }

        HashSet<string> linked = new(StringComparer.Ordinal);
        foreach (InterventionArtifactFile file in manifest.DerivedFiles)
        {
            string basePath = prefix + file.Path;
            if (!external.TryGetValue(basePath, out InterventionArtifactFile? source))
            {
                throw new InterventionMismatchException(
                    $"Derived file '{file.Path}' is not a link of a base external data artifact ('{basePath}').");
            }

            if (!string.Equals(source.Sha256, file.Sha256, StringComparison.Ordinal) || source.Size != file.Size)
            {
                throw new InterventionMismatchException(
                    $"Derived file '{file.Path}' records SHA-256 {file.Sha256} ({file.Size} bytes) but its base artifact '{basePath}' is {source.Sha256} ({source.Size} bytes).");
            }

            linked.Add(basePath);
        }

        foreach (string path in external.Keys)
        {
            if (!linked.Contains(path))
            {
                throw new InterventionMismatchException($"Base external data '{path}' has no linked derived file.");
            }
        }
    }
}
