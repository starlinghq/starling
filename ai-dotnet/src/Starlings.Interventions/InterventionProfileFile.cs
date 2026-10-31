namespace Starlings.Interventions;

/// <summary>
/// A <c>profile.json</c> read from disk: the parsed profile, the SHA-256 of the file bytes (the profile's identity in a
/// compiler manifest), and the directory its directions files are resolved against.
/// </summary>
public sealed class InterventionProfileFile
{
    private InterventionProfileFile(string path, string sha256, InterventionProfile profile)
    {
        Path = path;
        Directory = System.IO.Path.GetDirectoryName(path)!;
        Sha256 = sha256;
        Profile = profile;
    }

    /// <summary>Full path of the <c>profile.json</c> file.</summary>
    public string Path { get; }

    /// <summary>Directory of the profile file; directions file paths are relative to it.</summary>
    public string Directory { get; }

    /// <summary>SHA-256 of the file bytes as stored.</summary>
    public string Sha256 { get; }

    /// <summary>The parsed, validated profile.</summary>
    public InterventionProfile Profile { get; }

    /// <summary>Reads, hashes and validates the profile at <paramref name="path"/>.</summary>
    /// <exception cref="InterventionContractException">The document violates the contract.</exception>
    public static InterventionProfileFile Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string full = System.IO.Path.GetFullPath(path);
        FileInfo info = new(full);
        if (!info.Exists)
        {
            throw new FileNotFoundException($"Intervention profile not found: {full}", full);
        }

        if (info.Length > Internal.StrictJson.MaxDocumentBytes)
        {
            throw new InterventionContractException($"{full} is {info.Length} bytes; the limit is {Internal.StrictJson.MaxDocumentBytes}.");
        }

        byte[] bytes = File.ReadAllBytes(full);
        InterventionProfile profile = InterventionProfile.Parse(bytes);
        return new InterventionProfileFile(full, InterventionHashes.ComputeSha256(bytes), profile);
    }

    /// <summary>Full path of <paramref name="site"/>'s directions file (never outside <see cref="Directory"/>).</summary>
    public string GetDirectionsPath(InterventionSite site)
    {
        ArgumentNullException.ThrowIfNull(site);
        return Internal.StrictJson.Resolve(Directory, site.DirectionsFile);
    }

    /// <summary>Loads and verifies <paramref name="site"/>'s directions (size, SHA-256, finite values).</summary>
    /// <exception cref="InterventionMismatchException">The file's SHA-256 differs from the profile's.</exception>
    /// <exception cref="InterventionContractException">The file has the wrong size or a non-finite value.</exception>
    public InterventionDirections LoadDirections(InterventionSite site) => InterventionDirections.Load(GetDirectionsPath(site), site);

    /// <summary>Verifies every site's directions file; the profile is then complete and consistent on disk.</summary>
    public void VerifyDirections()
    {
        foreach (InterventionSite site in Profile.Sites)
        {
            _ = LoadDirections(site);
        }
    }
}
