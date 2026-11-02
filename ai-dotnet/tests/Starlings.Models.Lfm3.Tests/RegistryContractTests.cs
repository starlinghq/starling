using Starlings.TestSupport;

namespace Starlings.Models.Lfm3.Tests;

/// <summary>
/// The package capability table (<see cref="Lfm3Capabilities.All"/>) equals the registry manifest of its profile key
/// for key (the manifest spells a package key's <c>.</c> as <c>_</c>), and the package carries the model's license.
/// Weight-free; runs in CI.
/// </summary>
public sealed class RegistryContractTests
{
    private static string RepositoryRoot { get; } = FindRepositoryRoot();

    private static RegistryManifest Manifest { get; } = Assert.Single(RegistryManifest.Family(RepositoryRoot, "lfm3-1.7b"));

    [Fact]
    public void TheRegisteredProfileIsThePackageProfile()
    {
        Assert.Equal(Lfm3Capabilities.CpuProfile, Manifest.Id);
        Assert.Equal("experimental", Manifest.Maturity);
    }

    [Fact]
    public void CapabilitiesEqualTheManifestKeyForKey()
    {
        var package = Lfm3Capabilities.All.ToDictionary(
            entry => entry.Key.Replace('.', '_'),
            entry => entry.Value.ToString().ToLowerInvariant(),
            StringComparer.Ordinal);

        Assert.Equal(
            Manifest.Capabilities.OrderBy(entry => entry.Key, StringComparer.Ordinal),
            package.OrderBy(entry => entry.Key, StringComparer.Ordinal));
    }

    [Fact]
    public void CompatibilityRowsFollowThePackageStatuses()
    {
        Assert.Equal("experimental", Manifest.Compatibility["linux-x64 cpu"]);
        Assert.Equal(Lfm3Capabilities.All["generation.cuda"].ToString().ToLowerInvariant(), Manifest.Compatibility["linux-x64 cuda"]);
        Assert.Equal("untested", Manifest.Compatibility["windows-x64 cpu"]);
    }

    // The LFM Open License v1.0 is the package's license file: its revenue condition leads the file, the description
    // and the README, and the full upstream text (SHA-256 4d28ca14...) follows the header unchanged.
    [Fact]
    public void ThePackageCarriesTheModelLicenseProminently()
    {
        string dir = Path.Combine(RepositoryRoot, "ai-dotnet", "src", "Starlings.Models.Lfm3");
        string license = File.ReadAllText(Path.Combine(dir, "LICENSE.txt"));
        Assert.StartsWith("Starlings.Models.Lfm3: LICENSE", license, StringComparison.Ordinal);
        Assert.Contains("USD 10,000,000", license, StringComparison.Ordinal);
        Assert.Contains("LFM Open License v1.0 \n\nTERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION", license, StringComparison.Ordinal);
        Assert.EndsWith("END OF TERMS AND CONDITIONS", license.TrimEnd(), StringComparison.Ordinal);

        string project = File.ReadAllText(Path.Combine(dir, "Starlings.Models.Lfm3.csproj"));
        Assert.Contains("<PackageLicenseFile>LICENSE.txt</PackageLicenseFile>", project, StringComparison.Ordinal);
        Assert.Contains("<PackageRequireLicenseAcceptance>true</PackageRequireLicenseAcceptance>", project, StringComparison.Ordinal);
        Assert.Contains("<Description>MODEL LICENSE: LFM Open License v1.0", project, StringComparison.Ordinal);

        string readme = File.ReadAllText(Path.Combine(dir, "README.md"));
        string firstSection = readme[..readme.IndexOf("\n## ", StringComparison.Ordinal)];
        Assert.Contains("**Model license: LFM Open License v1.0, not MIT or Apache-2.0.**", firstSection, StringComparison.Ordinal);
        Assert.Contains("USD 10 million", firstSection, StringComparison.Ordinal);
    }

    /// <summary>The generated capability records equal the registry manifests field for field and capability key for key.</summary>
    [Fact]
    public void CapabilityRecordsEqualTheManifests()
    {
        CapabilityRecordChecks.AssertRecordsEqualManifests(Lfm3Capabilities.ProfileCapabilities, [Manifest]);
    }

    private static string FindRepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "registry")) &&
                Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException("Repository root not found above " + AppContext.BaseDirectory);
    }
}
