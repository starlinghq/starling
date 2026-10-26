using Starlings.TestSupport;

namespace Starlings.Models.Whisper.Tests;

/// <summary>
/// The package capability table (<see cref="WhisperCapabilities.All"/>) equals the registry manifest of its profile key
/// for key: the manifest spells a package key's <c>.</c> as <c>_</c>. Weight-free; runs in CI.
/// </summary>
public sealed class RegistryContractTests
{
    private static RegistryManifest Manifest { get; } =
        Assert.Single(RegistryManifest.Family(WhisperTestAssets.RepositoryRoot, "whisper-large-v3-turbo"));

    [Fact]
    public void TheRegisteredProfileIsThePackageProfile()
    {
        Assert.Equal(WhisperCapabilities.CpuProfile, Manifest.Id);
        Assert.Equal("experimental", Manifest.Maturity);
    }

    [Fact]
    public void CapabilitiesEqualTheManifestKeyForKey()
    {
        var package = WhisperCapabilities.All.ToDictionary(
            entry => entry.Key.Replace('.', '_'),
            entry => entry.Value.ToString().ToLowerInvariant(),
            StringComparer.Ordinal);

        Assert.Equal(
            Manifest.Capabilities.OrderBy(entry => entry.Key, StringComparer.Ordinal),
            package.OrderBy(entry => entry.Key, StringComparer.Ordinal));
    }

    [Fact]
    public void AnExperimentalProfileSupportsNothing()
    {
        Assert.Equal("experimental", Manifest.Maturity);
        Assert.DoesNotContain(CapabilityStatus.Supported, WhisperCapabilities.All.Values);
        Assert.DoesNotContain("supported", Manifest.Capabilities.Values);
        Assert.DoesNotContain("supported", Manifest.Compatibility.Values);
    }

    [Fact]
    public void CompatibilityRowsFollowThePackageStatuses()
    {
        // CPU transcription is what the route evidenced; CUDA follows the package's transcription.cuda status.
        Assert.Equal(Name(WhisperCapabilities.All["transcription"]), Manifest.Compatibility["linux-x64 cpu"]);
        Assert.Equal(Name(WhisperCapabilities.All["transcription.cuda"]), Manifest.Compatibility["linux-x64 cuda"]);
        Assert.Equal("untested", Manifest.Compatibility["windows-x64 cpu"]);
    }

    private static string Name(CapabilityStatus status) => status.ToString().ToLowerInvariant();
}
