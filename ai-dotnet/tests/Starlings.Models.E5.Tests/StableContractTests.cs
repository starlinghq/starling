using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// The capability table of the Stable-contract profile equals its registry manifest key for key, and the package-wide
/// table agrees with it. Weight-free; runs in CI.
/// </summary>
public class StableContractTests
{
    // Manifest capability -> the package-wide keys (E5Capabilities.All) that carry the same status on the CPU profile.
    private static readonly Dictionary<string, string[]> PackageKeys = new(StringComparer.Ordinal)
    {
        ["batch"] = ["batch"],
        ["streaming"] = ["streaming"],
        ["cancellation"] = ["cancellation"],
        ["persistence"] = ["persistence", "mlnet.persistence"],
        ["training"] = ["training", "mlnet.training"],
        ["aot"] = ["aot"],
        ["extensions_ai"] = ["extensions_ai.embedding_generator"],
        ["mlnet"] = ["mlnet.embedding_transform"],
    };

    private static IReadOnlyList<RegistryManifest> Family { get; } = RegistryManifest.Family(E5TestAssets.RepositoryRoot, "multilingual-e5-small");

    private static string Name(CapabilityStatus status) => status.ToString().ToLowerInvariant();

    [Fact]
    public void EveryStableProfileOfTheFamilyHasACodeTable()
    {
        foreach (var manifest in Family.Where(m => m.Maturity == "stable"))
        {
            Assert.True(E5Capabilities.Profiles.ContainsKey(manifest.Id), $"{manifest.Id} is stable in the registry but has no E5Capabilities.Profiles entry");
        }

        Assert.Equal([E5Capabilities.StableProfile], E5Capabilities.Profiles.Keys);
    }

    [Fact]
    public void ProfileTableEqualsTheManifestKeyForKey()
    {
        foreach (var (profile, table) in E5Capabilities.Profiles)
        {
            var manifest = Family.Single(m => m.Id == profile);
            Assert.Equal(manifest.Capabilities.Keys.Order(StringComparer.Ordinal), table.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in manifest.Capabilities)
            {
                Assert.True(status == Name(table[key]), $"{profile} {key}: manifest {status}, E5Capabilities.Profiles {Name(table[key])}");
            }

            // The validator refuses Stable with an untested capability; the code table follows the same rule.
            Assert.DoesNotContain(CapabilityStatus.Untested, table.Values);
        }
    }

    [Fact]
    public void PackageTableAgreesWithTheStableProfile()
    {
        var table = E5Capabilities.Profiles[E5Capabilities.StableProfile];
        Assert.Equal(table.Keys.Order(StringComparer.Ordinal), PackageKeys.Keys.Order(StringComparer.Ordinal));
        foreach (var (key, packageKeys) in PackageKeys)
        {
            foreach (string packageKey in packageKeys)
            {
                Assert.True(table[key] == E5Capabilities.All[packageKey], $"{key}: profile {table[key]}, E5Capabilities.All[{packageKey}] {E5Capabilities.All[packageKey]}");
            }
        }

        var stable = Family.Single(m => m.Id == E5Capabilities.StableProfile);
        // Every gate passed, but Stable is blocked (the weekly hosted soak fails, see the README): experimental.
        Assert.Equal("experimental", stable.Compatibility["linux-x64 cpu"]);
        Assert.Equal(stable.Compatibility["windows-x64 cpu"], Name(E5Capabilities.All["windows"]));
        Assert.Equal(CapabilityStatus.Experimental, E5Capabilities.All["embedding.pretokenized.cpu"]);
        Assert.Equal(CapabilityStatus.Experimental, E5Capabilities.All["embedding.pretokenized.cuda"]);
        Assert.DoesNotContain(CapabilityStatus.Untested, E5Capabilities.All.Values);
    }
}
