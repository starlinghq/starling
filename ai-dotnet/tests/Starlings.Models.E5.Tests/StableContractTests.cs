using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// The capability table of each Stable profile equals its registry manifest key for key, and the package-wide
/// table agrees with it. Weight-free; runs in CI.
/// </summary>
public class StableContractTests
{
    // Per Stable profile: manifest capability -> the package-wide keys (E5Capabilities.All) that carry the same status on
    // that profile (keys without a provider suffix: the CPU profile; .cuda: the CUDA profile).
    private static readonly Dictionary<string, Dictionary<string, string[]>> PackageKeys = new(StringComparer.Ordinal)
    {
        [E5Capabilities.StableProfile] = new(StringComparer.Ordinal)
        {
            ["batch"] = ["batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation"],
            ["persistence"] = ["persistence", "mlnet.persistence"],
            ["training"] = ["training", "mlnet.training"],
            ["aot"] = ["aot"],
            ["extensions_ai"] = ["extensions_ai.embedding_generator"],
            ["mlnet"] = ["mlnet.embedding_transform"],
        },
        [E5Capabilities.StableCudaProfile] = new(StringComparer.Ordinal)
        {
            ["batch"] = ["batch", "embedding.pretokenized.cuda"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation"],
            ["persistence"] = ["persistence", "mlnet.persistence"],
            ["training"] = ["training", "mlnet.training"],
            ["aot"] = ["aot.cuda"],
            ["extensions_ai"] = ["extensions_ai.embedding_generator.cuda"],
            ["mlnet"] = ["mlnet.embedding_transform.cuda"],
        },
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

        Assert.Equal(
            [E5Capabilities.StableProfile, E5Capabilities.StableCudaProfile],
            E5Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(
            Family.Where(m => m.Maturity == "stable").Select(m => m.Id).Order(StringComparer.Ordinal),
            E5Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
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
    public void PackageTableAgreesWithTheStableProfiles()
    {
        Assert.Equal(E5Capabilities.Profiles.Keys.Order(StringComparer.Ordinal), PackageKeys.Keys.Order(StringComparer.Ordinal));
        foreach (var (profile, keys) in PackageKeys)
        {
            var table = E5Capabilities.Profiles[profile];
            Assert.Equal(table.Keys.Order(StringComparer.Ordinal), keys.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, packageKeys) in keys)
            {
                foreach (string packageKey in packageKeys)
                {
                    Assert.True(table[key] == E5Capabilities.All[packageKey], $"{profile} {key}: profile {table[key]}, E5Capabilities.All[{packageKey}] {E5Capabilities.All[packageKey]}");
                }
            }

            Assert.Equal("stable", Family.Single(m => m.Id == profile).Maturity);
        }

        var cuda = Family.Single(m => m.Id == E5Capabilities.StableCudaProfile);
        Assert.Equal("supported", cuda.Compatibility["linux-x64 cuda"]);
        Assert.Equal(CapabilityStatus.Supported, E5Capabilities.All["embedding.pretokenized.cuda"]);

        var stable = Family.Single(m => m.Id == E5Capabilities.StableProfile);
        Assert.Equal("stable", stable.Maturity);
        Assert.Equal("supported", stable.Compatibility["linux-x64 cpu"]);
        Assert.Equal(CapabilityStatus.Supported, E5Capabilities.All["embedding.pretokenized.cpu"]);
        Assert.Equal(CapabilityStatus.Supported, E5Capabilities.All["embedding.text"]);
        Assert.Equal(stable.Compatibility["windows-x64 cpu"], Name(E5Capabilities.All["windows"]));
        Assert.DoesNotContain(CapabilityStatus.Untested, E5Capabilities.All.Values);
    }
}
