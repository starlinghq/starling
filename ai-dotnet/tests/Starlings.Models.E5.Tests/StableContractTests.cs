using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// The capability table of each Stable profile equals its registry manifest key for key, and the package-wide
/// table agrees with it. Weight-free; runs in CI.
/// </summary>
public class StableContractTests
{
    // Per Stable profile: manifest capability -> the package-wide keys (E5Capabilities.Statuses) that carry the same status on
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

    private static string Name(Status status) => status.ToString().ToLowerInvariant();

    // The capability tables of the Stable profiles, from the generated records.
    private static IReadOnlyDictionary<string, IReadOnlyDictionary<string, Status>> StableTables { get; } =
        E5Capabilities.ProfileCapabilities.Values
            .Where(p => p.Maturity == ProfileMaturity.Stable)
            .ToDictionary(p => p.ProfileId, p => p.Capabilities, StringComparer.Ordinal);

    [Fact]
    public void GeneratedRecordsEqualTheManifestsFieldForField() =>
        CapabilityRecordChecks.AssertRecordsEqualManifests(E5Capabilities.ProfileCapabilities, Family.Where(m => m.Id is E5Capabilities.StableProfile or E5Capabilities.StableCudaProfile));

    [Fact]
    public void ObsoleteMembersEqualTheNewOnes()
    {
#pragma warning disable CS0618 // The former members stay for one minor version and must keep their values.
        Assert.Equal(E5Capabilities.Statuses.Keys.Order(StringComparer.Ordinal), E5Capabilities.All.Keys.Order(StringComparer.Ordinal));
        foreach (var (key, status) in E5Capabilities.Statuses)
        {
            Assert.Equal(status.ToString(), E5Capabilities.All[key].ToString());
            Assert.Equal((int)status, (int)E5Capabilities.All[key]);
        }

        Assert.Equal(StableTables.Keys.Order(StringComparer.Ordinal), E5Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
        foreach (var (profile, table) in StableTables)
        {
            Assert.Equal(table.Keys.Order(StringComparer.Ordinal), E5Capabilities.Profiles[profile].Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in table)
            {
                Assert.Equal(status.ToString(), E5Capabilities.Profiles[profile][key].ToString());
            }
        }
#pragma warning restore CS0618
    }

    [Fact]
    public void EveryStableProfileOfTheFamilyHasACodeTable()
    {
        foreach (var manifest in Family.Where(m => m.Maturity == "stable"))
        {
            Assert.True(StableTables.ContainsKey(manifest.Id), $"{manifest.Id} is stable in the registry but has no StableTables entry");
        }

        Assert.Equal(
            [E5Capabilities.StableProfile, E5Capabilities.StableCudaProfile],
            StableTables.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(
            Family.Where(m => m.Maturity == "stable").Select(m => m.Id).Order(StringComparer.Ordinal),
            StableTables.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void ProfileTableEqualsTheManifestKeyForKey()
    {
        foreach (var (profile, table) in StableTables)
        {
            var manifest = Family.Single(m => m.Id == profile);
            Assert.Equal(manifest.Capabilities.Keys.Order(StringComparer.Ordinal), table.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in manifest.Capabilities)
            {
                Assert.True(status == Name(table[key]), $"{profile} {key}: manifest {status}, StableTables {Name(table[key])}");
            }

            // The validator refuses Stable with an untested capability; the code table follows the same rule.
            Assert.DoesNotContain(Status.Untested, table.Values);
        }
    }

    [Fact]
    public void PackageTableAgreesWithTheStableProfiles()
    {
        Assert.Equal(StableTables.Keys.Order(StringComparer.Ordinal), PackageKeys.Keys.Order(StringComparer.Ordinal));
        foreach (var (profile, keys) in PackageKeys)
        {
            var table = StableTables[profile];
            Assert.Equal(table.Keys.Order(StringComparer.Ordinal), keys.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, packageKeys) in keys)
            {
                foreach (string packageKey in packageKeys)
                {
                    Assert.True(table[key] == E5Capabilities.Statuses[packageKey], $"{profile} {key}: profile {table[key]}, E5Capabilities.Statuses[{packageKey}] {E5Capabilities.Statuses[packageKey]}");
                }
            }

            Assert.Equal("stable", Family.Single(m => m.Id == profile).Maturity);
        }

        var cuda = Family.Single(m => m.Id == E5Capabilities.StableCudaProfile);
        Assert.Equal("supported", cuda.Compatibility["linux-x64 cuda"]);
        Assert.Equal(Status.Supported, E5Capabilities.Statuses["embedding.pretokenized.cuda"]);

        var stable = Family.Single(m => m.Id == E5Capabilities.StableProfile);
        Assert.Equal("stable", stable.Maturity);
        Assert.Equal("supported", stable.Compatibility["linux-x64 cpu"]);
        Assert.Equal(Status.Supported, E5Capabilities.Statuses["embedding.pretokenized.cpu"]);
        Assert.Equal(Status.Supported, E5Capabilities.Statuses["embedding.text"]);
        Assert.Equal(stable.Compatibility["windows-x64 cpu"], Name(E5Capabilities.Statuses["windows"]));
        Assert.DoesNotContain(Status.Untested, E5Capabilities.Statuses.Values);
    }
}
