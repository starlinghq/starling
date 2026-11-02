using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// The capability table of each Stable profile equals its registry manifest key for key, and the package-wide table
/// agrees with it. Weight-free; runs in CI.
/// </summary>
public class StableContractTests
{
    // Per Stable profile: manifest capability -> the package-wide keys (Gliner2Capabilities.Statuses) that carry the same status
    // on that profile. Keys without a provider suffix describe the CPU profile, keys with .cuda the CUDA profile; shared
    // keys are listed where both profiles have the same status.
    private static readonly Dictionary<string, Dictionary<string, string[]>> PackageKeys = new(StringComparer.Ordinal)
    {
        [Gliner2Capabilities.StableProfile] = new(StringComparer.Ordinal)
        {
            ["batch"] = ["batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation"],
            ["persistence"] = ["persistence"],
            ["training"] = ["training"],
            ["aot"] = ["aot"],
            ["extensions_ai"] = ["extensions_ai"],
            ["mlnet"] = ["mlnet"],
            ["entity_extraction"] = ["entity_extraction.cpu", "entity_extraction.staged", "threshold"],
            ["offsets_original_text"] = ["offsets.original_text"],
            ["concurrent_calls"] = ["concurrent_calls"],
            ["label_descriptions"] = ["label_descriptions"],
            ["count_gate_closed"] = ["count_gate.closed"],
            ["input_over_512_tokens"] = ["input_over_512_tokens", "truncation", "text.truncation"],
            ["plain_deduplicated_output"] = ["plain_deduplicated_output"],
            ["mixed_label_sets_in_batch"] = ["batch.mixed_label_sets"],
            ["classification"] = ["classification"],
            ["relation_extraction"] = ["relation_extraction"],
            ["structured_extraction"] = ["structured_extraction"],
        },
        [Gliner2Capabilities.StableCudaProfile] = new(StringComparer.Ordinal)
        {
            ["batch"] = ["batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation"],
            ["persistence"] = ["persistence"],
            ["training"] = ["training"],
            ["aot"] = ["aot.cuda"],
            ["extensions_ai"] = ["extensions_ai"],
            ["mlnet"] = ["mlnet"],
            ["entity_extraction"] = ["entity_extraction.cuda", "entity_extraction.staged", "threshold"],
            ["offsets_original_text"] = ["offsets.original_text"],
            ["concurrent_calls"] = ["concurrent_calls.cuda.serialized"],
            ["concurrent_calls_unserialized"] = ["concurrent_calls.cuda.unserialized"],
            ["label_descriptions"] = ["label_descriptions"],
            ["count_gate_closed"] = ["count_gate.closed"],
            ["input_over_512_tokens"] = ["input_over_512_tokens", "truncation", "text.truncation"],
            ["plain_deduplicated_output"] = ["plain_deduplicated_output"],
            ["mixed_label_sets_in_batch"] = ["batch.mixed_label_sets"],
            ["classification"] = ["classification.cuda"],
            ["relation_extraction"] = ["relation_extraction"],
            ["structured_extraction"] = ["structured_extraction"],
        },
    };

    private static IReadOnlyList<RegistryManifest> Family { get; } = RegistryManifest.Family(Gliner2TestAssets.RepositoryRoot, "gliner2-base-v1");

    private static string Name(Status status) => status.ToString().ToLowerInvariant();

    // The capability tables of the Stable profiles, from the generated records.
    private static IReadOnlyDictionary<string, IReadOnlyDictionary<string, Status>> StableTables { get; } =
        Gliner2Capabilities.ProfileCapabilities.Values
            .Where(p => p.Maturity == ProfileMaturity.Stable)
            .ToDictionary(p => p.ProfileId, p => p.Capabilities, StringComparer.Ordinal);

    [Fact]
    public void GeneratedRecordsEqualTheManifestsFieldForField() =>
        CapabilityRecordChecks.AssertRecordsEqualManifests(Gliner2Capabilities.ProfileCapabilities, Family);

    [Fact]
    public void ObsoleteMembersEqualTheNewOnes()
    {
#pragma warning disable CS0618 // The former members stay for one minor version and must keep their values.
        Assert.Equal(Gliner2Capabilities.Statuses.Keys.Order(StringComparer.Ordinal), Gliner2Capabilities.All.Keys.Order(StringComparer.Ordinal));
        foreach (var (key, status) in Gliner2Capabilities.Statuses)
        {
            Assert.Equal(status.ToString(), Gliner2Capabilities.All[key].ToString());
            Assert.Equal((int)status, (int)Gliner2Capabilities.All[key]);
        }

        Assert.Equal(StableTables.Keys.Order(StringComparer.Ordinal), Gliner2Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
        foreach (var (profile, table) in StableTables)
        {
            Assert.Equal(table.Keys.Order(StringComparer.Ordinal), Gliner2Capabilities.Profiles[profile].Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in table)
            {
                Assert.Equal(status.ToString(), Gliner2Capabilities.Profiles[profile][key].ToString());
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
            [Gliner2Capabilities.StableProfile, Gliner2Capabilities.StableCudaProfile],
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
                    Assert.True(table[key] == Gliner2Capabilities.Statuses[packageKey], $"{profile} {key}: profile {table[key]}, Gliner2Capabilities.Statuses[{packageKey}] {Gliner2Capabilities.Statuses[packageKey]}");
                }
            }

            Assert.Equal("stable", Family.Single(m => m.Id == profile).Maturity);
        }

        var stable = Family.Single(m => m.Id == Gliner2Capabilities.StableProfile);
        var cuda = Family.Single(m => m.Id == Gliner2Capabilities.StableCudaProfile);
        Assert.Equal("supported", stable.Compatibility["linux-x64 cpu"]);
        Assert.Equal("supported", cuda.Compatibility["linux-x64 cuda"]);
        // The CPU manifest's CUDA row mirrors the CUDA profile.
        Assert.Equal(stable.Compatibility["linux-x64 cuda"], Name(Gliner2Capabilities.Statuses["entity_extraction.cuda"]));
        Assert.Equal(cuda.Compatibility["linux-x64 cuda"], Name(Gliner2Capabilities.Statuses["entity_extraction.cuda"]));
        // Windows: CI weight-free evidence only (experimental), the same in the code table and the manifest.
        Assert.Equal(stable.Compatibility["windows-x64 cpu"], Name(Gliner2Capabilities.Statuses["windows"]));
        Assert.DoesNotContain(Status.Untested, Gliner2Capabilities.Statuses.Values);
    }
}
