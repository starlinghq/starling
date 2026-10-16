using Starlings.TestSupport;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// The capability table of each Stable profile equals its registry manifest key for key, and the package-wide table
/// agrees with it. Weight-free; runs in CI.
/// </summary>
public class StableContractTests
{
    // Per Stable profile: manifest capability -> the package-wide keys (Gliner2Capabilities.All) that carry the same status
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
            ["input_over_512_tokens"] = ["input_over_512_tokens", "truncation"],
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
            ["input_over_512_tokens"] = ["input_over_512_tokens", "truncation"],
            ["plain_deduplicated_output"] = ["plain_deduplicated_output"],
            ["mixed_label_sets_in_batch"] = ["batch.mixed_label_sets"],
            ["classification"] = ["classification"],
            ["relation_extraction"] = ["relation_extraction"],
            ["structured_extraction"] = ["structured_extraction"],
        },
    };

    private static IReadOnlyList<RegistryManifest> Family { get; } = RegistryManifest.Family(Gliner2TestAssets.RepositoryRoot, "gliner2-base-v1");

    private static string Name(Gliner2CapabilityStatus status) => status.ToString().ToLowerInvariant();

    [Fact]
    public void EveryStableProfileOfTheFamilyHasACodeTable()
    {
        foreach (var manifest in Family.Where(m => m.Maturity == "stable"))
        {
            Assert.True(Gliner2Capabilities.Profiles.ContainsKey(manifest.Id), $"{manifest.Id} is stable in the registry but has no Gliner2Capabilities.Profiles entry");
        }

        Assert.Equal(
            [Gliner2Capabilities.StableProfile, Gliner2Capabilities.StableCudaProfile],
            Gliner2Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(
            Family.Where(m => m.Maturity == "stable").Select(m => m.Id).Order(StringComparer.Ordinal),
            Gliner2Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void ProfileTableEqualsTheManifestKeyForKey()
    {
        foreach (var (profile, table) in Gliner2Capabilities.Profiles)
        {
            var manifest = Family.Single(m => m.Id == profile);
            Assert.Equal(manifest.Capabilities.Keys.Order(StringComparer.Ordinal), table.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in manifest.Capabilities)
            {
                Assert.True(status == Name(table[key]), $"{profile} {key}: manifest {status}, Gliner2Capabilities.Profiles {Name(table[key])}");
            }

            Assert.DoesNotContain(Gliner2CapabilityStatus.Untested, table.Values);
        }
    }

    [Fact]
    public void PackageTableAgreesWithTheStableProfiles()
    {
        Assert.Equal(Gliner2Capabilities.Profiles.Keys.Order(StringComparer.Ordinal), PackageKeys.Keys.Order(StringComparer.Ordinal));
        foreach (var (profile, keys) in PackageKeys)
        {
            var table = Gliner2Capabilities.Profiles[profile];
            Assert.Equal(table.Keys.Order(StringComparer.Ordinal), keys.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, packageKeys) in keys)
            {
                foreach (string packageKey in packageKeys)
                {
                    Assert.True(table[key] == Gliner2Capabilities.All[packageKey], $"{profile} {key}: profile {table[key]}, Gliner2Capabilities.All[{packageKey}] {Gliner2Capabilities.All[packageKey]}");
                }
            }

            Assert.Equal("stable", Family.Single(m => m.Id == profile).Maturity);
        }

        var stable = Family.Single(m => m.Id == Gliner2Capabilities.StableProfile);
        var cuda = Family.Single(m => m.Id == Gliner2Capabilities.StableCudaProfile);
        Assert.Equal("supported", stable.Compatibility["linux-x64 cpu"]);
        Assert.Equal("supported", cuda.Compatibility["linux-x64 cuda"]);
        // The CPU manifest's CUDA row mirrors the CUDA profile.
        Assert.Equal(stable.Compatibility["linux-x64 cuda"], Name(Gliner2Capabilities.All["entity_extraction.cuda"]));
        Assert.Equal(cuda.Compatibility["linux-x64 cuda"], Name(Gliner2Capabilities.All["entity_extraction.cuda"]));
        // Windows: CI weight-free evidence only (experimental), the same in the code table and the manifest.
        Assert.Equal(stable.Compatibility["windows-x64 cpu"], Name(Gliner2Capabilities.All["windows"]));
        Assert.DoesNotContain(Gliner2CapabilityStatus.Untested, Gliner2Capabilities.All.Values);
    }
}
