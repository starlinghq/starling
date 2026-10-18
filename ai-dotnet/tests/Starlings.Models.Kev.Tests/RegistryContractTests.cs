using Starlings.Models.Kev;
using Starlings.TestSupport;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// The capability table of each Kev profile equals its registry manifest key for key, and the package-wide table agrees
/// with it. Weight-free; runs in CI.
/// </summary>
public sealed class RegistryContractTests
{
    // Manifest capability -> the package-wide keys (KevCapabilities.All) that carry the same status on that provider. Keys
    // without a provider suffix describe the CPU profiles, keys with .cuda the CUDA profiles. streaming and persistence
    // have no package-wide row (the package offers neither); their profile status is checked against the manifest only.
    private static readonly Dictionary<string, string[]> CpuKeys = new(StringComparer.Ordinal)
    {
        ["batch"] = ["batched_rows"],
        ["streaming"] = [],
        ["cancellation"] = ["cancellation"],
        ["persistence"] = [],
        ["training"] = ["training"],
        ["aot"] = ["aot"],
        ["extensions_ai"] = ["extensions_ai"],
        ["mlnet"] = ["mlnet"],
        ["decision_noul"] = ["decision.noul"],
        ["decision_choice"] = ["decision.choice"],
        ["decision_score"] = ["decision.score"],
        ["calibrated_temperature"] = ["calibrated_temperature"],
        ["raw_probabilities"] = ["raw_probabilities"],
        ["delimiter_escaping"] = ["delimiter_escaping"],
        ["question_isolation"] = ["question_isolation"],
        ["state_cache_within_request"] = ["state_cache_within_request"],
        ["state_json_rendering"] = ["state_json_rendering"],
        ["concurrent_calls"] = ["concurrent_calls"],
        ["asset_verification_offline"] = ["asset_verification.offline"],
        ["state_over_8192_tokens"] = ["state_over_8192_tokens", "truncation"],
        ["asset_acquisition_download"] = ["asset_acquisition.download"],
        ["state_prefix_reuse_across_requests"] = ["state_prefix_reuse_across_requests"],
        ["probability_rounding"] = ["probability_rounding"],
        ["usage_output_tokens"] = ["usage_output_tokens"],
        ["date_facts"] = ["date_facts"],
        ["option_isolation"] = ["option_isolation"],
        ["http_server"] = ["http_server"],
        ["bf16"] = ["bf16"],
        ["fp16"] = ["fp16"],
        ["quantized"] = ["quantized"],
        ["calibration_fitting"] = ["calibration_fitting"],
        ["generation"] = ["generation"],
        ["non_english"] = ["non_english"],
    };

    private static readonly Dictionary<string, string[]> CudaKeys = new(CpuKeys, StringComparer.Ordinal)
    {
        ["concurrent_calls"] = ["concurrent_calls.cuda.serialized"],
        ["concurrent_calls_unserialized"] = ["concurrent_calls.cuda.unserialized"],
        ["tf32"] = ["tf32"],
    };

    private static readonly Dictionary<string, Dictionary<string, string[]>> PackageKeys = new(StringComparer.Ordinal)
    {
        [KevCapabilities.Kev08BCpuProfile] = CpuKeys,
        [KevCapabilities.Kev08BCudaProfile] = CudaKeys,
        [KevCapabilities.Kev4BCpuProfile] = CpuKeys,
        [KevCapabilities.Kev4BCudaProfile] = CudaKeys,
    };

    private static IReadOnlyList<RegistryManifest> Manifests { get; } =
        [.. RegistryManifest.Family(KevTestAssets.RepositoryRoot, "kev-0.8b"), .. RegistryManifest.Family(KevTestAssets.RepositoryRoot, "kev-4b")];

    private static string Name(KevCapabilityStatus status) => status.ToString().ToLowerInvariant();

    [Fact]
    public void EveryRegisteredProfileHasACodeTable()
    {
        Assert.Equal(
            Manifests.Select(m => m.Id).Order(StringComparer.Ordinal),
            KevCapabilities.Profiles.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(
            KevModelProfile.All.SelectMany(p => new[] { p.Id, p.CudaId }).Order(StringComparer.Ordinal),
            KevCapabilities.Profiles.Keys.Order(StringComparer.Ordinal));
        Assert.All(Manifests, m => Assert.Equal("experimental", m.Maturity));
    }

    [Fact]
    public void ProfileTableEqualsTheManifestKeyForKey()
    {
        foreach (var (profile, table) in KevCapabilities.Profiles)
        {
            var manifest = Manifests.Single(m => m.Id == profile);
            Assert.Equal(manifest.Capabilities.Keys.Order(StringComparer.Ordinal), table.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in manifest.Capabilities)
            {
                Assert.True(status == Name(table[key]), $"{profile} {key}: manifest {status}, KevCapabilities.Profiles {Name(table[key])}");
            }

            // Experimental profiles: nothing is supported.
            Assert.DoesNotContain(KevCapabilityStatus.Supported, table.Values);
        }
    }

    [Fact]
    public void PackageTableAgreesWithTheProfiles()
    {
        Assert.Equal(KevCapabilities.Profiles.Keys.Order(StringComparer.Ordinal), PackageKeys.Keys.Order(StringComparer.Ordinal));
        foreach (var (profile, keys) in PackageKeys)
        {
            var table = KevCapabilities.Profiles[profile];
            Assert.Equal(table.Keys.Order(StringComparer.Ordinal), keys.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, packageKeys) in keys)
            {
                foreach (string packageKey in packageKeys)
                {
                    Assert.True(
                        table[key] == KevCapabilities.All[packageKey],
                        $"{profile} {key}: profile {table[key]}, KevCapabilities.All[{packageKey}] {KevCapabilities.All[packageKey]}");
                }
            }
        }

        // Platforms: every CUDA row mirrors the package's CUDA status, Windows has no evidence.
        foreach (var model in KevModelProfile.All)
        {
            var cpu = Manifests.Single(m => m.Id == model.Id);
            var cuda = Manifests.Single(m => m.Id == model.CudaId);
            Assert.Equal(Name(KevCapabilityStatus.Experimental), cpu.Compatibility["linux-x64 cpu"]);
            Assert.Equal(Name(KevCapabilities.All["cuda"]), cpu.Compatibility["linux-x64 cuda"]);
            Assert.Equal(Name(KevCapabilities.All["cuda"]), cuda.Compatibility["linux-x64 cuda"]);
            Assert.Equal(Name(KevCapabilities.All["windows"]), cpu.Compatibility["windows-x64 cpu"]);
            Assert.Equal(Name(KevCapabilities.All["windows"]), cuda.Compatibility["windows-x64 cuda"]);
            Assert.False(cuda.Compatibility.ContainsKey("linux-x64 cpu"), $"{model.CudaId} records a CPU row");
        }
    }
}
