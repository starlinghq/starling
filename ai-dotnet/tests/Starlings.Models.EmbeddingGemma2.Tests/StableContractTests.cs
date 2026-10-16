using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// The capability table of the Stable-contract profile equals its registry manifest key for key, and the package-wide
/// table agrees with it. Weight-free; runs in CI.
/// </summary>
public class StableContractTests
{
    // Manifest capability -> the package-wide keys (EmbeddingGemma2Capabilities.All) that carry the same status on the
    // CPU text profile. image and video have none: in the package table they are the experimental image and video
    // profiles' capabilities, while the text profile records them unsupported.
    private static readonly Dictionary<string, string[]> PackageKeys = new(StringComparer.Ordinal)
    {
        ["batch"] = ["batch"],
        ["streaming"] = ["streaming"],
        ["cancellation"] = ["cancellation"],
        ["persistence"] = ["mlnet.persistence"],
        ["training"] = ["training", "mlnet.training"],
        ["aot"] = ["aot"],
        ["extensions_ai"] = ["extensions_ai.embedding_generator", "extensions_ai.dimensions_matryoshka"],
        ["mlnet"] = ["mlnet.embedding_transform", "mlnet.batching"],
        ["embedding_pretokenized"] = ["embedding.pretokenized.cpu"],
        ["embedding_text"] = ["embedding.text"],
        ["tokenizer_owned"] = ["tokenizer.owned"],
        ["prompts_query_document"] = ["prompts.query_document"],
        ["prompts_task"] = ["prompts.task"],
        ["matryoshka"] = ["matryoshka.512", "matryoshka.256", "matryoshka.128"],
        ["input_up_to_2049_tokens"] = ["input.up_to_2049_tokens"],
        ["input_2050_to_8192_tokens"] = ["input.2050_to_8192_tokens"],
        ["input_over_8192_tokens"] = ["input.over_8192_tokens", "truncation"],
        ["concurrent_calls"] = ["concurrent_calls"],
        ["image"] = [],
        ["audio"] = ["audio"],
        ["video"] = [],
    };

    private static IReadOnlyList<RegistryManifest> Family { get; } = RegistryManifest.Family(EmbeddingGemma2TestAssets.RepositoryRoot, "embeddinggemma-2");

    private static string Name(EmbeddingGemma2CapabilityStatus status) => status.ToString().ToLowerInvariant();

    [Fact]
    public void EveryStableProfileOfTheFamilyHasACodeTable()
    {
        foreach (var manifest in Family.Where(m => m.Maturity == "stable"))
        {
            Assert.True(EmbeddingGemma2Capabilities.Profiles.ContainsKey(manifest.Id), $"{manifest.Id} is stable in the registry but has no EmbeddingGemma2Capabilities.Profiles entry");
        }

        Assert.Equal([EmbeddingGemma2Capabilities.StableProfile], EmbeddingGemma2Capabilities.Profiles.Keys);
    }

    [Fact]
    public void ProfileTableEqualsTheManifestKeyForKey()
    {
        foreach (var (profile, table) in EmbeddingGemma2Capabilities.Profiles)
        {
            var manifest = Family.Single(m => m.Id == profile);
            Assert.Equal(manifest.Capabilities.Keys.Order(StringComparer.Ordinal), table.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in manifest.Capabilities)
            {
                Assert.True(status == Name(table[key]), $"{profile} {key}: manifest {status}, EmbeddingGemma2Capabilities.Profiles {Name(table[key])}");
            }

            Assert.DoesNotContain(EmbeddingGemma2CapabilityStatus.Untested, table.Values);
        }
    }

    [Fact]
    public void PackageTableAgreesWithTheStableProfile()
    {
        var table = EmbeddingGemma2Capabilities.Profiles[EmbeddingGemma2Capabilities.StableProfile];
        Assert.Equal(table.Keys.Order(StringComparer.Ordinal), PackageKeys.Keys.Order(StringComparer.Ordinal));
        foreach (var (key, packageKeys) in PackageKeys)
        {
            foreach (string packageKey in packageKeys)
            {
                Assert.True(table[key] == EmbeddingGemma2Capabilities.All[packageKey], $"{key}: profile {table[key]}, EmbeddingGemma2Capabilities.All[{packageKey}] {EmbeddingGemma2Capabilities.All[packageKey]}");
            }
        }

        var stable = Family.Single(m => m.Id == EmbeddingGemma2Capabilities.StableProfile);
        Assert.Equal("stable", stable.Maturity);
        Assert.Equal("supported", stable.Compatibility["linux-x64 cpu"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, EmbeddingGemma2Capabilities.All["embedding.pretokenized.cpu"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, EmbeddingGemma2Capabilities.All["embedding.text"]);
        // Outside the contract: proposed rows (experimental) and the image and video profiles (not Stable).
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["input.2050_to_8192_tokens"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["prompts.task"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["image.cpu"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["video.cpu"]);
        Assert.DoesNotContain(EmbeddingGemma2CapabilityStatus.Untested, table.Values);
        Assert.Equal(stable.Compatibility["windows-x64 cpu"], Name(EmbeddingGemma2Capabilities.All["windows"]));
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["embedding.pretokenized.cuda"]);
    }
}
