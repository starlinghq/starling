using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// The capability table of each Stable profile equals its registry manifest key for key, and the package-wide table
/// agrees with it. Weight-free; runs in CI.
/// </summary>
public class StableContractTests
{
    // Per Stable profile: manifest capability -> the package-wide keys (EmbeddingGemma2Capabilities.All) that carry the
    // same status on that profile (keys without a provider suffix: the CPU profiles; .cuda: the CUDA profiles). An empty list means the package table has no key with that profile's status: the
    // text profile records image and video unsupported, while in the package table they are the image and video
    // profiles' capabilities (and the reverse for text-only rows or features shared by the profiles).
    private static readonly Dictionary<string, Dictionary<string, string[]>> PackageKeys = new(StringComparer.Ordinal)
    {
        [EmbeddingGemma2Capabilities.StableProfile] = new(StringComparer.Ordinal)
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
        },
        [EmbeddingGemma2Capabilities.StableImageProfile] = new(StringComparer.Ordinal)
        {
            ["image"] = ["image", "image.cpu"],
            ["image_interleaved_text"] = ["image.interleaved_text"],
            ["image_multiple_per_input"] = ["image.multiple_per_input"],
            ["image_prompts"] = ["image.prompts"],
            ["image_budget_280"] = ["image.budget_280"],
            ["image_budget_other"] = ["image.budget_other"],
            ["image_batch"] = ["image.batch"],
            ["image_matryoshka"] = ["image.matryoshka"],
            ["image_decode_png_jpeg"] = ["image.decode.png_jpeg"],
            ["image_decode_gif_bmp_webp"] = ["image.decode.gif_bmp_webp"],
            ["image_decode_gif_transparency"] = ["image.decode.gif_transparency"],
            ["image_exif_orientation_opt_in"] = ["image.exif_orientation_opt_in"],
            ["image_16_bit"] = ["image.16_bit_per_channel"],
            ["image_alpha_compositing"] = ["image.alpha_compositing"],
            ["image_over_default_pixel_limit_with_option"] = [],
            ["text_only_inputs"] = ["embedding.text"],
            ["batch"] = ["batch", "image.batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation"],
            ["persistence"] = ["mlnet.persistence"],
            ["training"] = ["training"],
            ["aot"] = ["aot"],
            ["extensions_ai"] = ["extensions_ai.image_embedding_generator"],
            ["mlnet"] = ["mlnet.image_column"],
            ["concurrent_calls"] = ["concurrent_calls"],
            ["input_over_8192_tokens"] = ["image.over_8192_tokens"],
            ["audio"] = ["audio"],
            ["video"] = [],
        },
        [EmbeddingGemma2Capabilities.StableCudaProfile] = new(StringComparer.Ordinal)
        {
            ["batch"] = ["batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation"],
            ["persistence"] = ["mlnet.persistence"],
            ["training"] = ["training", "mlnet.training"],
            ["aot"] = ["aot.cuda"],
            ["extensions_ai"] = ["extensions_ai.embedding_generator.cuda", "extensions_ai.dimensions_matryoshka"],
            ["mlnet"] = ["mlnet.embedding_transform.cuda", "mlnet.batching"],
            ["embedding_pretokenized"] = ["embedding.pretokenized.cuda"],
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
        },
        [EmbeddingGemma2Capabilities.StableCudaImageProfile] = new(StringComparer.Ordinal)
        {
            ["image"] = ["image", "image.cuda"],
            ["image_interleaved_text"] = ["image.interleaved_text"],
            ["image_multiple_per_input"] = ["image.multiple_per_input"],
            ["image_prompts"] = ["image.prompts"],
            ["image_budget_280"] = ["image.budget_280"],
            ["image_budget_other"] = ["image.budget_other"],
            ["image_batch"] = ["image.batch"],
            ["image_matryoshka"] = ["image.matryoshka"],
            ["image_decode_png_jpeg"] = ["image.decode.png_jpeg"],
            ["image_decode_gif_bmp_webp"] = ["image.decode.gif_bmp_webp"],
            ["image_decode_gif_transparency"] = ["image.decode.gif_transparency"],
            ["image_exif_orientation_opt_in"] = ["image.exif_orientation_opt_in"],
            ["image_16_bit"] = ["image.16_bit_per_channel"],
            ["image_alpha_compositing"] = ["image.alpha_compositing"],
            ["image_over_default_pixel_limit_with_option"] = [],
            ["text_only_inputs"] = ["embedding.text"],
            ["batch"] = ["batch", "image.batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation"],
            ["persistence"] = ["mlnet.persistence"],
            ["training"] = ["training"],
            ["aot"] = ["aot.cuda"],
            ["extensions_ai"] = ["extensions_ai.image_embedding_generator.cuda"],
            ["mlnet"] = ["mlnet.image_column"],
            ["concurrent_calls"] = ["concurrent_calls"],
            ["input_over_8192_tokens"] = ["image.over_8192_tokens"],
            ["audio"] = ["audio"],
            ["video"] = [],
        },
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

        Assert.Equal(
            [
                EmbeddingGemma2Capabilities.StableImageProfile,
                EmbeddingGemma2Capabilities.StableCudaImageProfile,
                EmbeddingGemma2Capabilities.StableProfile,
                EmbeddingGemma2Capabilities.StableCudaProfile,
            ],
            EmbeddingGemma2Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(
            Family.Where(m => m.Maturity == "stable").Select(m => m.Id).Order(StringComparer.Ordinal),
            EmbeddingGemma2Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
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
    public void PackageTableAgreesWithTheStableProfiles()
    {
        Assert.Equal(EmbeddingGemma2Capabilities.Profiles.Keys.Order(StringComparer.Ordinal), PackageKeys.Keys.Order(StringComparer.Ordinal));
        foreach (var (profile, keys) in PackageKeys)
        {
            var table = EmbeddingGemma2Capabilities.Profiles[profile];
            Assert.Equal(table.Keys.Order(StringComparer.Ordinal), keys.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, packageKeys) in keys)
            {
                foreach (string packageKey in packageKeys)
                {
                    Assert.True(table[key] == EmbeddingGemma2Capabilities.All[packageKey], $"{profile} {key}: profile {table[key]}, EmbeddingGemma2Capabilities.All[{packageKey}] {EmbeddingGemma2Capabilities.All[packageKey]}");
                }
            }

            var manifest = Family.Single(m => m.Id == profile);
            Assert.Equal("stable", manifest.Maturity);
            Assert.Equal("supported", manifest.Compatibility[profile.EndsWith("-cuda", StringComparison.Ordinal) ? "linux-x64 cuda" : "linux-x64 cpu"]);
            Assert.DoesNotContain(EmbeddingGemma2CapabilityStatus.Untested, table.Values);
        }

        var stable = Family.Single(m => m.Id == EmbeddingGemma2Capabilities.StableProfile);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, EmbeddingGemma2Capabilities.All["embedding.pretokenized.cpu"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, EmbeddingGemma2Capabilities.All["embedding.text"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, EmbeddingGemma2Capabilities.All["image.cpu"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["video.cpu"]);
        // Outside the contracts: proposed rows and opt-in deviations (experimental), the video profiles (not Stable).
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["input.2050_to_8192_tokens"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["prompts.task"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["image.budget_other"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["video.hdr"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, EmbeddingGemma2Capabilities.All["image.cuda"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["video.cuda"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, EmbeddingGemma2Capabilities.All["embedding.pretokenized.cuda"]);
        Assert.Equal("experimental", Family.Single(m => m.Id == "embeddinggemma-2/video-fp32-cuda").Maturity);
        Assert.Equal(stable.Compatibility["windows-x64 cpu"], Name(EmbeddingGemma2Capabilities.All["windows"]));
    }
}
