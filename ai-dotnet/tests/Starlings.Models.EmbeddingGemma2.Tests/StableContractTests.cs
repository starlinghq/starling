using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// The capability table of each Stable profile equals its registry manifest key for key, and the package-wide table
/// agrees with it. Weight-free; runs in CI.
/// </summary>
public class StableContractTests
{
    // Per Stable profile: manifest capability -> the package-wide keys (EmbeddingGemma2Capabilities.Statuses) that carry the
    // same status on that profile (keys without a provider suffix: the CPU profiles; .cuda: the CUDA profiles). An empty list means the package table has no key with that profile's status: the
    // text profile records image and video unsupported, and the image profile video unsupported, while in the package
    // table they are the image and video profiles' capabilities (and the reverse for text-only rows or features shared
    // by the profiles).
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
            ["input_over_8192_tokens"] = ["input.over_8192_tokens", "truncation", "text.truncation"],
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
        [EmbeddingGemma2Capabilities.StableVideoProfile] = new(StringComparer.Ordinal)
        {
            ["video"] = ["video", "video.cpu"],
            ["video_decoded_frames_with_timing"] = ["video.decoded_frames_with_timing"],
            ["video_pre_sampled_frames"] = ["video.pre_sampled_frames"],
            ["video_frames_without_timing"] = ["video.frames_without_timing"],
            ["video_frames_without_timing_opt_in"] = ["video.frames_without_timing.upstream_opt_in"],
            ["video_interleaved_text"] = ["video.interleaved_text"],
            ["video_interleaved_image"] = ["video.interleaved_image"],
            ["video_multiple_per_input"] = ["video.multiple_per_input"],
            ["video_prompts"] = ["video.prompts"],
            ["video_batch"] = ["video.batch"],
            ["video_matryoshka"] = ["video.matryoshka"],
            ["video_timestamps"] = ["video.timestamps", "video.other_fps_max_frames_or_budget"],
            ["video_variable_frame_rate"] = ["video.variable_frame_rate"],
            ["video_interlaced"] = ["video.interlaced"],
            ["video_hdr"] = ["video.hdr"],
            ["video_audio_track"] = ["video.audio_track"],
            ["video_decode_ffmpeg8"] = ["video.decode.ffmpeg8"],
            ["video_decode_ffmpeg9"] = ["video.decode.ffmpeg9_adapter"],
            ["video_decode_rotation_90"] = ["video.decode.rotation_90"],
            ["video_decode_rotation_other"] = ["video.decode.rotation_other"],
            ["video_decode_streams"] = ["video.decode.streams"],
            ["image"] = ["image", "image.cpu"],
            ["text_only_inputs"] = ["embedding.text"],
            ["batch"] = ["batch", "video.batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation"],
            ["persistence"] = ["mlnet.persistence"],
            ["training"] = ["training"],
            ["aot"] = ["aot"],
            ["extensions_ai"] = ["extensions_ai.video_embedding_generator"],
            ["mlnet"] = ["mlnet.video_column"],
            ["concurrent_calls"] = ["concurrent_calls"],
            ["input_over_8192_tokens"] = ["video.over_8192_tokens"],
            ["audio"] = ["audio"],
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
            ["input_over_8192_tokens"] = ["input.over_8192_tokens", "truncation", "text.truncation"],
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
        [EmbeddingGemma2Capabilities.StableCudaVideoProfile] = new(StringComparer.Ordinal)
        {
            ["video"] = ["video", "video.cuda"],
            ["video_decoded_frames_with_timing"] = ["video.decoded_frames_with_timing"],
            ["video_pre_sampled_frames"] = ["video.pre_sampled_frames"],
            ["video_frames_without_timing"] = ["video.frames_without_timing"],
            ["video_frames_without_timing_opt_in"] = ["video.frames_without_timing.upstream_opt_in"],
            ["video_interleaved_text"] = ["video.interleaved_text"],
            ["video_interleaved_image"] = ["video.interleaved_image"],
            ["video_multiple_per_input"] = ["video.multiple_per_input"],
            ["video_prompts"] = ["video.prompts"],
            ["video_batch"] = ["video.batch"],
            ["video_matryoshka"] = ["video.matryoshka"],
            ["video_timestamps"] = ["video.timestamps", "video.other_fps_max_frames_or_budget"],
            ["video_variable_frame_rate"] = ["video.variable_frame_rate"],
            ["video_interlaced"] = ["video.interlaced"],
            ["video_hdr"] = ["video.hdr"],
            ["video_audio_track"] = ["video.audio_track"],
            ["video_decode_ffmpeg8"] = ["video.decode.ffmpeg8"],
            ["video_decode_ffmpeg9"] = ["video.decode.ffmpeg9_adapter"],
            ["video_decode_rotation_90"] = ["video.decode.rotation_90"],
            ["video_decode_rotation_other"] = ["video.decode.rotation_other"],
            ["video_decode_streams"] = ["video.decode.streams"],
            ["image"] = ["image", "image.cuda"],
            ["text_only_inputs"] = ["embedding.text"],
            ["batch"] = ["batch", "video.batch"],
            ["streaming"] = ["streaming"],
            ["cancellation"] = ["cancellation"],
            ["persistence"] = ["mlnet.persistence"],
            ["training"] = ["training"],
            ["aot"] = ["aot.cuda"],
            ["extensions_ai"] = ["extensions_ai.video_embedding_generator.cuda"],
            ["mlnet"] = ["mlnet.video_column"],
            ["concurrent_calls"] = ["concurrent_calls"],
            ["input_over_8192_tokens"] = ["video.over_8192_tokens"],
            ["audio"] = ["audio"],
        },
    };

    private static IReadOnlyList<RegistryManifest> Family { get; } = RegistryManifest.Family(EmbeddingGemma2TestAssets.RepositoryRoot, "embeddinggemma-2");

    private static string Name(Status status) => status.ToString().ToLowerInvariant();

    // The capability tables of the Stable profiles, from the generated records.
    private static IReadOnlyDictionary<string, IReadOnlyDictionary<string, Status>> StableTables { get; } =
        EmbeddingGemma2Capabilities.ProfileCapabilities.Values
            .Where(p => p.Maturity == ProfileMaturity.Stable)
            .ToDictionary(p => p.ProfileId, p => p.Capabilities, StringComparer.Ordinal);

    [Fact]
    public void GeneratedRecordsEqualTheManifestsFieldForField() =>
        CapabilityRecordChecks.AssertRecordsEqualManifests(EmbeddingGemma2Capabilities.ProfileCapabilities, Family);

    [Fact]
    public void ObsoleteMembersEqualTheNewOnes()
    {
#pragma warning disable CS0618 // The former members stay for one minor version and must keep their values.
        Assert.Equal(EmbeddingGemma2Capabilities.Statuses.Keys.Order(StringComparer.Ordinal), EmbeddingGemma2Capabilities.All.Keys.Order(StringComparer.Ordinal));
        foreach (var (key, status) in EmbeddingGemma2Capabilities.Statuses)
        {
            Assert.Equal(status.ToString(), EmbeddingGemma2Capabilities.All[key].ToString());
            Assert.Equal((int)status, (int)EmbeddingGemma2Capabilities.All[key]);
        }

        Assert.Equal(StableTables.Keys.Order(StringComparer.Ordinal), EmbeddingGemma2Capabilities.Profiles.Keys.Order(StringComparer.Ordinal));
        foreach (var (profile, table) in StableTables)
        {
            Assert.Equal(table.Keys.Order(StringComparer.Ordinal), EmbeddingGemma2Capabilities.Profiles[profile].Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in table)
            {
                Assert.Equal(status.ToString(), EmbeddingGemma2Capabilities.Profiles[profile][key].ToString());
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
            [
                EmbeddingGemma2Capabilities.StableImageProfile,
                EmbeddingGemma2Capabilities.StableCudaImageProfile,
                EmbeddingGemma2Capabilities.StableProfile,
                EmbeddingGemma2Capabilities.StableCudaProfile,
                EmbeddingGemma2Capabilities.StableVideoProfile,
                EmbeddingGemma2Capabilities.StableCudaVideoProfile,
            ],
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
                    Assert.True(table[key] == EmbeddingGemma2Capabilities.Statuses[packageKey], $"{profile} {key}: profile {table[key]}, EmbeddingGemma2Capabilities.Statuses[{packageKey}] {EmbeddingGemma2Capabilities.Statuses[packageKey]}");
                }
            }

            var manifest = Family.Single(m => m.Id == profile);
            Assert.Equal("stable", manifest.Maturity);
            Assert.Equal("supported", manifest.Compatibility[profile.EndsWith("-cuda", StringComparison.Ordinal) ? "linux-x64 cuda" : "linux-x64 cpu"]);
            Assert.DoesNotContain(Status.Untested, table.Values);
        }

        var stable = Family.Single(m => m.Id == EmbeddingGemma2Capabilities.StableProfile);
        Assert.Equal(Status.Supported, EmbeddingGemma2Capabilities.Statuses["embedding.pretokenized.cpu"]);
        Assert.Equal(Status.Supported, EmbeddingGemma2Capabilities.Statuses["embedding.text"]);
        Assert.Equal(Status.Supported, EmbeddingGemma2Capabilities.Statuses["image.cpu"]);
        Assert.Equal(Status.Supported, EmbeddingGemma2Capabilities.Statuses["video.cpu"]);
        // Outside the contracts: proposed rows and opt-in deviations (experimental).
        Assert.Equal(Status.Experimental, EmbeddingGemma2Capabilities.Statuses["input.2050_to_8192_tokens"]);
        Assert.Equal(Status.Experimental, EmbeddingGemma2Capabilities.Statuses["prompts.task"]);
        Assert.Equal(Status.Experimental, EmbeddingGemma2Capabilities.Statuses["image.budget_other"]);
        Assert.Equal(Status.Experimental, EmbeddingGemma2Capabilities.Statuses["video.hdr"]);
        Assert.Equal(Status.Supported, EmbeddingGemma2Capabilities.Statuses["image.cuda"]);
        Assert.Equal(Status.Experimental, EmbeddingGemma2Capabilities.Statuses["video.frames_without_timing.upstream_opt_in"]);
        Assert.Equal(Status.Supported, EmbeddingGemma2Capabilities.Statuses["video.cuda"]);
        Assert.Equal(Status.Supported, EmbeddingGemma2Capabilities.Statuses["embedding.pretokenized.cuda"]);
        // ARM64 and x86-64 without AVX2 stay outside the video contracts (swscale SIMD paths are unmeasured there).
        Assert.Equal(Status.Untested, EmbeddingGemma2Capabilities.Statuses["video.decode.arm64_or_non_avx2"]);
        Assert.Equal(stable.Compatibility["windows-x64 cpu"], Name(EmbeddingGemma2Capabilities.Statuses["windows"]));
    }
}
