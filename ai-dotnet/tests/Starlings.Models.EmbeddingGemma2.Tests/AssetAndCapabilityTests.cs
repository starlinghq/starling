using Starlings.Acquisition;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>The asset lock against the registry manifests, the notices and the capability descriptor. Runs in CI.</summary>
public class AssetAndCapabilityTests
{
    private const string PolicySentence = "Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).";

    private static string ManifestPath(string profile) =>
        Path.Combine(EmbeddingGemma2TestAssets.RepositoryRoot, "model-pipeline", "registry", "embeddinggemma-2", profile, "manifest.yaml");

    private static string Manifest(string profile) => File.ReadAllText(ManifestPath(profile)).Replace("\r\n", "\n", StringComparison.Ordinal);

    [Theory]
    [InlineData("onnx-fp32-cpu")]
    [InlineData("onnx-fp32-cuda")]
    public void LockPinsTheManifestFilesAndCommit(string profile)
    {
        var assetLock = EmbeddingGemma2ModelAssets.Lock;
        string manifest = Manifest(profile);
        Assert.Equal("google/embeddinggemma-2", assetLock.Repository);
        Assert.Equal(40, assetLock.Revision.Length);
        Assert.Contains($"revision: \"{assetLock.Revision}\"", manifest, StringComparison.Ordinal);
        foreach (var file in assetLock.Files)
        {
            Assert.Contains($"- path: {file.Path}\n      sha256: \"{file.Sha256}\"\n      size: {file.Size}", manifest, StringComparison.Ordinal);
        }

        Assert.Contains($"path: {EmbeddingGemma2ModelAssets.ModelFile}\n      sha256: \"{EmbeddingGemma2ModelAssets.ModelSha256}\"\n      size: {EmbeddingGemma2ModelAssets.ModelSize}\n      # Public Hugging Face repository", manifest, StringComparison.Ordinal);
        Assert.Contains($"source: {EmbeddingGemma2ModelAssets.ModelSource}\n      source_status: public", manifest, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("image-fp32-cpu")]
    [InlineData("image-fp32-cuda")]
    public void ImageLockPinsTheImageManifestFiles(string profile)
    {
        var assetLock = EmbeddingGemma2ModelAssets.ImageLock;
        string manifest = Manifest(profile);
        Assert.Equal(EmbeddingGemma2ModelAssets.Lock.Revision, assetLock.Revision);
        Assert.Equal(EmbeddingGemma2ModelAssets.Lock.Files.Count + 2, assetLock.Files.Count);
        foreach (var file in assetLock.Files)
        {
            Assert.Contains($"- path: {file.Path}\n      sha256: \"{file.Sha256}\"\n      size: {file.Size}", manifest, StringComparison.Ordinal);
        }

        foreach (string path in new[] { EmbeddingGemma2ModelAssets.VisionModelFile, EmbeddingGemma2ModelAssets.MultimodalModelFile })
        {
            var file = assetLock.GetFile(path);
            Assert.Equal(Uri.UriSchemeHttps, file.Source!.Scheme);
            Assert.Contains($"/{EmbeddingGemma2ModelAssets.ImageHostingRepository}/resolve/{EmbeddingGemma2ModelAssets.ImageHostingRevision}/", file.Source.AbsoluteUri, StringComparison.Ordinal);
            Assert.Contains($"source: {file.Source.AbsoluteUri}\n        source_status: public", manifest, StringComparison.Ordinal);
        }

        Assert.False(EmbeddingGemma2ModelAssets.ImageModelSourceIsPlaceholder);
        Assert.Equal("embeddinggemma-2-vision-onnx-r1", EmbeddingGemma2ModelAssets.ImageReleaseTag);
        Assert.Equal(assetLock.Files, ModelAssetLock.Parse(assetLock.ToJson()).Files);
    }

    [Fact]
    public void TheImageLockMatchesTheImageFixtureEnvironment()
    {
        // The image fixtures were computed from the same tokenizer and checkpoint.
        string env = File.ReadAllText(Path.Combine(EmbeddingGemma2TestAssets.ProtectedRoot, "image-fp32-cpu", "image-environment.json"));
        Assert.Contains(EmbeddingGemma2ModelAssets.TokenizerSha256, env, StringComparison.Ordinal);
        Assert.Contains(EmbeddingGemma2ModelAssets.Revision, env, StringComparison.Ordinal);
    }

    [Fact]
    public void TheLockMatchesTheProtectedFixtureEnvironment()
    {
        // The fixtures were computed from these exact tokenizer and prompt files (environment.json).
        string env = File.ReadAllText(Path.Combine(EmbeddingGemma2TestAssets.FixtureDirectory, "environment.json"));
        Assert.Contains(EmbeddingGemma2ModelAssets.TokenizerSha256, env, StringComparison.Ordinal);
        Assert.Contains(EmbeddingGemma2ModelAssets.PromptConfigSha256, env, StringComparison.Ordinal);
        Assert.Contains(EmbeddingGemma2ModelAssets.Revision, env, StringComparison.Ordinal);
    }

    [Fact]
    public void TheGraphHasAnExplicitHttpsHostedSourceAndTheOtherFilesComeFromTheHub()
    {
        var graph = EmbeddingGemma2ModelAssets.Lock.GetFile(EmbeddingGemma2ModelAssets.ModelFile);
        Assert.Equal(Uri.UriSchemeHttps, graph.Source!.Scheme);
        Assert.Equal("huggingface.co", graph.Source.Host);
        Assert.Equal(EmbeddingGemma2ModelAssets.ModelSource, graph.Source.AbsoluteUri);
        Assert.Contains($"/{EmbeddingGemma2ModelAssets.HostingRepository}/resolve/{EmbeddingGemma2ModelAssets.HostingRevision}/", EmbeddingGemma2ModelAssets.ModelSource, StringComparison.Ordinal);
        Assert.False(EmbeddingGemma2ModelAssets.ModelSourceIsPlaceholder);
        Assert.Null(EmbeddingGemma2ModelAssets.Lock.GetFile(EmbeddingGemma2ModelAssets.TokenizerFile).Source);
        Assert.Null(EmbeddingGemma2ModelAssets.Lock.GetFile(EmbeddingGemma2ModelAssets.PromptConfigFile).Source);
        string json = EmbeddingGemma2ModelAssets.Lock.ToJson();
        Assert.Equal(EmbeddingGemma2ModelAssets.Lock.Files, ModelAssetLock.Parse(json).Files);
    }

    [Fact]
    public void OpenDirectoryNeverDownloadsAndNamesTheMissingFiles()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"df-eg2-empty-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var ex = Assert.Throws<ModelAssetMissingException>(() => EmbeddingGemma2ModelAssets.OpenDirectory(dir));
            Assert.Equal([EmbeddingGemma2ModelAssets.ModelFile, EmbeddingGemma2ModelAssets.TokenizerFile, EmbeddingGemma2ModelAssets.PromptConfigFile], ex.MissingPaths);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void TheReadmeAndTheReleaseNoticeRepeatTheUsagePolicyVerbatim()
    {
        string root = EmbeddingGemma2TestAssets.RepositoryRoot;
        string readme = File.ReadAllText(Path.Combine(root, "ai-dotnet", "src", "Starlings.Models.EmbeddingGemma2", "README.md"));
        string notice = File.ReadAllText(Path.Combine(root, "model-pipeline", "reference", "embeddinggemma2", "NOTICE-embeddinggemma2-onnx.txt"));
        Assert.Contains(PolicySentence, readme, StringComparison.Ordinal);
        Assert.Contains(PolicySentence, notice, StringComparison.Ordinal);
        Assert.Contains("Apache License", notice, StringComparison.Ordinal);
        Assert.Contains(EmbeddingGemma2ModelAssets.Revision, notice, StringComparison.Ordinal);
        Assert.Contains("Images", readme, StringComparison.Ordinal);
        string visionNotice = File.ReadAllText(Path.Combine(root, "model-pipeline", "reference", "embeddinggemma2", "NOTICE-embeddinggemma2-vision-onnx.txt"));
        Assert.Contains(PolicySentence, visionNotice, StringComparison.Ordinal);
        Assert.Contains("Apache License", visionNotice, StringComparison.Ordinal);
        Assert.Contains(EmbeddingGemma2ModelAssets.Revision, visionNotice, StringComparison.Ordinal);
        Assert.Contains(EmbeddingGemma2ModelAssets.ImageReleaseTag, visionNotice, StringComparison.Ordinal);
        Assert.Contains(EmbeddingGemma2ModelAssets.ModelSha256, visionNotice, StringComparison.Ordinal);
    }

    [Fact]
    public void CapabilitiesAreExplicitAndMatchTheContract()
    {
        var all = EmbeddingGemma2Capabilities.All;
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, all["embedding.pretokenized.cpu"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, all["embedding.pretokenized.cuda"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, all["embedding.text"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, all["tokenizer.owned"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, all["input.2050_to_8192_tokens"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, all["prompts.task"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Unsupported, all["image.decode.gif_transparency"]);
        foreach (var key in new[] { "audio", "video.frames_without_timing", "video.timestamps", "video.over_8192_tokens", "video.decode.ffmpeg9", "mlnet.video_column", "media_placeholder_tokens", "image.16_bit_per_channel", "image.over_8192_tokens", "mlnet.image_column", "input.over_8192_tokens", "truncation", "tf32", "fp16", "matryoshka.other_sizes", "prompt_insertion.implicit" })
        {
            Assert.Equal(EmbeddingGemma2CapabilityStatus.Unsupported, all[key]);
        }

        Assert.All(all.Values, v => Assert.True(Enum.IsDefined(v)));

        string cpu = Manifest("onnx-fp32-cpu");
        Assert.Contains("    version: 0.3.2\n    profile_id: gemma-bpe-embeddinggemma2\n", cpu, StringComparison.Ordinal);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, all["image"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, all["image.cuda"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, all["image.budget_other"]);
        string image = Manifest("image-fp32-cpu");
        Assert.Contains("maturity: stable", image, StringComparison.Ordinal);
        Assert.Contains("  image: supported", image, StringComparison.Ordinal);
        Assert.Contains("maturity: stable", Manifest("image-fp32-cuda"), StringComparison.Ordinal); // RTX 3090 contract
        Assert.Contains("provider_options:\n    use_tf32: 0", Manifest("image-fp32-cuda"), StringComparison.Ordinal);
        foreach (var key in new[] { "video", "video.cpu", "video.cuda", "video.decoded_frames_with_timing", "video.interleaved_text", "video.interleaved_image", "video.decode.ffmpeg8", "video.decode.ffmpeg9_adapter", "extensions_ai.video_embedding_generator" })
        {
            Assert.Equal(EmbeddingGemma2CapabilityStatus.Supported, all[key]);
        }

        foreach (var key in new[] { "video.pre_sampled_frames", "video.frames_without_timing.upstream_opt_in", "video.variable_frame_rate", "video.interlaced", "video.hdr", "video.decode.rotation_other", "extensions_ai.video_embedding_generator.cuda" })
        {
            Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, all[key]);
        }

        string video = Manifest("video-fp32-cpu");
        Assert.Contains("maturity: stable", video, StringComparison.Ordinal);
        Assert.Contains("  video: supported", video, StringComparison.Ordinal);
        Assert.Contains("maturity: stable", Manifest("video-fp32-cuda"), StringComparison.Ordinal); // RTX 3090 contract
        Assert.Contains("  video_frames_without_timing: unsupported", video, StringComparison.Ordinal);
        Assert.Contains("provider_options:\n    use_tf32: 0", Manifest("video-fp32-cuda"), StringComparison.Ordinal);
        Assert.Contains("  video: unsupported", Manifest("image-fp32-cpu"), StringComparison.Ordinal); // the image profile; video is the video-fp32-* profiles
        Assert.Contains("max_input_tokens: 8192", cpu, StringComparison.Ordinal);
        Assert.Contains("truncation: error", cpu, StringComparison.Ordinal);
        Assert.Contains("provider_options:\n    use_tf32: 0", Manifest("onnx-fp32-cuda"), StringComparison.Ordinal);
    }

    [Fact]
    public void ProfileDefaults()
    {
        var options = new EmbeddingGemma2EncoderOptions();
        Assert.Equal(768, options.Dimension);
        Assert.Equal(8192, options.MaxSequenceLength);
        Assert.Equal(16, options.MaxBatchSize);
        Assert.Equal(8192, options.MaxBatchTokens);
        Assert.Equal(0L, EmbeddingGemma2Encoder.PadTokenId);
    }
}
