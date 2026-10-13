using Starlings.Acquisition;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>Verified local paths of the EmbeddingGemma 2 runtime files.</summary>
/// <param name="ModelPath">Absolute path of the text-path ONNX graph (<see cref="EmbeddingGemma2ModelAssets.ModelFile"/>).</param>
/// <param name="TokenizerPath">Absolute path of the pinned <c>tokenizer.json</c>.</param>
/// <param name="PromptConfigPath">Absolute path of the pinned <c>config_sentence_transformers.json</c> (the named prompts).</param>
/// <param name="Snapshot">The verified snapshot.</param>
public sealed record EmbeddingGemma2ModelFiles(string ModelPath, string TokenizerPath, string PromptConfigPath, ModelSnapshot Snapshot)
{
    /// <summary>Paths of the files in a snapshot verified against <see cref="EmbeddingGemma2ModelAssets.Lock"/>.</summary>
    public static EmbeddingGemma2ModelFiles From(ModelSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        return new EmbeddingGemma2ModelFiles(
            snapshot.GetFilePath(EmbeddingGemma2ModelAssets.ModelFile),
            snapshot.GetFilePath(EmbeddingGemma2ModelAssets.TokenizerFile),
            snapshot.GetFilePath(EmbeddingGemma2ModelAssets.PromptConfigFile),
            snapshot);
    }
}

/// <summary>
/// The immutable asset lock of the profiles <c>embeddinggemma-2/onnx-fp32-cpu</c> and <c>onnx-fp32-cuda</c>, and
/// explicit acquisition of those files. The encoder never downloads; call <see cref="DownloadAsync"/> once (setup
/// step), then <see cref="OpenLocal"/> or <see cref="OpenDirectory"/> at startup.
/// </summary>
/// <remarks>
/// <para>
/// <c>tokenizer.json</c> and <c>config_sentence_transformers.json</c> come from the Hub repository
/// <c>google/embeddinggemma-2</c> at the pinned commit. The Hub repository has no ONNX graph, so the graph is
/// Starling's export of the text path of the same commit (token embedding, per-layer embeddings, 24 layers, final
/// norm, 512-to-768 projection; torch 2.14.1 dynamo exporter, opset 20;
/// <c>model-pipeline/reference/embeddinggemma2/export_embeddinggemma2_onnx.py</c> then
/// <c>release_embeddinggemma2_onnx.py</c>, which strips the exporter's run-dependent metadata and stores the weights
/// inline so the digest is reproducible). It is pinned by size and SHA-256 and downloaded from an explicit source
/// (<see cref="ModelSource"/>) under the license of the weights (Apache-2.0; the release carries a notice with the
/// model card's usage-policy sentence: deployments must adhere to the Gemma Prohibited Use Policy).
/// </para>
/// <para>
/// TODO(maintainer): <see cref="ModelSource"/> is in a private Hugging Face repository (<see cref="HostingRepository"/>)
/// until the packages leave preview (<see cref="ModelSourceIsPlaceholder"/>). Until then, without a token that can read
/// it, <see cref="DownloadAsync"/> downloads the Hub files and fails on the graph; produce the graph with the pinned
/// scripts (the digest must match <see cref="ModelSha256"/>) and place it in the snapshot directory, or use
/// <see cref="OpenDirectory"/>.
/// </para>
/// </remarks>
public static class EmbeddingGemma2ModelAssets
{
    /// <summary>Hub repository of the checkpoint, the tokenizer and the prompt configuration.</summary>
    public const string Repository = "google/embeddinggemma-2";

    /// <summary>Pinned commit of <see cref="Repository"/>.</summary>
    public const string Revision = "914f7f89142e33e77833254d9c9b90c3cef7303b";

    /// <summary>Path of the ONNX graph in the snapshot directory.</summary>
    public const string ModelFile = "onnx/embeddinggemma2-text-fp32.onnx";

    /// <summary>Tokenizer path inside the repository.</summary>
    public const string TokenizerFile = "tokenizer.json";

    /// <summary>Prompt configuration path inside the repository (the 20 named prompts, <c>default_prompt_name</c> null).</summary>
    public const string PromptConfigFile = "config_sentence_transformers.json";

    /// <summary>SHA-256 of <see cref="ModelFile"/>; usable as <c>OnnxSessionOptions.ExpectedModelSha256</c>.</summary>
    public const string ModelSha256 = "02cc7ecccbfcb29d812eb104e4c6ea6c599bc87e94f01082606f6a11f0252d11";

    /// <summary>Size of <see cref="ModelFile"/> in bytes (weights inline, one file).</summary>
    public const long ModelSize = 1_084_327_317;

    /// <summary>SHA-256 of <see cref="TokenizerFile"/>.</summary>
    public const string TokenizerSha256 = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4";

    /// <summary>Size of <see cref="TokenizerFile"/> in bytes.</summary>
    public const long TokenizerSize = 32_170_510;

    /// <summary>SHA-256 of <see cref="PromptConfigFile"/>.</summary>
    public const string PromptConfigSha256 = "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5";

    /// <summary>Size of <see cref="PromptConfigFile"/> in bytes.</summary>
    public const long PromptConfigSize = 1_565;

    /// <summary>Release name of the graph (tag <c>r1</c> of <see cref="HostingRepository"/>).</summary>
    public const string ReleaseTag = "embeddinggemma-2-onnx-r1";

    /// <summary>Hugging Face repository that hosts the graph (private for now).</summary>
    public const string HostingRepository = "Starlinghq/embeddinggemma-2-onnx";

    /// <summary>Pinned commit of <see cref="HostingRepository"/>.</summary>
    public const string HostingRevision = "92a02ae2887a91180e611776dd9c286dc5fa35da";

    /// <summary>
    /// Download address of <see cref="ModelFile"/>: <c>resolve</c> in <see cref="HostingRepository"/> at
    /// <see cref="HostingRevision"/>. The digest, not the address, pins the bytes.
    /// </summary>
    public const string ModelSource = "https://huggingface.co/" + HostingRepository + "/resolve/" + HostingRevision + "/embeddinggemma2-text-fp32.onnx";

    /// <summary>True while <see cref="HostingRepository"/> is private (TODO(maintainer): make it public, then set false).</summary>
    public const bool ModelSourceIsPlaceholder = true;

    /// <summary>The lock: the graph, the tokenizer and the prompt configuration (about 1.12 GB).</summary>
    public static ModelAssetLock Lock { get; } = new(
        "embeddinggemma-2",
        Repository,
        Revision,
        [
            new ModelAssetFile(ModelFile, ModelSize, ModelSha256, new Uri(ModelSource)),
            new ModelAssetFile(TokenizerFile, TokenizerSize, TokenizerSha256),
            new ModelAssetFile(PromptConfigFile, PromptConfigSize, PromptConfigSha256),
        ]);

    /// <summary>Explicitly downloads (or verifies already cached) EmbeddingGemma 2 files. See <see cref="ModelAssetStore.DownloadAsync"/>.</summary>
    public static async Task<EmbeddingGemma2ModelFiles> DownloadAsync(
        ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        using var store = new ModelAssetStore(options);
        return EmbeddingGemma2ModelFiles.From(await store.DownloadAsync(Lock, progress, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Offline: the cached files, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    public static EmbeddingGemma2ModelFiles OpenLocal(ModelAcquisitionOptions? options = null)
    {
        using var store = new ModelAssetStore(options);
        return EmbeddingGemma2ModelFiles.From(store.OpenLocal(Lock));
    }

    /// <summary>
    /// Verifies an existing directory holding <see cref="TokenizerFile"/>, <see cref="PromptConfigFile"/> and
    /// <see cref="ModelFile"/> against the lock. Never downloads.
    /// </summary>
    public static EmbeddingGemma2ModelFiles OpenDirectory(string directory, CacheVerification verification = CacheVerification.Sha256) =>
        EmbeddingGemma2ModelFiles.From(ModelAssetStore.VerifyDirectory(Lock, directory, verification));

    /// <summary>Path of the vision graph in the snapshot directory (pixels and patch positions to soft tokens).</summary>
    public const string VisionModelFile = "onnx/embeddinggemma2-vision-fp32.onnx";

    /// <summary>
    /// Path of the multimodal text graph in the snapshot directory: the released text graph (<see cref="ModelFile"/>)
    /// with two more inputs (<c>multimodal_embeds</c>, <c>multimodal_mask</c>) that put soft tokens in place of the token
    /// embedding. Used only for inputs with images; text-only inputs keep <see cref="ModelFile"/>.
    /// </summary>
    public const string MultimodalModelFile = "onnx/embeddinggemma2-text-mm-fp32.onnx";

    /// <summary>SHA-256 of <see cref="VisionModelFile"/>.</summary>
    public const string VisionModelSha256 = "6c97f477a82f6863d22d800476de6d0a618b23e514aa9b65a46b220c4b106b92";

    /// <summary>Size of <see cref="VisionModelFile"/> in bytes (weights inline, one file).</summary>
    public const long VisionModelSize = 671_256_072;

    /// <summary>SHA-256 of <see cref="MultimodalModelFile"/>.</summary>
    public const string MultimodalModelSha256 = "a2c22e072ec230b105b35c2787dcfccba186254e6a921e729c45b40e50dbbb74";

    /// <summary>Size of <see cref="MultimodalModelFile"/> in bytes (weights inline, one file).</summary>
    public const long MultimodalModelSize = 1_084_327_898;

    /// <summary>Release name of the vision graph and the multimodal text graph (tag <c>r1</c> of <see cref="ImageHostingRepository"/>).</summary>
    public const string ImageReleaseTag = "embeddinggemma-2-vision-onnx-r1";

    /// <summary>Hugging Face repository that hosts the vision graph and the multimodal text graph (private for now).</summary>
    public const string ImageHostingRepository = "Starlinghq/embeddinggemma-2-vision-onnx";

    /// <summary>Pinned commit of <see cref="ImageHostingRepository"/>.</summary>
    public const string ImageHostingRevision = "1a22d40bf395c7f0a42388df72c883b661a7b3d6";

    /// <summary>Download address of <see cref="VisionModelFile"/> (the digest, not the address, pins the bytes).</summary>
    public const string VisionModelSource = "https://huggingface.co/" + ImageHostingRepository + "/resolve/" + ImageHostingRevision + "/embeddinggemma2-vision-fp32.onnx";

    /// <summary>Download address of <see cref="MultimodalModelFile"/> (the digest, not the address, pins the bytes).</summary>
    public const string MultimodalModelSource = "https://huggingface.co/" + ImageHostingRepository + "/resolve/" + ImageHostingRevision + "/embeddinggemma2-text-mm-fp32.onnx";

    /// <summary>
    /// True while <see cref="ImageHostingRepository"/> is private (TODO(maintainer): make it public, then set false).
    /// </summary>
    public const bool ImageModelSourceIsPlaceholder = true;

    /// <summary>
    /// The image lock: every file of <see cref="Lock"/> plus the vision graph and the multimodal text graph (about
    /// 2.9 GB). Same snapshot directory as <see cref="Lock"/>, so a text-only download is reused.
    /// </summary>
    public static ModelAssetLock ImageLock { get; } = new(
        "embeddinggemma-2-image",
        Repository,
        Revision,
        [
            .. Lock.Files,
            new ModelAssetFile(VisionModelFile, VisionModelSize, VisionModelSha256, new Uri(VisionModelSource)),
            new ModelAssetFile(MultimodalModelFile, MultimodalModelSize, MultimodalModelSha256, new Uri(MultimodalModelSource)),
        ]);

    /// <summary>Explicitly downloads (or verifies already cached) the files of <see cref="ImageLock"/>.</summary>
    public static async Task<EmbeddingGemma2ImageModelFiles> DownloadImageAssetsAsync(
        ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        using var store = new ModelAssetStore(options);
        return EmbeddingGemma2ImageModelFiles.From(await store.DownloadAsync(ImageLock, progress, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Offline: the cached files of <see cref="ImageLock"/>, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    public static EmbeddingGemma2ImageModelFiles OpenLocalImageAssets(ModelAcquisitionOptions? options = null)
    {
        using var store = new ModelAssetStore(options);
        return EmbeddingGemma2ImageModelFiles.From(store.OpenLocal(ImageLock));
    }

    /// <summary>Verifies an existing directory against <see cref="ImageLock"/>. Never downloads.</summary>
    public static EmbeddingGemma2ImageModelFiles OpenImageDirectory(string directory, CacheVerification verification = CacheVerification.Sha256) =>
        EmbeddingGemma2ImageModelFiles.From(ModelAssetStore.VerifyDirectory(ImageLock, directory, verification));
}

/// <summary>Verified local paths of the files of <see cref="EmbeddingGemma2ModelAssets.ImageLock"/>.</summary>
/// <param name="Text">The text files (released text graph, tokenizer, prompt configuration).</param>
/// <param name="VisionModelPath">Absolute path of the vision graph.</param>
/// <param name="MultimodalModelPath">Absolute path of the multimodal text graph.</param>
public sealed record EmbeddingGemma2ImageModelFiles(EmbeddingGemma2ModelFiles Text, string VisionModelPath, string MultimodalModelPath)
{
    /// <summary>Paths of the files in a snapshot verified against <see cref="EmbeddingGemma2ModelAssets.ImageLock"/>.</summary>
    public static EmbeddingGemma2ImageModelFiles From(ModelSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        return new EmbeddingGemma2ImageModelFiles(
            EmbeddingGemma2ModelFiles.From(snapshot),
            snapshot.GetFilePath(EmbeddingGemma2ModelAssets.VisionModelFile),
            snapshot.GetFilePath(EmbeddingGemma2ModelAssets.MultimodalModelFile));
    }
}
