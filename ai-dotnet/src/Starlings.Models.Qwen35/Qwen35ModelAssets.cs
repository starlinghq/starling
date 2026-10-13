using Starlings.Acquisition;

namespace Starlings.Models.Qwen35;

/// <summary>
/// A built ONNX Runtime GenAI variant of the checkpoint. Both are built from a copy of the snapshot whose
/// <c>config.json</c> declares float32 ("x"), with the onnxruntime-genai 0.17.1 builder and <c>exclude_mtp=true</c>
/// (<c>model-pipeline/reference/generation/build_ortgenai_bundle.py</c>).
/// </summary>
public enum Qwen35Variant
{
    /// <summary>fp32 graph built for the CPU provider (<c>-p fp32 -e cpu</c>); runs on CPU and, with TF32 off, on CUDA.</summary>
    Fp32x,

    /// <summary>fp16 graph built for the CUDA provider (<c>-p fp16 -e cuda</c>); CUDA only.</summary>
    Fp16x,
}

/// <summary>Verified local paths of one variant's runtime files.</summary>
/// <param name="Variant">The variant.</param>
/// <param name="ModelDirectory">Directory with <c>genai_config.json</c>, <c>model.onnx</c> and <c>model.onnx.data</c>.</param>
/// <param name="TokenizerPath"><c>tokenizer.json</c> for the owned tokenizer (profile <c>qwen35-bpe-chat</c>).</param>
/// <param name="ChatTemplatePath">The pinned <c>chat_template.jinja</c> (checked against <see cref="Qwen35ChatTemplate.TemplateSha256"/> at load).</param>
/// <param name="Snapshot">The verified snapshot.</param>
public sealed record Qwen35ModelFiles(Qwen35Variant Variant, string ModelDirectory, string TokenizerPath, string ChatTemplatePath, ModelSnapshot Snapshot)
{
    /// <summary>Paths of a variant's files in a snapshot verified against <see cref="Qwen35ModelAssets.GetLock"/>.</summary>
    public static Qwen35ModelFiles From(Qwen35Variant variant, ModelSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        string config = snapshot.GetFilePath(Qwen35ModelAssets.GraphFile(variant, "genai_config.json"));
        return new Qwen35ModelFiles(
            variant,
            Path.GetDirectoryName(config)!,
            snapshot.GetFilePath(Qwen35ModelAssets.TokenizerFile),
            snapshot.GetFilePath(Qwen35ModelAssets.ChatTemplateFile),
            snapshot);
    }
}

/// <summary>
/// The immutable asset locks of the profiles <c>qwen3.5-0.8b/ortgenai-fp32x</c> and <c>qwen3.5-0.8b/ortgenai-fp16x</c>,
/// and explicit acquisition of those files. The model never downloads; call <see cref="DownloadAsync"/> once (setup
/// step), then <see cref="OpenLocal"/> or <see cref="OpenDirectory"/> at startup.
/// </summary>
/// <remarks>
/// <para>
/// The Hub repository <c>Qwen/Qwen3.5-0.8B</c> has no ONNX Runtime GenAI build, and its <c>tokenizer.json</c> is not the
/// pipeline transformers runs, so the graph files and <c>transformers-5.18.0/tokenizer.json</c> (the builder's copy of the
/// transformers 5.18.0 runtime tokenizer, equal to the <c>qwen35-bpe-chat</c> profile asset) are Starling's
/// reproducible build of the pinned commit (<c>model-pipeline/reference/generation/build_ortgenai_bundle.py</c>), under
/// the license of the weights (Apache-2.0, attribution in the package README).
/// </para>
/// <para>
/// Every file, including the upstream <c>chat_template.jinja</c>, is downloaded from one place: the Hugging Face
/// repository <see cref="HostingRepository"/> at commit <see cref="HostingRevision"/>, whose layout equals the snapshot
/// layout (<see cref="HostingBaseUrl"/> + path). The digests in the lock, not the address, pin the bytes. The hosting
/// repository is private for now (<see cref="HostingIsPrivate"/>; it moves to an organization later): downloading needs a
/// Hugging Face token with read access to it, through <see cref="ModelAcquisitionOptions.HuggingFaceToken"/> or
/// <c>$HF_TOKEN</c>. The token is sent only to <c>https://huggingface.co</c>, never to the storage host the Hub redirects
/// to. Without access <see cref="DownloadAsync"/> fails with <see cref="ModelAssetAccessDeniedException"/>; the files can
/// also be built with the bundle script into the snapshot directory (the digests must match), or opened with
/// <see cref="OpenDirectory"/>.
/// </para>
/// </remarks>
public static class Qwen35ModelAssets
{
    /// <summary>Hub repository of the checkpoint.</summary>
    public const string Repository = "Qwen/Qwen3.5-0.8B";

    /// <summary>Pinned commit of <see cref="Repository"/>.</summary>
    public const string Revision = "2fc06364715b967f1860aea9cf38778875588b17";

    /// <summary>onnxruntime-genai builder version that wrote the graphs (must equal the runtime version).</summary>
    public const string BuilderVersion = "0.17.1";

    /// <summary>onnxruntime-genai source commit of the builder (<c>rel-0.17.1</c> head; the repository has no v0.17.1 tag).</summary>
    public const string BuilderSourceCommit = "83de55a1f3886cddbb32a5517e26adb3f9d65159";

    /// <summary>Path of the owned tokenizer's <c>tokenizer.json</c> in the snapshot directory.</summary>
    public const string TokenizerFile = "transformers-5.18.0/tokenizer.json";

    /// <summary>SHA-256 of <see cref="TokenizerFile"/> (the <c>qwen35-bpe-chat</c> profile asset).</summary>
    public const string TokenizerSha256 = "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f";

    /// <summary>Owned tokenizer profile.</summary>
    public const string TokenizerProfile = "qwen35-bpe-chat";

    /// <summary>Path of the chat template in the snapshot directory (a Hub file).</summary>
    public const string ChatTemplateFile = "chat_template.jinja";

    /// <summary>Hugging Face repository that hosts every file of both locks, in the snapshot layout (private for now).</summary>
    public const string HostingRepository = "Starlinghq/qwen3.5-0.8b-ortgenai";

    /// <summary>Pinned commit of <see cref="HostingRepository"/>.</summary>
    public const string HostingRevision = "95a690c1774f144957cb7b76986a9126e8b1bb5f";

    /// <summary>Base address of the files: <c>resolve</c> at <see cref="HostingRevision"/>; a file's source is this plus its snapshot path.</summary>
    public const string HostingBaseUrl = "https://huggingface.co/" + HostingRepository + "/resolve/" + HostingRevision + "/";

    /// <summary>
    /// True while <see cref="HostingRepository"/> is private: <see cref="DownloadAsync"/> then needs a Hugging Face token with
    /// read access to it (<see cref="ModelAcquisitionOptions.HuggingFaceToken"/> or <c>$HF_TOKEN</c>).
    /// </summary>
    public const bool HostingIsPrivate = true;

    private const long TokenizerSize = 19_989_339;
    private const long ChatTemplateSize = 7_755;

    private static readonly ModelAssetLock Fp32xLock = Build(Qwen35Variant.Fp32x,
        ("genai_config.json", 1_935, "e7e1957b2544fdad6170a9cc81cbbb1b88c8819b7016a609170ad7cbebba81ba"),
        ("model.onnx", 308_301, "4a052c4a1444c9437e00ffd48baecb0838d3d9ca1703c3153dbd46e945e84703"),
        ("model.onnx.data", 3_077_177_344, "129c58a320b6de7192713cc9c7e4ba744dcf7fabb0f57c6535021b45b3dc5143"));

    private static readonly ModelAssetLock Fp16xLock = Build(Qwen35Variant.Fp16x,
        ("genai_config.json", 2_109, "ded164b67284883a13c6454d7445f814014a1e12277ce323011683cd6a69d055"),
        ("model.onnx", 316_521, "26895023e4be1f528aef04ce3a07a14e70a2421cfb400a49fec639ea9a14f46d"),
        ("model.onnx.data", 1_572_143_104, "47f6ee15d95080acc66f7c4acdf4012c83b20063cf0cfeb59a6b7ae09950a3b2"));

    /// <summary>Directory of a variant's graph files in the snapshot directory (<c>ortgenai-fp32x</c>, <c>ortgenai-fp16x</c>).</summary>
    public static string VariantDirectory(Qwen35Variant variant) => variant switch
    {
        Qwen35Variant.Fp32x => "ortgenai-fp32x",
        Qwen35Variant.Fp16x => "ortgenai-fp16x",
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown variant."),
    };

    /// <summary>The lock of one variant: its three graph files, the tokenizer and the chat template.</summary>
    public static ModelAssetLock GetLock(Qwen35Variant variant) => variant switch
    {
        Qwen35Variant.Fp32x => Fp32xLock,
        Qwen35Variant.Fp16x => Fp16xLock,
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown variant."),
    };

    /// <summary>Explicitly downloads (or verifies already cached) a variant's files. See <see cref="ModelAssetStore.DownloadAsync"/>.</summary>
    public static async Task<Qwen35ModelFiles> DownloadAsync(
        Qwen35Variant variant, ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        using var store = new ModelAssetStore(options);
        return Qwen35ModelFiles.From(variant, await store.DownloadAsync(GetLock(variant), progress, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Offline: a variant's cached files, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    public static Qwen35ModelFiles OpenLocal(Qwen35Variant variant, ModelAcquisitionOptions? options = null)
    {
        using var store = new ModelAssetStore(options);
        return Qwen35ModelFiles.From(variant, store.OpenLocal(GetLock(variant)));
    }

    /// <summary>
    /// Verifies an existing directory in the snapshot layout (<c>chat_template.jinja</c>, <see cref="TokenizerFile"/>,
    /// <c>ortgenai-&lt;variant&gt;/</c>) against the variant's lock. Never downloads.
    /// </summary>
    public static Qwen35ModelFiles OpenDirectory(string directory, Qwen35Variant variant, CacheVerification verification = CacheVerification.Sha256) =>
        Qwen35ModelFiles.From(variant, ModelAssetStore.VerifyDirectory(GetLock(variant), directory, verification));

    private static Uri HostedSource(string path) => new(HostingBaseUrl + path);

    internal static string GraphFile(Qwen35Variant variant, string name) => $"{VariantDirectory(variant)}/{name}";

    private static ModelAssetLock Build(Qwen35Variant variant, params (string Name, long Size, string Sha256)[] graph)
    {
        string dir = VariantDirectory(variant);
        var files = new List<ModelAssetFile>
        {
            new(ChatTemplateFile, ChatTemplateSize, Qwen35ChatTemplate.TemplateSha256, HostedSource(ChatTemplateFile)),
            new(TokenizerFile, TokenizerSize, TokenizerSha256, HostedSource(TokenizerFile)),
        };
        foreach (var (name, size, sha) in graph)
        {
            files.Add(new ModelAssetFile($"{dir}/{name}", size, sha, HostedSource($"{dir}/{name}")));
        }

        return new ModelAssetLock($"qwen3.5-0.8b-{dir}", Repository, Revision, files);
    }
}
