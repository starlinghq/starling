using Starlings.Acquisition;

namespace Starlings.Models.Gliner2;

/// <summary>Verified local paths of the GLiNER2 runtime files.</summary>
/// <param name="ModelPath">Absolute path of the ONNX graph (<see cref="Gliner2ModelAssets.ModelFile"/>).</param>
/// <param name="TokenizerPath">Absolute path of <c>tokenizer.json</c> for the owned tokenizer (profile <c>gliner2-base-v1-unigram</c>).</param>
/// <param name="Snapshot">The verified snapshot.</param>
public sealed record Gliner2ModelFiles(string ModelPath, string TokenizerPath, ModelSnapshot Snapshot)
{
    /// <summary>Paths of the GLiNER2 files in a snapshot verified against <see cref="Gliner2ModelAssets.Lock"/>.</summary>
    public static Gliner2ModelFiles From(ModelSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        return new Gliner2ModelFiles(
            snapshot.GetFilePath(Gliner2ModelAssets.ModelFile),
            snapshot.GetFilePath(Gliner2ModelAssets.TokenizerFile),
            snapshot);
    }

    /// <summary>
    /// SHA-256 of <see cref="ModelPath"/> when <see cref="Snapshot"/> hashed it (<see cref="ModelSnapshot.Verification"/> <see cref="CacheVerification.Sha256"/>); otherwise null. The model factories pass it as <c>OnnxSessionOptions.KnownModelSha256</c>, so the graph is not read a second time at load.
    /// </summary>
    public string? ModelSha256 => Snapshot.GetVerifiedSha256(Gliner2ModelAssets.ModelFile);
}

/// <summary>
/// The immutable asset lock of the profile <c>gliner2-base-v1/onnx-fp32-cpu</c> (and its CUDA run), and explicit
/// acquisition of those files. The extractor never downloads; call <see cref="DownloadAsync"/> once (setup step), then
/// <see cref="OpenLocal"/> or <see cref="OpenDirectory"/> at startup.
/// </summary>
/// <remarks>
/// <para>
/// <c>tokenizer.json</c> comes from the Hub repository <c>fastino/gliner2-base-v1</c> at the pinned commit. The Hub
/// repository has no ONNX graph, so the graph is Starling's conversion of the same commit (route A of the P-014
/// spike: one graph from token ids and routing positions to count and span logits; torch 2.14.1 dynamo exporter,
/// opset 18; <c>model-pipeline/reference/gliner2/export_gliner2_onnx.py</c> then <c>release_gliner2_onnx.py</c>, which
/// strips the exporter's run-dependent metadata so the digest is reproducible). It is pinned by size and SHA-256 and
/// downloaded from an explicit source (<see cref="ModelSource"/>), under the license of the weights (Apache-2.0,
/// attribution in the package README).
/// </para>
/// <para>
/// <see cref="ModelSource"/> is in the public Hugging Face repository <see cref="HostingRepository"/>, so
/// <see cref="DownloadAsync"/> needs no token. The graph can also be produced with the pinned scripts (the digest must
/// match <see cref="ModelSha256"/>) and placed in the snapshot directory, or opened with <see cref="OpenDirectory"/>.
/// </para>
/// </remarks>
public static class Gliner2ModelAssets
{
    /// <summary>Hub repository of the checkpoint and the tokenizer.</summary>
    public const string Repository = "fastino/gliner2-base-v1";

    /// <summary>Pinned commit of <see cref="Repository"/>.</summary>
    public const string Revision = "f9634218e53580c56edf0de97ca1a7d3f1c2354e";

    /// <summary>GLiNER2 source commit the graph and the C# port follow (version 2.0.0).</summary>
    public const string SourceRevision = "55656fbfa01d3d4a77485e1a1eeeaf682990ccdf";

    /// <summary>Path of the ONNX graph in the snapshot directory.</summary>
    public const string ModelFile = "onnx/gliner2-entities-fp32.onnx";

    /// <summary>Tokenizer path inside the repository.</summary>
    public const string TokenizerFile = "tokenizer.json";

    /// <summary>SHA-256 of <see cref="ModelFile"/>; usable as <c>OnnxSessionOptions.ExpectedModelSha256</c>.</summary>
    public const string ModelSha256 = "0ba2db95b18116906ae4ec21aa62ef493e69e0780dcfdfa98c4456ecf4123265";

    /// <summary>Size of <see cref="ModelFile"/> in bytes (weights inline, one file).</summary>
    public const long ModelSize = 829_284_599;

    /// <summary>SHA-256 of <see cref="TokenizerFile"/>.</summary>
    public const string TokenizerSha256 = "1b7fbabfb4c690bed84c6793bfecae9b8dfe205751b04f9ffd1e76a1e7df9c16";

    /// <summary>Size of <see cref="TokenizerFile"/> in bytes.</summary>
    public const long TokenizerSize = 8_650_701;

    /// <summary>Hugging Face repository that hosts the graph (release <c>gliner2-base-v1-onnx-r1</c>, tag <c>r1</c>; public).</summary>
    public const string HostingRepository = "Starlinghq/gliner2-base-v1-onnx";

    /// <summary>Pinned commit of <see cref="HostingRepository"/>.</summary>
    public const string HostingRevision = "c02c3350a0dcaf0bd6d2878043cc6a4ba950dccb";

    /// <summary>
    /// Download address of <see cref="ModelFile"/>: <c>resolve</c> in <see cref="HostingRepository"/> at
    /// <see cref="HostingRevision"/>. The digest, not the address, pins the bytes.
    /// </summary>
    public const string ModelSource = "https://huggingface.co/" + HostingRepository + "/resolve/" + HostingRevision + "/gliner2-entities-fp32.onnx";

    /// <summary>True while <see cref="HostingRepository"/> is private. False: the repository is public (2026-10-14).</summary>
    public const bool ModelSourceIsPlaceholder = false;

    /// <summary>The lock: the two runtime files (about 838 MB).</summary>
    public static ModelAssetLock Lock { get; } = new(
        "gliner2-base-v1",
        Repository,
        Revision,
        [
            new ModelAssetFile(ModelFile, ModelSize, ModelSha256, new Uri(ModelSource)),
            new ModelAssetFile(TokenizerFile, TokenizerSize, TokenizerSha256),
        ]);

    /// <summary>Explicitly downloads (or verifies already cached) GLiNER2 files. See <see cref="ModelAssetStore.DownloadAsync"/>.</summary>
    public static async Task<Gliner2ModelFiles> DownloadAsync(
        ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        using var store = new ModelAssetStore(options);
        return Gliner2ModelFiles.From(await store.DownloadAsync(Lock, progress, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Offline: the cached GLiNER2 files, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    public static Gliner2ModelFiles OpenLocal(ModelAcquisitionOptions? options = null)
    {
        using var store = new ModelAssetStore(options);
        return Gliner2ModelFiles.From(store.OpenLocal(Lock));
    }

    /// <summary>
    /// Verifies an existing directory holding <c>tokenizer.json</c> and <see cref="ModelFile"/> against the lock.
    /// Never downloads.
    /// </summary>
    public static Gliner2ModelFiles OpenDirectory(string directory, CacheVerification verification = CacheVerification.Sha256) =>
        Gliner2ModelFiles.From(ModelAssetStore.VerifyDirectory(Lock, directory, verification));

    /// <summary>
    /// The GLiNER2 lock and acquisition members of this class as an <see cref="IModelAssets{TFiles}"/>, for setup code
    /// that acquires several models the same way. Each member calls the static member of the same name.
    /// </summary>
    public static IModelAssets<Gliner2ModelFiles> Assets { get; } = new AssetsInstance();

    private sealed class AssetsInstance : IModelAssets<Gliner2ModelFiles>
    {
        public ModelAssetLock Lock => Gliner2ModelAssets.Lock;

        public Task<Gliner2ModelFiles> DownloadAsync(ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default) =>
            Gliner2ModelAssets.DownloadAsync(options, progress, cancellationToken);

        public Gliner2ModelFiles OpenLocal(ModelAcquisitionOptions? options = null) => Gliner2ModelAssets.OpenLocal(options);

        public Gliner2ModelFiles OpenDirectory(string directory, CacheVerification verification = CacheVerification.Sha256) =>
            Gliner2ModelAssets.OpenDirectory(directory, verification);
    }
}
