using Starlings.Acquisition;

namespace Starlings.Models.E5;

/// <summary>Verified local paths of the e5 runtime files.</summary>
/// <param name="ModelPath">Absolute path of <c>onnx/model.onnx</c>; pass it to <see cref="E5Encoder.Create(string, E5EncoderOptions?)"/>.</param>
/// <param name="TokenizerPath">Absolute path of <c>tokenizer.json</c> for the owned tokenizer (profile <c>xlmr-unigram-e5</c>).</param>
/// <param name="Snapshot">The verified snapshot.</param>
public sealed record E5ModelFiles(string ModelPath, string TokenizerPath, ModelSnapshot Snapshot)
{
    /// <summary>Paths of the e5 files in a snapshot verified against <see cref="E5ModelAssets.Lock"/>.</summary>
    public static E5ModelFiles From(ModelSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        return new E5ModelFiles(
            snapshot.GetFilePath(E5ModelAssets.ModelFile),
            snapshot.GetFilePath(E5ModelAssets.TokenizerFile),
            snapshot);
    }

    /// <summary>
    /// SHA-256 of <see cref="ModelPath"/> when <see cref="Snapshot"/> hashed it (<see cref="ModelSnapshot.Verification"/> <see cref="CacheVerification.Sha256"/>); otherwise null. The model factories pass it as <c>OnnxSessionOptions.KnownModelSha256</c>, so the graph is not read a second time at load.
    /// </summary>
    public string? ModelSha256 => Snapshot?.GetVerifiedSha256(E5ModelAssets.ModelFile);
}

/// <summary>
/// The immutable asset lock of the profiles <c>multilingual-e5-small/onnx-fp32-cpu</c> and <c>onnx-fp32-cuda</c>
/// (values from <c>origin.files</c> of their registry manifests), and explicit acquisition of those files. The encoder
/// never downloads; call <see cref="DownloadAsync"/> once (setup step), then <see cref="OpenLocal"/> at startup.
/// </summary>
public static class E5ModelAssets
{
    /// <summary>Hub repository.</summary>
    public const string Repository = "intfloat/multilingual-e5-small";

    /// <summary>Pinned commit.</summary>
    public const string Revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3";

    /// <summary>ONNX graph path inside the repository.</summary>
    public const string ModelFile = "onnx/model.onnx";

    /// <summary>Tokenizer path inside the repository.</summary>
    public const string TokenizerFile = "tokenizer.json";

    /// <summary>SHA-256 of <see cref="ModelFile"/>; usable as <c>OnnxSessionOptions.ExpectedModelSha256</c>.</summary>
    public const string ModelSha256 = "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665";

    /// <summary>Size of <see cref="ModelFile"/> in bytes.</summary>
    public const long ModelSize = 470_268_510;

    /// <summary>SHA-256 of <see cref="TokenizerFile"/>.</summary>
    public const string TokenizerSha256 = "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39";

    /// <summary>Size of <see cref="TokenizerFile"/> in bytes.</summary>
    public const long TokenizerSize = 17_082_730;

    /// <summary>The lock: the two runtime files (about 487 MB). Other repository files are not needed at run time.</summary>
    public static ModelAssetLock Lock { get; } = new(
        "multilingual-e5-small",
        Repository,
        Revision,
        [
            new ModelAssetFile(ModelFile, ModelSize, ModelSha256),
            new ModelAssetFile(TokenizerFile, TokenizerSize, TokenizerSha256),
        ]);

    /// <summary>Explicitly downloads (or verifies already cached) e5 files. See <see cref="ModelAssetStore.DownloadAsync"/>.</summary>
    public static async Task<E5ModelFiles> DownloadAsync(
        ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        using var store = new ModelAssetStore(options);
        return E5ModelFiles.From(await store.DownloadAsync(Lock, progress, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>
    /// Offline: the cached e5 files, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.
    /// </summary>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    public static E5ModelFiles OpenLocal(ModelAcquisitionOptions? options = null)
    {
        using var store = new ModelAssetStore(options);
        return E5ModelFiles.From(store.OpenLocal(Lock));
    }

    /// <summary>Verifies an existing snapshot directory (for example a Hub snapshot) against the lock. Never downloads.</summary>
    public static E5ModelFiles OpenDirectory(string directory, CacheVerification verification = CacheVerification.Sha256) =>
        E5ModelFiles.From(ModelAssetStore.VerifyDirectory(Lock, directory, verification));

    /// <summary>
    /// The e5 lock and acquisition members of this class as an <see cref="IModelAssets{TFiles}"/>, for setup code
    /// that acquires several models the same way. Each member calls the static member of the same name.
    /// </summary>
    public static IModelAssets<E5ModelFiles> Assets { get; } = new AssetsInstance();

    private sealed class AssetsInstance : IModelAssets<E5ModelFiles>
    {
        public ModelAssetLock Lock => E5ModelAssets.Lock;

        public Task<E5ModelFiles> DownloadAsync(ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default) =>
            E5ModelAssets.DownloadAsync(options, progress, cancellationToken);

        public E5ModelFiles OpenLocal(ModelAcquisitionOptions? options = null) => E5ModelAssets.OpenLocal(options);

        public E5ModelFiles OpenDirectory(string directory, CacheVerification verification = CacheVerification.Sha256) =>
            E5ModelAssets.OpenDirectory(directory, verification);
    }
}
