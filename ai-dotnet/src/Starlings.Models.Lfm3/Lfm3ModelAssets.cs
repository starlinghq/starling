using Starlings.Acquisition;

namespace Starlings.Models.Lfm3;

/// <summary>Verified local paths of one LFM3-1.7B ONNX graph and the tokenizer.</summary>
/// <param name="ModelPath">Absolute path of the graph (<c>onnx/model.onnx</c> or <c>onnx/model_fp16.onnx</c>); its external data files sit next to it.</param>
/// <param name="TokenizerPath">Absolute path of <c>tokenizer.json</c> for the owned tokenizer (profile <c>lfm3-bpe-chat</c>).</param>
/// <param name="Precision">The graph's precision.</param>
/// <param name="Snapshot">The verified snapshot.</param>
public sealed record Lfm3ModelFiles(string ModelPath, string TokenizerPath, Lfm3Precision Precision, ModelSnapshot Snapshot)
{
    /// <summary>Paths of the files in a snapshot verified against <see cref="Lfm3ModelAssets.LockFor"/>.</summary>
    public static Lfm3ModelFiles From(ModelSnapshot snapshot, Lfm3Precision precision)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        return new Lfm3ModelFiles(
            snapshot.GetFilePath(Lfm3ModelAssets.ModelFile(precision)),
            snapshot.GetFilePath(Lfm3ModelAssets.TokenizerFile),
            precision,
            snapshot);
    }

    /// <summary>
    /// SHA-256 of <see cref="ModelPath"/> when <see cref="Snapshot"/> hashed it; otherwise null. Usable as
    /// <c>OnnxSessionOptions.KnownModelSha256</c> in <see cref="Lfm3Options.Session"/>.
    /// </summary>
    public string? ModelSha256 => Snapshot?.GetVerifiedSha256(Lfm3ModelAssets.ModelFile(Precision));
}

/// <summary>Precision of the LFM3 ONNX graph.</summary>
public enum Lfm3Precision
{
    /// <summary><c>onnx/model.onnx</c>: fp32 weights, activations and caches. The parity reference; CPU.</summary>
    Fp32,

    /// <summary>
    /// <c>onnx/model_fp16.onnx</c>: fp16 weights, activations and caches (logits stay fp32). For CUDA: the export's
    /// card notes that the CUDA provider has no fp32 GroupQueryAttention kernel.
    /// </summary>
    Fp16,
}

/// <summary>
/// The immutable asset locks of <c>LiquidAI/LFM3-1.7B-ONNX</c> (one per precision, each the graph, its external data
/// and the tokenizer), and explicit acquisition of those files. The model never downloads; call
/// <see cref="DownloadAsync"/> once (setup step), then <see cref="OpenLocal"/> at startup.
/// </summary>
/// <remarks>
/// The weights are under the LFM Open License v1.0, which licenses commercial use only for entities below 10 million
/// USD annual revenue (its section 5). The application that downloads them accepts those terms.
/// </remarks>
public static class Lfm3ModelAssets
{
    /// <summary>Hub repository of the ONNX export.</summary>
    public const string Repository = "LiquidAI/LFM3-1.7B-ONNX";

    /// <summary>Pinned commit.</summary>
    public const string Revision = "c680759afee1ce8310c90d6c621f4558200dee77";

    /// <summary>Tokenizer path inside the repository.</summary>
    public const string TokenizerFile = "tokenizer.json";

    /// <summary>
    /// SHA-256 of <see cref="TokenizerFile"/>. The export's file is the <c>LiquidAI/LFM3-1.7B</c> file at
    /// <c>40ebe0aa</c> (the <c>lfm3-bpe-chat</c> profile asset, SHA-256 <c>7e226e34…</c>) without its final newline:
    /// the same JSON, one byte shorter.
    /// </summary>
    public const string TokenizerSha256 = "41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44";

    private static readonly ModelAssetFile Tokenizer = new(TokenizerFile, 17908634, TokenizerSha256);

    /// <summary>The fp32 lock: graph, four external data files and the tokenizer (about 7.2 GB).</summary>
    public static ModelAssetLock Fp32Lock { get; } = new(
        "lfm3-1.7b-onnx-fp32",
        Repository,
        Revision,
        [
            new ModelAssetFile("onnx/model.onnx", 274961, "add6653fc8b3cf2c7ee9fa0205a4a9a833fcb85d3449a5fea06f8aa494b5ef77"),
            new ModelAssetFile("onnx/model.onnx_data", 2122694656, "bf44fe98d130ad2bca747e9b62f28b18a3277d45bf1367da15a478538c849057"),
            new ModelAssetFile("onnx/model.onnx_data_1", 2140200960, "bb631a02237e4f8a491e12b25c333849046332b6db60fe0a2120f278c982fa76"),
            new ModelAssetFile("onnx/model.onnx_data_2", 2135916544, "ad39ed4b378c9c1b34c1a3cdb879880c7e110061070ff29fca118e8dd0b48010"),
            new ModelAssetFile("onnx/model.onnx_data_3", 784818176, "1538827aa1c279442457028c4e3d784b7ea82bcf03abd4b55e45fa4facf9fe62"),
            Tokenizer,
        ]);

    /// <summary>The fp16 lock: graph, two external data files and the tokenizer (about 3.6 GB).</summary>
    public static ModelAssetLock Fp16Lock { get; } = new(
        "lfm3-1.7b-onnx-fp16",
        Repository,
        Revision,
        [
            new ModelAssetFile("onnx/model_fp16.onnx", 270733, "d3093d69f6f954bd74f1dc9e59452ec5e9766e62d9c409456fb8ea5698a830fe"),
            new ModelAssetFile("onnx/model_fp16.onnx_data", 2131447808, "38205c6368e442fea49f52f2e66bdcca65fbdb635938ed69efb6ecb91a0bf908"),
            new ModelAssetFile("onnx/model_fp16.onnx_data_1", 1460367360, "9c7dbadee6ca54f41125b8a3641b1368a02cddb3b394f9d09e06a72f359135bb"),
            Tokenizer,
        ]);

    /// <summary>The graph path of <paramref name="precision"/> inside the repository.</summary>
    public static string ModelFile(Lfm3Precision precision) => precision switch
    {
        Lfm3Precision.Fp32 => "onnx/model.onnx",
        Lfm3Precision.Fp16 => "onnx/model_fp16.onnx",
        _ => throw new ArgumentOutOfRangeException(nameof(precision), precision, null),
    };

    /// <summary>The lock of <paramref name="precision"/>.</summary>
    public static ModelAssetLock LockFor(Lfm3Precision precision) => precision switch
    {
        Lfm3Precision.Fp32 => Fp32Lock,
        Lfm3Precision.Fp16 => Fp16Lock,
        _ => throw new ArgumentOutOfRangeException(nameof(precision), precision, null),
    };

    /// <summary>Explicitly downloads (or verifies already cached) the files of one precision. See <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/>.</summary>
    public static async Task<Lfm3ModelFiles> DownloadAsync(
        Lfm3Precision precision = Lfm3Precision.Fp32, ModelAcquisitionOptions? options = null,
        IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        using var store = new ModelAssetStore(options);
        return Lfm3ModelFiles.From(await store.DownloadAsync(LockFor(precision), progress, cancellationToken).ConfigureAwait(false), precision);
    }

    /// <summary>Offline: the cached files of one precision, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    public static Lfm3ModelFiles OpenLocal(Lfm3Precision precision = Lfm3Precision.Fp32, ModelAcquisitionOptions? options = null)
    {
        using var store = new ModelAssetStore(options);
        return Lfm3ModelFiles.From(store.OpenLocal(LockFor(precision)), precision);
    }

    /// <summary>Verifies an existing snapshot directory (for example a Hub snapshot) against the lock of one precision. Never downloads.</summary>
    public static Lfm3ModelFiles OpenDirectory(string directory, Lfm3Precision precision = Lfm3Precision.Fp32, CacheVerification verification = CacheVerification.Sha256) =>
        Lfm3ModelFiles.From(ModelAssetStore.VerifyDirectory(LockFor(precision), directory, verification), precision);
}
