using Starlings.Acquisition;

namespace Starlings.Models.PaddleOcr;

/// <summary>Verified local paths of the PP-OCRv5 mobile detector and recognizer files.</summary>
/// <param name="DetectorPath">Absolute path of the detector's <c>inference.onnx</c>.</param>
/// <param name="DetectorConfigPath">Absolute path of the detector's <c>inference.yml</c>.</param>
/// <param name="RecognizerPath">Absolute path of the recognizer's <c>inference.onnx</c>.</param>
/// <param name="RecognizerConfigPath">Absolute path of the recognizer's <c>inference.yml</c> (holds the dictionary).</param>
/// <param name="Detector">The verified detector snapshot.</param>
/// <param name="Recognizer">The verified recognizer snapshot.</param>
public sealed record PaddleOcrModelFiles(
    string DetectorPath,
    string DetectorConfigPath,
    string RecognizerPath,
    string RecognizerConfigPath,
    ModelSnapshot? Detector = null,
    ModelSnapshot? Recognizer = null)
{
    /// <summary>Paths of the files in two snapshots verified against <see cref="PaddleOcrModelAssets.DetectorLock"/> and <see cref="PaddleOcrModelAssets.RecognizerLock"/>.</summary>
    public static PaddleOcrModelFiles From(ModelSnapshot detector, ModelSnapshot recognizer)
    {
        ArgumentNullException.ThrowIfNull(detector);
        ArgumentNullException.ThrowIfNull(recognizer);
        return new PaddleOcrModelFiles(
            detector.GetFilePath(PaddleOcrModelAssets.GraphFile),
            detector.GetFilePath(PaddleOcrModelAssets.ConfigFile),
            recognizer.GetFilePath(PaddleOcrModelAssets.GraphFile),
            recognizer.GetFilePath(PaddleOcrModelAssets.ConfigFile),
            detector,
            recognizer);
    }

    /// <summary>SHA-256 of <see cref="DetectorPath"/> when <see cref="Detector"/> hashed it; otherwise null.</summary>
    public string? DetectorSha256 => Detector?.GetVerifiedSha256(PaddleOcrModelAssets.GraphFile);

    /// <summary>SHA-256 of <see cref="RecognizerPath"/> when <see cref="Recognizer"/> hashed it; otherwise null.</summary>
    public string? RecognizerSha256 => Recognizer?.GetVerifiedSha256(PaddleOcrModelAssets.GraphFile);
}

/// <summary>
/// The immutable asset locks of the PP-OCRv5 mobile files: the official PaddlePaddle ONNX exports of the detector and
/// the recognizer, each with its <c>inference.yml</c>. The model never downloads; call <see cref="DownloadAsync"/> once
/// (setup step), then <see cref="OpenLocal"/> at startup.
/// </summary>
public static class PaddleOcrModelAssets
{
    /// <summary>Detector repository.</summary>
    public const string DetectorRepository = "PaddlePaddle/PP-OCRv5_mobile_det_onnx";

    /// <summary>Pinned detector commit.</summary>
    public const string DetectorRevision = "e6f4fa85f00e168c862bc462aebca69eef9b3d3d";

    /// <summary>Recognizer repository.</summary>
    public const string RecognizerRepository = "PaddlePaddle/PP-OCRv5_mobile_rec_onnx";

    /// <summary>Pinned recognizer commit.</summary>
    public const string RecognizerRevision = "ed152b8b495f84de93cda5709d768548a9127622";

    /// <summary>Graph path inside either repository.</summary>
    public const string GraphFile = "inference.onnx";

    /// <summary>Configuration path inside either repository.</summary>
    public const string ConfigFile = "inference.yml";

    /// <summary>SHA-256 of the detector graph.</summary>
    public const string DetectorSha256 = "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d";

    /// <summary>SHA-256 of the detector configuration.</summary>
    public const string DetectorConfigSha256 = "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699";

    /// <summary>SHA-256 of the recognizer graph.</summary>
    public const string RecognizerSha256 = "da72dc72ca4dc220df0dfde68c1dedc31c58d3e76a25871122e5056227d50092";

    /// <summary>SHA-256 of the recognizer configuration.</summary>
    public const string RecognizerConfigSha256 = "5dfeb2777f6d0db8177d8128a8acfcf6e6276dc4ac73ea3bf0dc06d6a5e85d8e";

    /// <summary>The detector lock (about 4.8 MB).</summary>
    public static ModelAssetLock DetectorLock { get; } = new(
        "ppocrv5-mobile-det",
        DetectorRepository,
        DetectorRevision,
        [
            new ModelAssetFile(GraphFile, 4826518, DetectorSha256),
            new ModelAssetFile(ConfigFile, 903, DetectorConfigSha256),
        ]);

    /// <summary>The recognizer lock (about 16.7 MB).</summary>
    public static ModelAssetLock RecognizerLock { get; } = new(
        "ppocrv5-mobile-rec",
        RecognizerRepository,
        RecognizerRevision,
        [
            new ModelAssetFile(GraphFile, 16534782, RecognizerSha256),
            new ModelAssetFile(ConfigFile, 148345, RecognizerConfigSha256),
        ]);

    /// <summary>Explicitly downloads (or verifies already cached) detector and recognizer files.</summary>
    public static async Task<PaddleOcrModelFiles> DownloadAsync(
        ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        using var store = new ModelAssetStore(options);
        var detector = await store.DownloadAsync(DetectorLock, progress, cancellationToken).ConfigureAwait(false);
        var recognizer = await store.DownloadAsync(RecognizerLock, progress, cancellationToken).ConfigureAwait(false);
        return PaddleOcrModelFiles.From(detector, recognizer);
    }

    /// <summary>Offline: the cached files, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    public static PaddleOcrModelFiles OpenLocal(ModelAcquisitionOptions? options = null)
    {
        using var store = new ModelAssetStore(options);
        return PaddleOcrModelFiles.From(store.OpenLocal(DetectorLock), store.OpenLocal(RecognizerLock));
    }

    /// <summary>Verifies two existing snapshot directories (for example Hub snapshots) against the locks. Never downloads.</summary>
    public static PaddleOcrModelFiles OpenDirectories(string detectorDirectory, string recognizerDirectory, CacheVerification verification = CacheVerification.Sha256) =>
        PaddleOcrModelFiles.From(
            ModelAssetStore.VerifyDirectory(DetectorLock, detectorDirectory, verification),
            ModelAssetStore.VerifyDirectory(RecognizerLock, recognizerDirectory, verification));
}
