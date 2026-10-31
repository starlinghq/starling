using Starlings.Acquisition;

namespace Starlings.Models.Whisper;

/// <summary>Verified local paths of the whisper runtime files.</summary>
/// <param name="EncoderPath">Absolute path of <c>onnx/encoder_model_fp16.onnx</c>.</param>
/// <param name="DecoderPath">Absolute path of <c>onnx/decoder_model_fp16.onnx</c>.</param>
/// <param name="TokenizerPath">Absolute path of <c>tokenizer.json</c> for the owned tokenizer (profile <c>whisper-bpe</c>).</param>
/// <param name="Snapshot">The verified snapshot.</param>
public sealed record WhisperModelFiles(string EncoderPath, string DecoderPath, string TokenizerPath, ModelSnapshot Snapshot)
{
    /// <summary>Paths of the whisper files in a snapshot verified against <see cref="WhisperModelAssets.Lock"/>.</summary>
    public static WhisperModelFiles From(ModelSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        return new WhisperModelFiles(
            snapshot.GetFilePath(WhisperModelAssets.EncoderFile),
            snapshot.GetFilePath(WhisperModelAssets.DecoderFile),
            snapshot.GetFilePath(WhisperModelAssets.TokenizerFile),
            snapshot);
    }

    /// <summary>
    /// SHA-256 of <see cref="EncoderPath"/> when <see cref="Snapshot"/> hashed it (<see cref="ModelSnapshot.Verification"/>
    /// <see cref="CacheVerification.Sha256"/>); otherwise null. Usable as <c>OnnxSessionOptions.KnownModelSha256</c> in
    /// <see cref="WhisperOptions.EncoderSession"/>.
    /// </summary>
    public string? EncoderSha256 => Snapshot?.GetVerifiedSha256(WhisperModelAssets.EncoderFile);

    /// <summary>
    /// SHA-256 of <see cref="DecoderPath"/> when <see cref="Snapshot"/> hashed it; otherwise null. Usable as
    /// <c>OnnxSessionOptions.KnownModelSha256</c> in <see cref="WhisperOptions.DecoderSession"/>.
    /// </summary>
    public string? DecoderSha256 => Snapshot?.GetVerifiedSha256(WhisperModelAssets.DecoderFile);

    /// <summary>
    /// Absolute path of <c>onnx/decoder_with_past_model_fp16.onnx</c> (the KV-cache decoder) in <see cref="Snapshot"/>;
    /// null without a snapshot, which leaves <see cref="WhisperModel"/> on the full-context loop.
    /// </summary>
    public string? DecoderWithPastPath { get; init; } = Snapshot?.GetFilePath(WhisperModelAssets.DecoderWithPastFile);

    /// <summary>SHA-256 of <see cref="DecoderWithPastPath"/> when <see cref="Snapshot"/> hashed it; otherwise null.</summary>
    public string? DecoderWithPastSha256 => Snapshot?.GetVerifiedSha256(WhisperModelAssets.DecoderWithPastFile);
}

/// <summary>
/// The immutable asset lock of the whisper-large-v3-turbo files (fp16 ONNX graphs plus the tokenizer), and explicit
/// acquisition of those files. The model never downloads; call <see cref="DownloadAsync"/> once (setup step), then
/// <see cref="OpenLocal"/> at startup.
/// </summary>
public static class WhisperModelAssets
{
    /// <summary>Hub repository.</summary>
    public const string Repository = "onnx-community/whisper-large-v3-turbo-ONNX";

    /// <summary>Pinned commit.</summary>
    public const string Revision = "76ba4fb597cca51a329a1820e49282e26248b36b";

    /// <summary>Encoder graph path inside the repository.</summary>
    public const string EncoderFile = "onnx/encoder_model_fp16.onnx";

    /// <summary>Decoder graph path inside the repository.</summary>
    public const string DecoderFile = "onnx/decoder_model_fp16.onnx";

    /// <summary>KV-cache decoder graph path inside the repository.</summary>
    public const string DecoderWithPastFile = "onnx/decoder_with_past_model_fp16.onnx";

    /// <summary>Tokenizer path inside the repository.</summary>
    public const string TokenizerFile = "tokenizer.json";

    /// <summary>SHA-256 of <see cref="EncoderFile"/>.</summary>
    public const string EncoderSha256 = "b48fd309c51e77ba88324469632fbff8796d2dd37b1658c9516c1c1b1d96428d";

    /// <summary>SHA-256 of <see cref="DecoderFile"/>.</summary>
    public const string DecoderSha256 = "15870b95e08e666ea2423738ee12722de965d09cf05f4579fa947357332b5057";

    /// <summary>SHA-256 of <see cref="DecoderWithPastFile"/>.</summary>
    public const string DecoderWithPastSha256 = "33041ca209a268f9562d5ff544542ed219c3a4342209e95a34042e9124f055ad";

    /// <summary>SHA-256 of <see cref="TokenizerFile"/>.</summary>
    public const string TokenizerSha256 = "b3c8202bbf06d8ee4232c5984baa563784ac4737e2e7fdc42fa180200d3cfcdb";

    /// <summary>The lock: the encoder, the full-context and KV-cache decoders, and the tokenizer (about 1.9 GB).</summary>
    public static ModelAssetLock Lock { get; } = new(
        "whisper-large-v3-turbo",
        Repository,
        Revision,
        [
            new ModelAssetFile(EncoderFile, 1274342603, EncoderSha256),
            new ModelAssetFile(DecoderFile, 343974281, DecoderSha256),
            new ModelAssetFile(DecoderWithPastFile, 317742004, DecoderWithPastSha256),
            new ModelAssetFile(TokenizerFile, 2480645, TokenizerSha256),
        ]);

    /// <summary>Explicitly downloads (or verifies already cached) whisper files. See <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/>.</summary>
    public static async Task<WhisperModelFiles> DownloadAsync(
        ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        using var store = new ModelAssetStore(options);
        return WhisperModelFiles.From(await store.DownloadAsync(Lock, progress, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Offline: the cached whisper files, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    public static WhisperModelFiles OpenLocal(ModelAcquisitionOptions? options = null)
    {
        using var store = new ModelAssetStore(options);
        return WhisperModelFiles.From(store.OpenLocal(Lock));
    }

    /// <summary>Verifies an existing snapshot directory (for example a Hub snapshot) against the lock. Never downloads.</summary>
    public static WhisperModelFiles OpenDirectory(string directory, CacheVerification verification = CacheVerification.Sha256) =>
        WhisperModelFiles.From(ModelAssetStore.VerifyDirectory(Lock, directory, verification));
}
