using Starlings.Acquisition;

namespace Starlings.Models.Kev;

/// <summary>Verified local paths of one Kev bundle.</summary>
/// <param name="Profile">The profile the bundle was verified against.</param>
/// <param name="ModelPath">Absolute path of <c>model.onnx</c> (its weights file <c>model.onnx.data</c> sits next to it).</param>
/// <param name="HeadPath">Absolute path of <c>head.safetensors</c> (pointer head, fp32).</param>
/// <param name="HeadMetadataPath">Absolute path of <c>head.json</c> (temperature, dimensions, source digests).</param>
/// <param name="TokenizerPath">Absolute path of <c>tokenizer.json</c> (owned tokenizer profile <c>qwen2-bpe-kev</c>).</param>
/// <param name="Snapshot">The verified snapshot.</param>
public sealed record KevModelFiles(
    KevModelProfile Profile, string ModelPath, string HeadPath, string HeadMetadataPath, string TokenizerPath, ModelSnapshot Snapshot)
{
    /// <summary>Paths of the Kev files in a snapshot verified against <see cref="KevModelAssets.Lock"/>.</summary>
    public static KevModelFiles From(KevModelProfile profile, ModelSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(snapshot);
        return new KevModelFiles(
            profile,
            snapshot.GetFilePath(KevModelAssets.ModelFile),
            snapshot.GetFilePath(KevModelAssets.HeadFile),
            snapshot.GetFilePath(KevModelAssets.HeadMetadataFile),
            snapshot.GetFilePath(KevModelAssets.TokenizerFile),
            snapshot);
    }
}

/// <summary>
/// The immutable asset locks of the profiles <c>kev-0.8b/onnx-fp32-cpu</c> and <c>kev-4b/onnx-fp32-cpu</c>, and
/// verification of a local bundle against them. Nothing here downloads.
/// </summary>
/// <remarks>
/// <para>
/// No Hub repository ships an fp32 ONNX graph of Kev 1.0, so the bundle is Starling's build from the pinned sources
/// (route A of <c>model-pipeline/admissions/kev.md</c>): <c>model-pipeline/reference/kev/kev_merge.py</c> merges the
/// LoRA into the base in fp32 and writes the pointer head as <c>head.safetensors</c> plus <c>head.json</c> (so nothing
/// unpickles <c>head.pt</c>); the ONNX Runtime GenAI 0.17.1 model builder exports the backbone
/// (<c>-p fp32 -e cpu --extra_options exclude_lm_head=true exclude_mtp=true</c>). <c>tokenizer.json</c> is the Kev
/// repository's file at the pinned commit (the same bytes for both sizes).
/// </para>
/// <para>
/// The graphs are not hosted yet, so there is no download: build the bundle (or receive it) into a directory and open
/// it with <see cref="OpenDirectory"/>, which checks every file's size and SHA-256.
/// </para>
/// </remarks>
public static class KevModelAssets
{
    /// <summary>Path of the ONNX graph in the bundle directory.</summary>
    public const string ModelFile = "model.onnx";

    /// <summary>Path of the graph's external weights in the bundle directory (referenced by <see cref="ModelFile"/>).</summary>
    public const string ModelDataFile = "model.onnx.data";

    /// <summary>Path of the pointer head weights.</summary>
    public const string HeadFile = "head.safetensors";

    /// <summary>Path of the pointer head metadata.</summary>
    public const string HeadMetadataFile = "head.json";

    /// <summary>Path of the tokenizer.</summary>
    public const string TokenizerFile = "tokenizer.json";

    /// <summary>SHA-256 of <see cref="TokenizerFile"/> (both Kev repositories at their pinned commits).</summary>
    public const string TokenizerSha256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";

    /// <summary>Size of <see cref="TokenizerFile"/> in bytes.</summary>
    public const long TokenizerSize = 19_989_325;

    /// <summary>The Kev-0.8B bundle (about 3.1 GB).</summary>
    public static ModelAssetLock Kev08BLock { get; } = Bundle(
        KevModelProfile.Kev08B,
        modelSize: 307_884, modelSha256: "046aa7b4367dee6c29bd494f330e60c5d2f665540558e26ea02148ecfd7f6278",
        dataSize: 3_077_177_344, dataSha256: "dd78b21c101ef945dc4b7ca31367647da70e4d7fbd3f3d2e05a3245c573dc51a",
        headSize: 2_099_504, headSha256: "ab7c75921e79ec27ca97d8bf7b928e277dac843a6f4a528f9e8b7ab0fe7cd47d",
        headMetadataSize: 393, headMetadataSha256: "5fc39caf463d2625472240541b5a1e5646335ca9b940c094a29f22b2d43b255a");

    /// <summary>The Kev-4B bundle (about 17 GB).</summary>
    public static ModelAssetLock Kev4BLock { get; } = Bundle(
        KevModelProfile.Kev4B,
        modelSize: 410_133, modelSha256: "364540ffdadd6b7514c7a25d63e21cc3db61784a9a540835cd1458f8023b1cb5",
        dataSize: 16_890_920_960, dataSha256: "e8abebf694f97fde5c7ebc70391c38a15bb9bf2bc07f49449f786faf3863a063",
        headSize: 5_245_232, headSha256: "65698f2568fc1b6ebb60a1c2a24c7bb6622dcc23e1c6d9efe8d3274ad918a9a7",
        headMetadataSize: 392, headMetadataSha256: "df59ee369457425287e772462db515eae3060a8638c8b768bff2ead6f5d83097");

    /// <summary>The lock of a model.</summary>
    public static ModelAssetLock Lock(KevModel model) => model switch
    {
        KevModel.Kev08B => Kev08BLock,
        KevModel.Kev4B => Kev4BLock,
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, "Unknown Kev model."),
    };

    /// <summary>SHA-256 of a profile's <see cref="ModelFile"/>; usable as <c>OnnxSessionOptions.ExpectedModelSha256</c>.</summary>
    public static string ModelSha256(KevModel model) => Lock(model).GetFile(ModelFile).Sha256;

    /// <summary>
    /// Verifies a directory holding the five bundle files against the profile's lock. Never downloads.
    /// <see cref="CacheVerification.Sha256"/> hashes every file (the 4B weights take about a minute);
    /// <see cref="CacheVerification.Size"/> checks sizes only.
    /// </summary>
    /// <exception cref="ModelAssetMissingException">A file is missing.</exception>
    /// <exception cref="ModelAssetIntegrityException">A file has the wrong size or digest.</exception>
    public static KevModelFiles OpenDirectory(KevModel model, string directory, CacheVerification verification = CacheVerification.Sha256) =>
        KevModelFiles.From(KevModelProfile.For(model), ModelAssetStore.VerifyDirectory(Lock(model), directory, verification));

    private static ModelAssetLock Bundle(
        KevModelProfile profile,
        long modelSize, string modelSha256,
        long dataSize, string dataSha256,
        long headSize, string headSha256,
        long headMetadataSize, string headMetadataSha256) => new(
        profile.Id.Split('/')[0],
        profile.Repository,
        profile.Revision,
        [
            new ModelAssetFile(ModelFile, modelSize, modelSha256),
            new ModelAssetFile(ModelDataFile, dataSize, dataSha256),
            new ModelAssetFile(HeadFile, headSize, headSha256),
            new ModelAssetFile(HeadMetadataFile, headMetadataSize, headMetadataSha256),
            new ModelAssetFile(TokenizerFile, TokenizerSize, TokenizerSha256),
        ]);
}
