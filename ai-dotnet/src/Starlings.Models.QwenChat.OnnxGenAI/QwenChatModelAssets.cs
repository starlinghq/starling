using Starlings.Acquisition;

namespace Starlings.Models.QwenChat;

/// <summary>
/// Explicit acquisition of a profile's bundle (<see cref="QwenChatProfile.BundleFiles"/>) from Starling's hosting. The
/// model never downloads; call <see cref="DownloadAsync"/> once (setup step), then <see cref="OpenLocal"/> at startup and
/// pass the returned directory to <see cref="QwenChatModel.Load"/>.
/// </summary>
/// <remarks>
/// The Hub repositories of the originals ship no ONNX Runtime GenAI build, so the bundle is Starling's reproducible
/// build of the pinned original (<c>model-pipeline/reference/qwen-chat/build_bundle.py</c>, builder 0.17.1), under the
/// license of the weights (Apache-2.0). Each profile's six files are hosted publicly, one Hugging Face repository per
/// profile (<see cref="HostingRepository"/>) at a pinned commit (<see cref="HostingRevision"/>, tag <c>r1</c>), in the
/// bundle layout; downloading needs no token. The digests in the profile, not the address, pin the bytes, so a bundle
/// built locally with the script loads the same way.
/// </remarks>
public static class QwenChatModelAssets
{
    /// <summary>Hugging Face repository hosting the Qwen3-0.6B bundle (public, Starling build r1).</summary>
    public const string Qwen3HostingRepository = "Starlinghq/qwen3-0.6b-ortgenai";

    /// <summary>Pinned commit of <see cref="Qwen3HostingRepository"/> (tag <c>r1</c>).</summary>
    public const string Qwen3HostingRevision = "c24cd59cad1444863ddfdb372736da610a09c9c4";

    /// <summary>Hugging Face repository hosting the Qwen2.5-0.5B-Instruct bundle (public, Starling build r1).</summary>
    public const string Qwen25HostingRepository = "Starlinghq/qwen2.5-0.5b-instruct-ortgenai";

    /// <summary>Pinned commit of <see cref="Qwen25HostingRepository"/> (tag <c>r1</c>).</summary>
    public const string Qwen25HostingRevision = "6b6957c8305ef950450da2c0c846bce6fac3e8a0";

    /// <summary>False: both hosting repositories are public (2026-10-19), so downloads send no token.</summary>
    public const bool HostingIsPrivate = false;

    private static readonly ModelAssetLock Qwen3Lock = Build(QwenChatProfile.Qwen3);
    private static readonly ModelAssetLock Qwen25Lock = Build(QwenChatProfile.Qwen25);

    /// <summary>Hosting repository of a profile's bundle.</summary>
    public static string HostingRepository(QwenChatProfile profile) => Select(profile, Qwen3HostingRepository, Qwen25HostingRepository);

    /// <summary>Pinned commit of <see cref="HostingRepository"/>.</summary>
    public static string HostingRevision(QwenChatProfile profile) => Select(profile, Qwen3HostingRevision, Qwen25HostingRevision);

    /// <summary>Base address of a profile's files: <c>resolve</c> at the pinned commit; a file's source is this plus its name.</summary>
    public static string HostingBaseUrl(QwenChatProfile profile) =>
        "https://huggingface.co/" + HostingRepository(profile) + "/resolve/" + HostingRevision(profile) + "/";

    /// <summary>The acquisition lock of a profile: its six pinned bundle files with their hosted sources.</summary>
    public static ModelAssetLock Lock(QwenChatProfile profile) => Select(profile, Qwen3Lock, Qwen25Lock);

    /// <summary>
    /// Explicitly downloads (or verifies already cached) a profile's bundle and returns its directory. See
    /// <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/>.
    /// </summary>
    public static async Task<string> DownloadAsync(
        QwenChatProfile profile, ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        using var store = new ModelAssetStore(options);
        return (await store.DownloadAsync(Lock(profile), progress, cancellationToken).ConfigureAwait(false)).Directory;
    }

    /// <summary>Offline: a profile's cached bundle directory, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    public static string OpenLocal(QwenChatProfile profile, ModelAcquisitionOptions? options = null)
    {
        using var store = new ModelAssetStore(options);
        return store.OpenLocal(Lock(profile)).Directory;
    }

    private static T Select<T>(QwenChatProfile profile, T qwen3, T qwen25)
    {
        ArgumentNullException.ThrowIfNull(profile);
        return ReferenceEquals(profile, QwenChatProfile.Qwen3) ? qwen3
            : ReferenceEquals(profile, QwenChatProfile.Qwen25) ? qwen25
            : throw new NotSupportedException($"Unsupported Qwen chat profile: {profile.Id}");
    }

    private static ModelAssetLock Build(QwenChatProfile profile)
    {
        string baseUrl = HostingBaseUrl(profile);
        return new ModelAssetLock(profile.Id.Replace('/', '-'), profile.Repository, profile.Revision,
            profile.BundleFiles.Select(f => new ModelAssetFile(f.Path, f.Size, f.Sha256, new Uri(baseUrl + f.Path))));
    }
}
