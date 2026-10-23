using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;

namespace Starlings.Runtime.OnnxWeb;

/// <summary>
/// The [JSImport] bindings of <c>wwwroot/starlings-onnxweb.js</c>. Arrays cannot travel in a promise result, so
/// tensors and fetched bytes stay JavaScript objects (<see cref="JSObject"/> handles) and are copied in and out
/// synchronously through memory views.
/// </summary>
[SupportedOSPlatform("browser")]
internal static partial class OnnxWebInterop
{
    internal const string ModuleName = "starlings-onnxweb";

    [JSImport("initialize", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.String>>]
    internal static partial Task<string> Initialize(string bundleUrl, string wasmDirectoryUrl, string bundleName, int numThreads, bool proxy);

    [JSImport("fetchAsset", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.Object>>]
    internal static partial Task<JSObject> FetchAsset(
        string url,
        string? expectedSha256,
        string? cacheName,
        int requestId,
        [JSMarshalAs<JSType.Function<JSType.Number, JSType.Number>>] Action<double, double> onProgress);

    [JSImport("abortFetch", ModuleName)]
    internal static partial void AbortFetch(int requestId);

    [JSImport("assetLength", ModuleName)]
    internal static partial double AssetLength(JSObject asset);

    [JSImport("copyAsset", ModuleName)]
    internal static partial void CopyAsset(JSObject asset, [JSMarshalAs<JSType.MemoryView>] Span<byte> destination);

    [JSImport("releaseAsset", ModuleName)]
    internal static partial void ReleaseAsset(JSObject asset);

    [JSImport("createSession", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.Object>>]
    internal static partial Task<JSObject> CreateSession(JSObject asset, string executionProvider, string graphOptimizationLevel);

    [JSImport("sessionMetadata", ModuleName)]
    internal static partial string SessionMetadata(JSObject session);

    [JSImport("createTensor", ModuleName)]
    internal static partial JSObject CreateTensor(
        string type,
        [JSMarshalAs<JSType.MemoryView>] Span<byte> data,
        [JSMarshalAs<JSType.Array<JSType.Number>>] int[] dims);

    [JSImport("run", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.Object>>]
    internal static partial Task<JSObject> Run(
        JSObject session,
        [JSMarshalAs<JSType.Array<JSType.String>>] string[] names,
        [JSMarshalAs<JSType.Array<JSType.Object>>] JSObject[] tensors,
        string outputName);

    [JSImport("tensorType", ModuleName)]
    internal static partial string TensorType(JSObject tensor);

    [JSImport("tensorDims", ModuleName)]
    [return: JSMarshalAs<JSType.Array<JSType.Number>>]
    internal static partial int[] TensorDims(JSObject tensor);

    [JSImport("copyTensor", ModuleName)]
    internal static partial void CopyTensor(JSObject tensor, [JSMarshalAs<JSType.MemoryView>] Span<byte> destination);

    [JSImport("disposeTensor", ModuleName)]
    internal static partial void DisposeTensor(JSObject tensor);

    [JSImport("releaseSession", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.Void>>]
    internal static partial Task ReleaseSession(JSObject session);
}
