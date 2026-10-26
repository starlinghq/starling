using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;
using System.Text.Json;

namespace Starlings.Runtime.OnnxWeb;

/// <summary>The execution provider of an <see cref="OnnxWebSession"/>. There is never a silent fallback between them.</summary>
public enum OnnxWebExecutionProvider
{
    /// <summary>onnxruntime-web's WebAssembly CPU backend (SIMD; threads with cross-origin isolation). The qualified path.</summary>
    Wasm,

    /// <summary>WebGPU. Needs the <c>Starlings.Runtime.OnnxWeb.WebGpu</c> bundle and a browser that exposes <c>navigator.gpu</c>. Experimental.</summary>
    WebGpu,
}

/// <summary>
/// One onnxruntime-web distribution served as static web assets: the ES module and the directory holding its
/// Emscripten loader and binary. URLs are relative to the page's base URI (<c>document.baseURI</c>; in a Web Worker, the worker script's URL). A page loads one
/// bundle; <see cref="Wasm"/> is this package's, the WebGPU adapter package provides the other.
/// </summary>
/// <param name="Name">Short name reported in diagnostics.</param>
/// <param name="ModuleUrl">The onnxruntime-web ES module.</param>
/// <param name="WasmDirectoryUrl">Directory of the <c>ort-wasm-*.mjs</c> and <c>.wasm</c> files (<c>ort.env.wasm.wasmPaths</c>).</param>
/// <param name="SupportsWebGpu">Whether the bundle contains the WebGPU execution provider.</param>
public sealed record OnnxWebBundle(string Name, string ModuleUrl, string WasmDirectoryUrl, bool SupportsWebGpu)
{
    /// <summary>Static web asset base path of this package.</summary>
    public const string ContentPath = "_content/Starlings.Runtime.OnnxWeb/";

    /// <summary>
    /// onnxruntime-web 1.30.0's WASM-only bundle shipped by this package (<c>ort.wasm.min.mjs</c>,
    /// <c>ort-wasm-simd-threaded.wasm</c>: 14.2 MB, about 2.3 MB brotli).
    /// </summary>
    public static OnnxWebBundle Wasm { get; } = new("wasm", ContentPath + "ort/ort.wasm.min.mjs", ContentPath + "ort/", SupportsWebGpu: false);
}

/// <summary>Process-wide onnxruntime-web settings, fixed by the first <see cref="OnnxWebRuntime.InitializeAsync"/>.</summary>
public sealed record OnnxWebRuntimeOptions
{
    /// <summary>The onnxruntime-web distribution to load. Default <see cref="OnnxWebBundle.Wasm"/>.</summary>
    public OnnxWebBundle Bundle { get; init; } = OnnxWebBundle.Wasm;

    /// <summary>
    /// WebAssembly threads, or <see langword="null"/> (default) for onnxruntime-web's own default: half the logical
    /// cores, at most 4, when the page is cross-origin isolated, else 1. More than 1 on a page that is not cross-origin
    /// isolated fails with <see cref="NotSupportedException"/> instead of silently running one thread.
    /// </summary>
    public int? NumThreads { get; init; }

    /// <summary>
    /// Runs the WebAssembly backend in onnxruntime-web's proxy worker, so inference never blocks the page's main
    /// thread. Not available with <see cref="OnnxWebExecutionProvider.WebGpu"/>. Default <see langword="false"/>.
    /// Experimental.
    /// </summary>
    public bool Proxy { get; init; }

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Bundle);
        ArgumentException.ThrowIfNullOrWhiteSpace(Bundle.Name, nameof(Bundle));
        ArgumentException.ThrowIfNullOrWhiteSpace(Bundle.ModuleUrl, nameof(Bundle));
        ArgumentException.ThrowIfNullOrWhiteSpace(Bundle.WasmDirectoryUrl, nameof(Bundle));
        if (NumThreads is { } threads)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(threads, 1, nameof(NumThreads));
            ArgumentOutOfRangeException.ThrowIfGreaterThan(threads, 64, nameof(NumThreads));
        }
    }
}

/// <summary>What onnxruntime-web reported after initialization.</summary>
/// <param name="OrtVersion">onnxruntime-web version (<c>ort.env.versions.web</c>).</param>
/// <param name="Bundle">The bundle name (<see cref="OnnxWebBundle.Name"/>).</param>
/// <param name="CrossOriginIsolated">Whether the page is cross-origin isolated (needed for more than one thread).</param>
/// <param name="HardwareConcurrency">The browser's <c>navigator.hardwareConcurrency</c>.</param>
/// <param name="NumThreads">WebAssembly threads in effect.</param>
/// <param name="Proxy">Whether the proxy worker is on.</param>
/// <param name="WebGpuExposed">Whether the browser exposes <c>navigator.gpu</c> (an adapter may still be unavailable).</param>
/// <param name="SecureContext">Whether the page is a secure context (needed for SHA-256 verification).</param>
public sealed record OnnxWebRuntimeInfo(
    string? OrtVersion,
    string Bundle,
    bool CrossOriginIsolated,
    int HardwareConcurrency,
    int NumThreads,
    bool Proxy,
    bool WebGpuExposed,
    bool SecureContext)
{
    internal static OnnxWebRuntimeInfo Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        return new OnnxWebRuntimeInfo(
            root.TryGetProperty("ort_version", out var version) && version.ValueKind == JsonValueKind.String ? version.GetString() : null,
            root.GetProperty("bundle").GetString() ?? string.Empty,
            root.GetProperty("cross_origin_isolated").GetBoolean(),
            root.GetProperty("hardware_concurrency").GetInt32(),
            root.GetProperty("num_threads").GetInt32(),
            root.GetProperty("proxy").GetBoolean(),
            root.GetProperty("webgpu_exposed").GetBoolean(),
            root.GetProperty("secure_context").GetBoolean());
    }
}

/// <summary>
/// Loads onnxruntime-web once per page. Call <see cref="InitializeAsync"/> before creating an
/// <see cref="OnnxWebSession"/>; later calls with equal options return the same result.
/// </summary>
/// <remarks>
/// The calls go through [JSImport], which runs on the browser's main thread: this package supports the default
/// single-threaded .NET browser runtime (Blazor WebAssembly), not <c>WasmEnableThreads</c>.
/// </remarks>
public static class OnnxWebRuntime
{
    /// <summary>The onnxruntime-web version this package ships and was tested with.</summary>
    public const string OnnxRuntimeWebVersion = "1.30.0";

    /// <summary>Path of the JavaScript module, relative to the page's base URI.</summary>
    internal const string ModulePath = OnnxWebBundle.ContentPath + "starlings-onnxweb.js";

    private static readonly Lock Gate = new();
    private static Task? _module;
    private static Task<OnnxWebRuntimeInfo>? _initialization;
    private static OnnxWebRuntimeOptions? _initializedWith;
    private static OnnxWebRuntimeInfo? _info;

    /// <summary>What onnxruntime-web reported, or <see langword="null"/> before <see cref="InitializeAsync"/> completed.</summary>
    public static OnnxWebRuntimeInfo? Info => Volatile.Read(ref _info);

    /// <summary>
    /// Imports the module and onnxruntime-web with <paramref name="options"/>. Idempotent for equal options; different
    /// options after a successful initialization throw <see cref="InvalidOperationException"/>, because a page loads
    /// onnxruntime-web once. A failed initialization can be retried.
    /// </summary>
    /// <exception cref="NotSupportedException">More than one thread was requested on a page that is not cross-origin isolated.</exception>
    [SupportedOSPlatform("browser")]
    public static Task<OnnxWebRuntimeInfo> InitializeAsync(OnnxWebRuntimeOptions? options = null)
    {
        options ??= new OnnxWebRuntimeOptions();
        options.Validate();
        lock (Gate)
        {
            if (_initialization is not null)
            {
                if (!options.Equals(_initializedWith))
                {
                    throw new InvalidOperationException("onnxruntime-web is already initialized with other options; a page initializes it once.");
                }

                return _initialization;
            }

            _initializedWith = options;
            var initialization = InitializeCoreAsync(options);
            // A failure that completes synchronously has already run the reset in InitializeCoreAsync's catch; caching
            // the faulted task here would make every retry fail with "already initialized".
            if (!initialization.IsFaulted && !initialization.IsCanceled)
            {
                _initialization = initialization;
            }
            else
            {
                _initializedWith = null;
            }

            return initialization;
        }
    }

    [SupportedOSPlatform("browser")]
    private static async Task<OnnxWebRuntimeInfo> InitializeCoreAsync(OnnxWebRuntimeOptions options)
    {
        try
        {
            await EnsureModuleAsync().ConfigureAwait(false);
            string json;
            try
            {
                json = await OnnxWebInterop.Initialize(
                    options.Bundle.ModuleUrl, options.Bundle.WasmDirectoryUrl, options.Bundle.Name, options.NumThreads ?? 0, options.Proxy).ConfigureAwait(false);
            }
            catch (JSException ex) when (ex.Message.Contains("cross-origin isolation", StringComparison.Ordinal))
            {
                throw new NotSupportedException(ex.Message, ex);
            }

            var info = OnnxWebRuntimeInfo.Parse(json);
            Volatile.Write(ref _info, info);
            return info;
        }
        catch
        {
            lock (Gate)
            {
                _initialization = null;
                _initializedWith = null;
            }

            throw;
        }
    }

    /// <summary>Imports <c>starlings-onnxweb.js</c> once (resolved against <see cref="BaseUri"/>).</summary>
    [SupportedOSPlatform("browser")]
    internal static Task EnsureModuleAsync()
    {
        lock (Gate)
        {
            if (_module is not null)
            {
                return _module;
            }

            var module = ImportModuleAsync();
            // Same as InitializeAsync: never cache a task that already failed synchronously.
            if (!module.IsFaulted && !module.IsCanceled)
            {
                _module = module;
            }

            return module;
        }
    }

    /// <summary>
    /// The base the package's URLs resolve against: the page's <c>document.baseURI</c>, or in a Web Worker (no
    /// document) the worker script's <c>location.href</c>, so a worker next to <c>_framework/</c> and
    /// <c>_content/</c> resolves them as the page would.
    /// </summary>
    [SupportedOSPlatform("browser")]
    internal static string BaseUri()
    {
        using JSObject? document = JSHost.GlobalThis.GetPropertyAsJSObject("document");
        if (document?.GetPropertyAsString("baseURI") is { Length: > 0 } baseUri)
        {
            return baseUri;
        }

        using JSObject location = JSHost.GlobalThis.GetPropertyAsJSObject("location")
            ?? throw new NotSupportedException("Starlings.Runtime.OnnxWeb runs on a browser page or in a Web Worker (no document or location here).");
        return location.GetPropertyAsString("href") ?? throw new NotSupportedException("location.href is not available.");
    }

    [SupportedOSPlatform("browser")]
    private static async Task ImportModuleAsync()
    {
        try
        {
            var url = new Uri(new Uri(BaseUri()), ModulePath);
            await JSHost.ImportAsync(OnnxWebInterop.ModuleName, url.AbsoluteUri).ConfigureAwait(false);
        }
        catch
        {
            lock (Gate)
            {
                _module = null;
            }

            throw;
        }
    }

    /// <summary>
    /// Throws if <paramref name="provider"/> cannot run on this page with the initialized bundle and options: the check
    /// <see cref="OnnxWebSession.CreateAsync"/> makes, available before any model is downloaded.
    /// </summary>
    /// <exception cref="InvalidOperationException"><see cref="InitializeAsync"/> has not completed.</exception>
    /// <exception cref="NotSupportedException">
    /// WebGPU was requested but the bundle has no WebGPU provider, the browser does not expose WebGPU, or the proxy worker
    /// is on.
    /// </exception>
    public static void EnsureProviderAvailable(OnnxWebExecutionProvider provider)
    {
        if (!Enum.IsDefined(provider))
        {
            throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unknown execution provider.");
        }

        var runtime = RequireInitialized();
        if (provider != OnnxWebExecutionProvider.WebGpu)
        {
            return;
        }

        if (!CurrentBundleSupportsWebGpu)
        {
            throw new NotSupportedException($"The onnxruntime-web bundle '{runtime.Bundle}' has no WebGPU provider. Reference Starlings.Runtime.OnnxWeb.WebGpu and initialize with its bundle.");
        }

        if (!runtime.WebGpuExposed)
        {
            throw new NotSupportedException("This browser does not expose WebGPU (navigator.gpu is undefined).");
        }

        if (runtime.Proxy)
        {
            throw new NotSupportedException("onnxruntime-web's proxy worker supports only the WASM provider.");
        }
    }

    internal static bool CurrentBundleSupportsWebGpu
    {
        get
        {
            lock (Gate)
            {
                return _initializedWith?.Bundle.SupportsWebGpu == true;
            }
        }
    }

    internal static OnnxWebRuntimeInfo RequireInitialized() =>
        Info ?? throw new InvalidOperationException("Call OnnxWebRuntime.InitializeAsync (and await it) before creating a session.");
}
