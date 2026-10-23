namespace Starlings.Models.E5.Browser;

/// <summary>Which e5 graph the browser runs. Each is its own registry profile with its own tolerances.</summary>
public enum E5BrowserVariant
{
    /// <summary>
    /// The pinned <c>onnx/model.onnx</c> (470 MB, about 249 MB brotli): the reference graph, within the binding
    /// <c>onnx-fp32-cpu</c> tolerance in the browser. Loads from the Hub by default (CORS enabled).
    /// </summary>
    Fp32,

    /// <summary>
    /// int8 embedding tables (weight-only, every MatMul fp32), 182 MB (about 109 MB brotli): cosine 0.9998 to the
    /// reference and batch-independent. Not on the Hub: build it with
    /// <c>model-pipeline/reference/encoder/quantize_e5_browser.py</c> (deterministic) and host it next to the app.
    /// </summary>
    Int8Embed,
}

/// <summary>A pinned browser graph: profile, file size and SHA-256, and the default URL if one exists.</summary>
/// <param name="Variant">The variant.</param>
/// <param name="ProfileWasm">Registry profile on the WASM provider.</param>
/// <param name="ProfileWebGpu">Registry profile on the WebGPU provider.</param>
/// <param name="Sha256">SHA-256 of the graph; the loader enforces it.</param>
/// <param name="Size">Size in bytes.</param>
/// <param name="DefaultUrl">Where the graph loads from by default, or <see langword="null"/> when the app must host it.</param>
public sealed record E5BrowserGraph(E5BrowserVariant Variant, string ProfileWasm, string ProfileWebGpu, string Sha256, long Size, string? DefaultUrl);

/// <summary>
/// The pinned files of the e5 browser profiles. The tokenizer and the fp32 graph are the files of
/// <see cref="E5ModelAssets.Lock"/>; the int8-embed graph is derived from that fp32 graph by
/// <c>quantize_e5_browser.py</c> (onnxruntime 1.30.0's <c>quantize_dynamic</c> on Gather only).
/// </summary>
public static class E5BrowserAssets
{
    /// <summary>Hub download URL of a file at the pinned revision (the Hub sends CORS headers, so a page can fetch it directly).</summary>
    public static string HubUrl(string path) =>
        $"https://huggingface.co/{E5ModelAssets.Repository}/resolve/{E5ModelAssets.Revision}/{path}";

    /// <summary>Default URL of <c>tokenizer.json</c> (17 MB, about 2.5 MB brotli; the Hub serves it uncompressed).</summary>
    public static string TokenizerUrl { get; } = HubUrl(E5ModelAssets.TokenizerFile);

    /// <summary>The fp32 graph.</summary>
    public static E5BrowserGraph Fp32 { get; } = new(
        E5BrowserVariant.Fp32,
        "multilingual-e5-small/onnx-fp32-wasm",
        "multilingual-e5-small/onnx-fp32-webgpu",
        E5ModelAssets.ModelSha256,
        470_268_510,
        HubUrl(E5ModelAssets.ModelFile));

    /// <summary>The int8-embed graph (no public URL: host the file built by <c>quantize_e5_browser.py</c>).</summary>
    public static E5BrowserGraph Int8Embed { get; } = new(
        E5BrowserVariant.Int8Embed,
        "multilingual-e5-small/onnx-int8embed-wasm",
        "multilingual-e5-small/onnx-int8embed-webgpu",
        "374a3cc034a1dba112ecd1a9fba920be63f295cada3c65ab15a0a3a84739fb19",
        181_715_582,
        DefaultUrl: null);

    /// <summary>The pinned graph of <paramref name="variant"/>.</summary>
    public static E5BrowserGraph For(E5BrowserVariant variant) => variant switch
    {
        E5BrowserVariant.Fp32 => Fp32,
        E5BrowserVariant.Int8Embed => Int8Embed,
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unknown e5 browser variant."),
    };
}
