namespace Starlings.Runtime.OnnxWeb.WebGpu;

/// <summary>The WebGPU-capable onnxruntime-web bundle this package serves.</summary>
public static class OnnxWebGpuBundle
{
    /// <summary>Static web asset base path of this package.</summary>
    public const string ContentPath = "_content/Starlings.Runtime.OnnxWeb.WebGpu/";

    /// <summary>
    /// onnxruntime-web 1.30.0's WebGPU bundle (<c>ort.webgpu.min.mjs</c>, <c>ort-wasm-simd-threaded.asyncify.wasm</c>).
    /// Initialize with it to use <see cref="OnnxWebExecutionProvider.WebGpu"/>; it also runs
    /// <see cref="OnnxWebExecutionProvider.Wasm"/>.
    /// </summary>
    /// <example><c>await OnnxWebRuntime.InitializeAsync(new OnnxWebRuntimeOptions { Bundle = OnnxWebGpuBundle.WebGpu });</c></example>
    public static OnnxWebBundle WebGpu { get; } = new("webgpu", ContentPath + "ort/ort.webgpu.min.mjs", ContentPath + "ort/", SupportsWebGpu: true);
}
