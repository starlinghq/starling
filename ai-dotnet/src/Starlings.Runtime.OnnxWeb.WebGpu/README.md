# Starlings.Runtime.OnnxWeb.WebGpu

The voluntary WebGPU adapter of `Starlings.Runtime.OnnxWeb`. It ships onnxruntime-web 1.30.0's WebGPU-capable bundle (`ort.webgpu.min.mjs` and its asyncify binary, 26.8 MB, about 4 MB brotli) as static web assets, and `OnnxWebGpuBundle.WebGpu` to initialize with. No core package references it; reference it only if you want WebGPU.

## Maturity

<!-- catalog:maturity profiles=multilingual-e5-small/onnx-*-webgpu -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-webgpu` | experimental | linux-x64 webgpu chromium-153: experimental; linux-x64 webgpu firefox: untested; macos-arm64 webgpu safari: untested; windows-x64 webgpu chromium: untested |
| `multilingual-e5-small/onnx-int8embed-webgpu` | experimental | linux-x64 webgpu chromium-153: experimental; linux-x64 webgpu firefox: untested; macos-arm64 webgpu safari: untested; windows-x64 webgpu chromium: untested |
<!-- /catalog:maturity -->

**Experimental.** WebGPU stays local evidence (no CI job). Through this package, both e5 WebGPU profiles pass every bound of their protected tolerances, G1 and G2 included, on one NVIDIA RTX 4060 Ti in headless Chromium 153 (Chrome for Testing) on Linux: fp32 within 2.05e-7 of the reference, int8-embed 3.51e-3 with minimum cosine 0.99980 and 1.7e-7 from native ONNX Runtime on the same graph ([evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/browser-e5-package-2026-10-23/README.md)). fp16 graphs need the adapter feature `shader-f16`, which that setup does not expose. Other GPUs, vendors, browsers and operating systems are untested.

## Use

```csharp
using Starlings.Runtime.OnnxWeb;
using Starlings.Runtime.OnnxWeb.WebGpu;

await OnnxWebRuntime.InitializeAsync(new OnnxWebRuntimeOptions { Bundle = OnnxWebGpuBundle.WebGpu });
await using var session = await OnnxWebSession.CreateAsync(model, new OnnxWebSessionOptions { Provider = OnnxWebExecutionProvider.WebGpu });
```

A page loads one bundle. This one also runs the WASM provider, so an app that offers both providers initializes with it. When `navigator.gpu` is missing, `CreateAsync` throws `NotSupportedException` and never falls back to WASM on its own. The first run compiles shaders (0.3 to 0.7 s); after that, a call costs about 17 ms whatever the batch size, which wins on batches and long texts.
