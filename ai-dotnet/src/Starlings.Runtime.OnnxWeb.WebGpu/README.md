# Starlings.Runtime.OnnxWeb.WebGpu

The voluntary WebGPU adapter of `Starlings.Runtime.OnnxWeb`. It ships onnxruntime-web 1.30.0's WebGPU-capable bundle (`ort.webgpu.min.mjs` and its asyncify binary, 26.8 MB, about 4 MB brotli) as static web assets, and `OnnxWebGpuBundle.WebGpu` to initialize with. No core package references it; reference it only if you want WebGPU.

## Maturity

**Experimental.** Its WebGPU evidence is the 2026-10-11 browser spike's (the same onnxruntime-web 1.30.0 bundle on one NVIDIA RTX 4060 Ti, headless Chromium 153, Linux): e5 fp32 within 2.1e-7 of the reference. A rerun through the packages could not get GPU memory on that machine (`docs/evidence/browser-e5-package-2026-10-23/`), so WebGPU through this package is not yet re-verified. fp16 graphs need the adapter feature `shader-f16`, which that setup does not expose. Other GPUs and browsers are untested.

## Use

```csharp
using Starlings.Runtime.OnnxWeb;
using Starlings.Runtime.OnnxWeb.WebGpu;

await OnnxWebRuntime.InitializeAsync(new OnnxWebRuntimeOptions { Bundle = OnnxWebGpuBundle.WebGpu });
await using var session = await OnnxWebSession.CreateAsync(model, new OnnxWebSessionOptions { Provider = OnnxWebExecutionProvider.WebGpu });
```

A page loads one bundle. This one also runs the WASM provider, so an app that offers both providers initializes with it. When `navigator.gpu` is missing, `CreateAsync` throws `NotSupportedException` and never falls back to WASM on its own. The first run compiles shaders (0.3 to 0.7 s); after that, a call costs about 17 ms whatever the batch size, which wins on batches and long texts.
