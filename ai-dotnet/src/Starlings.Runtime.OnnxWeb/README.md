# Starlings.Runtime.OnnxWeb

[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) 1.30.0 for Starlings models in .NET `browser-wasm` apps (Blazor WebAssembly). It's the browser counterpart of `Starlings.Runtime.Onnx`: model download with SHA-256 verification and Cache Storage, session creation with no silent provider fallback, and asynchronous runs from C# through `[JSImport]`. Model bytes stay in JavaScript and never enter the .NET heap. The package ships onnxruntime-web's WASM-only bundle as static web assets under `_content/Starlings.Runtime.OnnxWeb/ort/` (14.2 MB, about 2.3 MB brotli). WebGPU is the separate, opt-in `Starlings.Runtime.OnnxWeb.WebGpu`. The API may still change.

## Maturity

<!-- catalog:maturity profiles=multilingual-e5-small/onnx-*-wasm,multilingual-e5-small/onnx-*-webgpu -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-wasm` | experimental | linux-x64 wasm chromium-153: experimental; linux-x64 wasm firefox: untested; macos-arm64 wasm safari: untested; windows-x64 wasm chromium: untested |
| `multilingual-e5-small/onnx-fp32-webgpu` | experimental | linux-x64 webgpu chromium-153: experimental; linux-x64 webgpu firefox: untested; macos-arm64 webgpu safari: untested; windows-x64 webgpu chromium: untested |
| `multilingual-e5-small/onnx-int8embed-wasm` | experimental | linux-x64 wasm chromium-153: experimental; linux-x64 wasm firefox: untested; macos-arm64 wasm safari: untested; windows-x64 wasm chromium: untested |
| `multilingual-e5-small/onnx-int8embed-webgpu` | experimental | linux-x64 webgpu chromium-153: experimental; linux-x64 webgpu firefox: untested; macos-arm64 webgpu safari: untested; windows-x64 webgpu chromium: untested |
<!-- /catalog:maturity -->

**Experimental.** WebAssembly has no support badge (plan 13.1). The model profiles that use this runtime are the e5 browser profiles above (the WebGPU ones through `Starlings.Runtime.OnnxWeb.WebGpu`); their capabilities are in `Starlings.Models.E5.Browser`. G1 and G2 pass in Chrome for Testing 153 on Linux x64 (`docs/evidence/browser-e5-package-2026-10-23/gates/`); other browsers and operating systems are untested.

## Use

```csharp
[assembly: System.Runtime.Versioning.SupportedOSPlatform("browser")]

var info = await OnnxWebRuntime.InitializeAsync();              // once per page; NumThreads, Proxy, Bundle are options
using var model = await OnnxWebAssets.FetchAsync(url, expectedSha256, progress: progress);
await using var session = await OnnxWebSession.CreateAsync(model); // consumes the bytes
var hidden = new float[rows * width * 384];
int[] shape = await session.RunAsync(
    [OnnxWebInput.FromInt64("input_ids", ids, rows, width), OnnxWebInput.FromInt64("attention_mask", mask, rows, width)],
    "last_hidden_state", hidden);
```

- **Inputs and outputs.** Inputs are int64 or float32, and outputs are copied as float32 into a destination of exactly the output's size. Runs on one session are serialized.
- **Cancellation.** It's checked before a run starts; onnxruntime-web cannot interrupt one in progress.
- **Threads.** More than one needs a cross-origin isolated page (COOP `same-origin`, COEP `require-corp` or `credentialless`); `OnnxWebRuntimeInfo` reports what's in effect.
- **Proxy worker.** `Proxy = true` runs the WASM backend in onnxruntime-web's worker, so the page never blocks during inference. It's experimental and WASM-only.
- **Threading model.** It supports the default single-threaded .NET browser runtime, not `WasmEnableThreads`.
- **Hosts.** It runs on a page (Blazor WebAssembly, or any .NET `browser-wasm` app) and in a Web Worker. In a worker, URLs resolve against the worker script's location, so place the worker next to `_framework/` and `_content/`. `ai-dotnet/samples/E5BrowserJs` hosts it in a worker behind a JavaScript API.

## Build

The onnxruntime-web files are not in Git. The build runs `npm ci` in `ai-dotnet/src/Shared/OnnxWeb` (integrity-checked against `package-lock.json`) and copies the three files into `wwwroot/ort/`, which Git ignores. Node.js and npm are needed to build or pack the package; consumers need neither. Notices: `ai-dotnet/THIRD-PARTY-NOTICES.md` (onnxruntime-web, MIT).
