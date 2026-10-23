# Starlings.Runtime.OnnxWeb

[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) 1.30.0 for Starlings models in .NET `browser-wasm` apps (Blazor WebAssembly). It's the browser counterpart of `Starlings.Runtime.Onnx`: model download with SHA-256 verification and Cache Storage, session creation with no silent provider fallback, and asynchronous runs from C# through `[JSImport]`. Model bytes stay in JavaScript and never enter the .NET heap. The package ships onnxruntime-web's WASM-only bundle as static web assets under `_content/Starlings.Runtime.OnnxWeb/ort/` (14.2 MB, about 2.3 MB brotli). WebGPU is the separate, opt-in `Starlings.Runtime.OnnxWeb.WebGpu`. The API may still change.

## Maturity

**Experimental.** WebAssembly has no support badge (plan 13.1). Model profiles that use this runtime are listed in `Starlings.Models.E5.Browser`. Tested in Chrome for Testing 153 on Linux x64; other browsers are untested.

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

## Build

The onnxruntime-web files are not in Git. The build runs `npm ci` in `ai-dotnet/src/Shared/OnnxWeb` (integrity-checked against `package-lock.json`) and copies the three files into `wwwroot/ort/`, which Git ignores. Node.js and npm are needed to build or pack the package; consumers need neither. Notices: `ai-dotnet/THIRD-PARTY-NOTICES.md` (onnxruntime-web, MIT).
