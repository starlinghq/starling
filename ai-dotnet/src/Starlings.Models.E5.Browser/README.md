# Starlings.Models.E5.Browser

`intfloat/multilingual-e5-small` (revision `614241f6`) sentence embeddings **in the browser**: Blazor WebAssembly or any other .NET `browser-wasm` app, on [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) through `Starlings.Runtime.OnnxWeb`. Text is tokenized by the owned tokenizer (`Starlings.Tokenizers`), and padding, mean pooling and L2 normalization are the same code as `Starlings.Models.E5`. Only the graph runs on a different backend. The API is asynchronous, because onnxruntime-web is, and may still change.

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

**Experimental.** WebAssembly has no support badge (plan 13.1). The table above is generated from the four browser profiles' registry manifests; the catalog has their gates and per-profile capabilities. G1 and G2 pass for all four with the protected evaluator on the development fixtures ([gate reports](https://github.com/starlinghq/starling/blob/main/docs/evidence/browser-e5-package-2026-10-23/gates/README.md)); the other gates have not been run. The evidence is in [docs/evidence/browser-e5-package-2026-10-23](https://github.com/starlinghq/starling/blob/main/docs/evidence/browser-e5-package-2026-10-23/README.md). The table below summarizes the evidence per feature; where it names a profile, the status is the manifest's.

| Capability | Status | Evidence |
| --- | --- | --- |
| `onnx-fp32-wasm`: fp32 graph, WASM backend | experimental | every bound of its tolerance file passes (max abs 1.4e-7), AOT and interpreter, Chrome for Testing 153 on Linux x64 |
| `onnx-int8embed-wasm`: int8-embed graph, WASM backend | experimental | every bound of G1 and G2 passes (max abs 3.5e-3, min cosine 0.99980, 1.3e-7 vs native ONNX Runtime on the same graph); `task_metric_before_preview` (retrieval top-10 overlap, needed for Preview) not evaluated yet |
| `onnx-fp32-webgpu`, `onnx-int8embed-webgpu` (with `Starlings.Runtime.OnnxWeb.WebGpu`) | experimental | every bound of G1 and G2 passes through these packages on one NVIDIA RTX 4060 Ti, headless Chromium 153 (Chrome for Testing), Linux: fp32 max abs 2.05e-7; int8-embed 3.5e-3, min cosine 0.99980, 1.7e-7 vs native; other GPUs and browsers untested |
| Owned tokenizer in the browser (ids, masks, offsets) | experimental | 0 mismatches on the protected fixtures, interpreter and AOT |
| `IEmbeddingGenerator<string, Embedding<float>>` (`E5BrowserEmbeddingGenerator`) | experimental | same request rules as `Starlings.Models.E5.Extensions.AI`; unit tests and the browser run |
| Cache Storage reuse across page loads | experimental for `onnx-int8embed-wasm`; untested for the other profiles | reload check in the evidence (int8-embed, WASM) |
| Cancellation | experimental: checked before each chunk and run; a run already inside onnxruntime-web completes first | unit tests |
| Micro-batching, fp16, dynamic int8 | unsupported | the spike rejected dynamic int8 (batch-dependent); fp16 fails on WebGPU without `shader-f16` |
| Firefox, Safari, Windows or macOS browsers, mobile, non-NVIDIA GPUs | untested | |

## Install

```sh
dotnet add package Starlings.Models.E5.Browser
dotnet add package Starlings.Runtime.OnnxWeb.WebGpu      # optional: WebGPU
```

Mark the app browser-only, because the browser APIs are `[SupportedOSPlatform("browser")]`:

```csharp
[assembly: System.Runtime.Versioning.SupportedOSPlatform("browser")]
```

## Use

```csharp
using Starlings.Models.E5.Browser;
using Starlings.Models.E5.Extensions.AI;

// Fetches tokenizer.json and onnx/model.onnx from the Hub at the pinned revision (CORS enabled), checks both SHA-256s,
// keeps them in Cache Storage, and creates the onnxruntime-web session.
using var embedder = await E5BrowserEmbedder.LoadAsync(progress: new Progress<E5BrowserLoadProgress>(p => Console.WriteLine(p)));

float[] query = await embedder.EmbedAsync(E5TextEmbedder.QueryPrefix + "how do I keep basil fresh?");

// Or through Microsoft.Extensions.AI, with the e5 prefix as an input type:
using var generator = new E5BrowserEmbeddingGenerator(embedder, E5InputType.Passage);
var embeddings = await generator.GenerateAsync(["Trim the stems and stand basil in water.", "Le Rhin se jette dans la mer du Nord."]);
```

The int8-embed graph (182 MB, about 109 MB brotli) has no public URL. Build it with `model-pipeline/reference/encoder/quantize_e5_browser.py`, which is deterministic and produces SHA-256 `374a3cc0…fb19`. Host it with the app, then load it:

```csharp
await E5BrowserEmbedder.LoadAsync(new E5BrowserLoadOptions { Variant = E5BrowserVariant.Int8Embed, ModelUrl = "models/model-int8-embed.onnx" });
```

The bytes must match the variant's pinned SHA-256 wherever they come from. `E5BrowserAssets` holds the pins.

## Hosting

- **Threads need cross-origin isolation.** Serve the app with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`). Without them onnxruntime-web runs one thread, which was 3.2x to 3.7x slower in the evidence run. Asking for more than one thread then fails with `NotSupportedException` rather than silently running one. Cross-origin model URLs must send CORS headers (the Hub does).
- **SHA-256 verification needs a secure context.** It uses the browser's SubtleCrypto, which is available only on https or localhost.
- **Memory.** The renderer peaked at about 1.4 GB with int8-embed and 2.3 GB with fp32. The graph bytes never enter the .NET heap: they go from `fetch` to Cache Storage to onnxruntime-web.
- **Build mode.** Publish with `-p:RunAOTCompilation=true` (the `wasm-tools` workload). The tokenizer is 3x to 4x faster under AOT; inference speed is onnxruntime-web's either way.
- **Download.** onnxruntime-web's WASM bundle is about 2.3 MB brotli on top of the .NET app. The tokenizer is 2.5 MB brotli (the Hub serves it uncompressed, 17 MB).

## Not included

There's no micro-batching, and no other e5 variants than fp32 and int8-embed. Other models (EmbeddingGemma 2, GLiNER2, Qwen) are out of scope until e5 is qualified in the browser (follow-up plan W6).
