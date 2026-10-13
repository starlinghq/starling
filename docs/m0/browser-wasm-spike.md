# Browser (WebAssembly) feasibility spike

Status: spike result, 2026-10-11. Brief: maintainer approval (Mark, 2026-10-11) for a browser feasibility spike. Plan 13.1: WebAssembly needs its own dependency and hardware evidence before any support badge; this document is that first evidence and grants no badge. Plan 10.1: quantized models are separate variants with a task-specific quality budget; the budgets below are proposed values only, for the maintainer to set.

Raw results: [docs/evidence/browser-wasm-spike/](../evidence/browser-wasm-spike/). Code: `browser/spike/` (app, native comparison, harness), `model-pipeline/reference/encoder/quantize_e5_browser.py` and `drift_e5_browser.py`.

Pins: .NET SDK 10.0.100 with the `wasm-tools` workload 10.0.12 (browser-wasm runtime pack 10.0.12, Emscripten 3.1.56), `Microsoft.AspNetCore.Components.WebAssembly` 10.0.12; onnxruntime-web 1.30.0 (npm, `browser/spike/package-lock.json`, integrity `sha512-q0y+Jrrt…KYQ==`); Python ONNX Runtime 1.30.0, onnx 1.23.1, Playwright 1.63.0 (`model-pipeline/reference/encoder/requirements-browser.txt`, hashed); Chrome for Testing 153.0.8010.12 (the build Playwright 1.63.0 pins; its downloader timed out, so the same zip was fetched with curl, SHA-256 `8aac3501…fd5a`). e5: `intfloat/multilingual-e5-small@614241f6`, `onnx/model.onnx` SHA-256 `ca456c06…8665`.

Machine: i7-5820K (6 cores, 12 threads, AVX2), 94 GB RAM, Linux x64, NVIDIA driver 610.57.04; GPU 0 RTX 3090 (not used, see section 5), GPU 1 RTX 4060 Ti (shared with other jobs). The timing tables come from one sequential run (`browser/spike/harness/run_all.sh`) at load average 1.3 to 3.0; an earlier run at load 15 had the same exactness and drift but 2 to 4 times slower timings. Exploratory, not a qualified benchmark.

## 1. Summary

| Question | Answer |
| --- | --- |
| Does the owned tokenizer run in the browser? | Yes. A Blazor WebAssembly app with a project reference to `Starlings.Tokenizers` runs all protected fixture cases of three profiles in headless Chrome: e5 Unigram (49 cases, 567 field checks), Qwen3.5 chat byte-level BPE (137 cases, 1,374 checks), EmbeddingGemma 2 Gemma BPE (49 cases, 505 checks). **0 mismatches** on ids, tokens, type ids, masks, sequence ids, offsets (UTF-16, scalar, UTF-8), decode both ways, truncation, padded batch, and the e5 embedding input ids, in both the interpreter and the AOT build. No code change in the library. |
| Trimming | 0 trim warnings (ILLink ran; Blazor hides trim warnings by default, the spike turns them on). |
| Tokenizer speed in the browser | AOT: load 1.4x to 1.8x native (e5 402 ms vs 255 ms), encode 1.9x to 3.0x slower than native. Interpreter: load 3.9x to 6.4x, encode 8x to 13x slower. Section 2. |
| App download | Startup (brotli, on the wire): 2.36 MB interpreter, 4.50 MB AOT. Plus onnxruntime-web 3.9 MB (WebGPU-capable binary) or 2.3 MB (WASM-only binary), tokenizer.json 2.5 to 3.5 MB, and the graph. |
| Inference route | onnxruntime-web 1.30.0 called from C# through `IJSRuntime` (a 150-line ES module). It works on the WASM backend (6 threads with cross-origin isolation) and on WebGPU. Linking native ONNX Runtime into the .NET wasm is not practical now (section 4). |
| Can the browser load the 470 MB fp32 e5 graph? | Yes: session creation 1.5 s (WASM) to 2.0 s (WebGPU) after the fetch, renderer peak RSS 2.4 GB (WASM) and 3.2 GB (WebGPU). It matches the reference to **1.4e-7** (WASM) and **2.1e-7** (WebGPU), inside the binding CPU tolerance of 1e-5. |
| Quantized variants | fp16 (235 MB): max abs 1.2e-4, min cosine 0.9999997. int8 dynamic (118 MB): max abs 0.046, min cosine **0.963**, and padded batch differs from single rows by 0.03 to 0.04 (activation scales depend on the batch). int8 embeddings only (182 MB, MatMuls fp32): max abs 3.5e-3, min cosine **0.9998**, batch-independent. Browser drift equals native drift except for dynamic int8 (different kernels). Section 3. |
| Latency (warm, 34-token passage, batch 1 / short batch of 8 / one 512-token text) | Native Python ORT CPU 6 threads, fp32: 8.5 / 42 / 136 ms. Browser WASM 6 threads, fp32: 27 / 153 / 309 ms (2.3x to 3.7x native). Browser WebGPU (RTX 4060 Ti), fp32: 17 / 17 / 16 ms. |
| Headless automation | Playwright (Python, pinned) with Chrome for Testing, a local static server with COOP/COEP, a fresh temporary profile. WebGPU in headless Chromium needs `--enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=native --disable-vulkan-surface`; without the last two the adapter is SwiftShader (CPU). |
| GPU pinning | `VK_LOADER_DEVICE_ID_FILTER=0x2805` in Chromium's environment pins WebGPU to the RTX 4060 Ti: the adapter reports `lovelace`; without the filter it reports `ampere` (RTX 3090). `nvidia-smi` during the runs shows Chrome's GPU process on GPU 1 only and no change on GPU 0. Section 5. |
| Other catalog models | EmbeddingGemma 2 text fp32 (1.08 GB external data) and GLiNER2 fp32 (829 MB) create sessions in WASM (inference not run). Qwen3.5-0.8B: the int4 graph fails (`GatherBlockQuantized` has no WASM kernel in 1.30.0) and the 3.08 GB fp32 file cannot even be fetched into one buffer. Section 6. |
| Estimate | 15 to 24 working days for an e5-only browser preview (section 7). |

## 2. Tokenizer in the browser

### 2.1 App and location

`browser/spike/BrowserSpike/` is a standalone Blazor WebAssembly app (`net10.0`) with a project reference to `tokenizers-dotnet/src/Starlings.Tokenizers`. The conformance and timing code (`browser/spike/Shared/TokenizerSuite.cs`) is compiled into both the app and a console app (`browser/spike/BrowserSpike.Native/`), so the browser and native numbers come from identical source. The harness (`browser/spike/harness/run_spike.py`) calls the app's `[JSInvokable]` methods with `DotNet.invokeMethodAsync`. The fixtures (`model-pipeline/protected/`), the pinned `tokenizer.json` assets and the graphs are served by the harness from outside the published app (nothing large in Git, nothing protected copied).

Why a new top-level `browser/spike/` and not `ai-dotnet/samples/` or a solution: the samples are package consumers that the release drill copies and builds outside the checkout, and this app needs a project reference; the AOT build needs the `wasm-tools` workload and the inference path needs npm, neither of which the CI runners install; keeping it out of `Tokenizers.slnx` and `AI.slnx` leaves CI unchanged. It still imports the repository `Directory.Build.props` (central versions, locked restore, warnings as errors) and has its own `Directory.Packages.props` and lock files. It is not packable and is not part of any shipped package. CI does not build it; how to build and run it is in section 9.

### 2.2 Exactness

The suite checks every field the fixtures record (same rules as the xunit conformance tests): ids, tokens, type ids, attention and special-token masks, sequence ids where recorded, offsets in UTF-16 code units and Unicode scalars (and UTF-8 bytes for Gemma, derived from the scalar offsets), decode with and without special tokens, the truncated encodings (e5 `long-english`, `long-mixed`), the padded batches (e5 and Gemma `tokenizer-batch.json`), and for e5 the 46 `embeddings.jsonl` input ids from a 512-truncated encode. The asset SHA-256 is verified in the browser before loading.

| Profile | Cases | Field checks | Native | Browser, interpreter | Browser, AOT |
| --- | ---: | ---: | --- | --- | --- |
| xlmr-unigram-e5 | 49 (+46 embedding inputs) | 567 | 0 mismatches | 0 mismatches | 0 mismatches |
| qwen35-bpe-chat | 137 | 1,374 | 0 | 0 | 0 |
| gemma-bpe-embeddinggemma2 | 49 | 505 | 0 | 0 | 0 |

The Qwen regex scanner, the Unicode tables (generated, not from the .NET runtime's ICU), NFC, the Precompiled normalizer and byte fallback all behave identically under Mono on WebAssembly. The browser-wasm runtime loads only one ICU slice (`icudt_*.dat`); the tokenizer does not depend on it.

### 2.3 Load and encode speed

Same code, same assets. Load = `Tokenizer.FromUtf8Json` on bytes already in memory (first call, then median of 3 warm reloads). Encode = every single-text fixture case of the profile, repeated for 3 s after a warm-up pass. Native: CoreCLR 10.0.12, Release.

| Profile | Measure | Native | Browser AOT | Browser interpreter |
| --- | --- | ---: | ---: | ---: |
| e5 (17 MB, 250k pieces) | first load, ms | 255 | 402 (1.6x) | 1,630 (6.4x) |
| | warm reload, ms | 181 | 270 (1.5x) | 1,411 (7.8x) |
| | encode, texts/s (µs/text) | 10,003 (100) | 3,370 (297), 3.0x slower | 761 (1,314), 13x slower |
| Qwen chat (20 MB, 248k tokens) | first load, ms | 131 | 233 (1.8x) | 610 (4.7x) |
| | warm reload, ms | 104 | 153 | 460 |
| | encode, texts/s (µs/text) | 28,349 (35) | 10,576 (95), 2.7x | 2,375 (421), 12x |
| Gemma (32 MB, 262k tokens, 515k merges) | first load, ms | 207 | 289 (1.4x) | 821 (4.0x) |
| | warm reload, ms | 210 | 296 | 808 |
| | encode, texts/s (µs/text) | 14,760 (68) | 7,962 (126), 1.9x | 1,822 (549), 8.1x |

Getting the bytes into .NET in the browser (`HttpClient.GetByteArrayAsync`, uncompressed from localhost) took 650 to 1,030 ms, against 7 ms from the page cache natively; a real deployment pays the network instead, with 2.5 to 3.5 MB brotli per file (section 2.4). The e5 end-to-end path in section 3 loads the tokenizer in about 0.4 s (AOT).

Recommendation: ship AOT for any tokenizer-heavy page. The interpreter is exact but 8x to 13x slower to encode and 4x to 8x slower to load.

### 2.4 Download size

Measured on the wire by the harness's server, which serves the publish output's precompressed `.br` files (as a static host would) up to the moment the app has rendered:

| Build | Startup on the wire (brotli) | Uncompressed | Startup to rendered | Renderer peak RSS (three tokenizers loaded) |
| --- | ---: | ---: | ---: | ---: |
| Interpreter (`dotnet publish -c Release`) | 2.36 MB (44 files) | 7.4 MB | 0.60 s | 0.68 GB |
| AOT (`-p:RunAOTCompilation=true`) | 4.50 MB | 19.2 MB | 0.71 s | 0.85 GB |

The tokenizer assembly is 206 KB (57 to 71 KB brotli). Per-model assets on top (brotli q11): e5 `tokenizer.json` 2.46 MB, Qwen chat 2.69 MB, Gemma 3.50 MB. The only 404 in every run is `/favicon.ico`.

## 3. Inference in the browser

### 3.1 Setup

`browser/spike/BrowserSpike/wwwroot/spike-ort.js` imports `ort.webgpu.min.mjs` (onnxruntime-web 1.30.0, served by the harness under `/ort/`), fetches the graph, creates the session (`executionProviders: [backend]`, `graphOptimizationLevel: 'all'`, `numThreads` 6) and runs it. C# (`SpikeEntryPoints.RunE5`) packs int64 ids and masks into `byte[]`, calls `run` through `IJSRuntime` (Blazor passes `byte[]` as `Uint8Array` both ways), and pools the returned `last_hidden_state` with the library's algorithm (mean over the mask in double, L2 normalization). Inputs are the protected `embeddings.jsonl` ids, so tokenization is out of the drift numbers; the end-to-end latency rows use the owned tokenizer in the browser.

Variants (`model-pipeline/reference/encoder/quantize_e5_browser.py`, ONNX Runtime 1.30.0's own tools; hashes and node counts in `docs/evidence/browser-wasm-spike/variants.json`):

| Variant | How | Size | Brotli q9 | SHA-256 |
| --- | --- | ---: | ---: | --- |
| fp32 | the snapshot's `onnx/model.onnx` | 470.3 MB | 249.3 MB | `ca456c06…8665` |
| fp16 | `convert_float_to_float16(keep_io_types=True)` | 235.3 MB | 216.1 MB | `f49ade3f…833c` |
| int8-dynamic | `quantize_dynamic(QInt8, per_channel=False)` (MatMul and Gather) | 118.1 MB | 77.0 MB | `8da4c9ba…d8b6` |
| int8-dynamic-pc | the same, `per_channel=True` | 118.3 MB | not measured | `739c8f25…d717` |
| int8-embed | `quantize_dynamic(QInt8, op_types_to_quantize=["Gather"])`: embedding tables int8 (weight-only), every MatMul fp32 | 181.7 MB | 109.3 MB | `374a3cc0…fb19` |

The word-embedding table is 96 M of e5-small's 118 M parameters, which is why weight-only int8 of the embeddings keeps most of the size reduction. All variants are deterministic (re-running the script reproduced the hashes).

### 3.2 Drift

Against the protected fp32 PyTorch embeddings (46 cases x 384 values). "vs native" compares the browser run to Python ONNX Runtime CPU on the same graph, which separates backend drift from quantization drift. "b8" is the padded batch of the first 8 short cases against the same backend's single rows.

| Variant | Backend | max abs | mean abs | min cosine (case) | vs native max abs | b8 vs single |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| fp32 | native ORT CPU | 2.3e-7 | 2.0e-8 | 1.0000000 | | 0 |
| fp32 | WASM | 1.4e-7 | | 1.0000000 | 2.0e-7 | 1.0e-7 |
| fp32 | WebGPU | 2.1e-7 | | 1.0000000 | 2.1e-7 | 1.5e-7 |
| fp16 | native ORT CPU | 1.25e-4 | 1.2e-5 | 0.9999997 (ascii-sentence) | | 0 |
| fp16 | WASM | 1.16e-4 | | 0.9999997 | 4.4e-5 | 3.5e-5 |
| fp16 | WebGPU | fails: `Gather requires f16 but the device does not support it` (section 5.3) | | | | |
| int8-dynamic | native ORT CPU | 0.046 | 6.1e-3 | 0.9635 (repeated-char) | | 0.041 |
| int8-dynamic | WASM | 0.046 | | 0.9624 | 0.017 | 0.027 |
| int8-dynamic | WebGPU | 0.051 | | 0.9634 | 0.017 | 0.030 |
| int8-dynamic-pc | native ORT CPU | 0.027 | 3.7e-3 | 0.9854 (special-literal-mask) | | 0.027 |
| int8-dynamic-pc | WASM | 0.026 | | 0.9888 | 0.026 | 0.030 |
| int8-dynamic-pc | WebGPU | 0.023 | | 0.9886 | 0.026 | 0.030 |
| int8-embed | native ORT CPU | 3.5e-3 | 3.8e-4 | 0.99980 (repeated-char) | | 0 |
| int8-embed | WASM | 3.5e-3 | | 0.99980 | 1.3e-7 | 1.0e-7 |
| int8-embed | WebGPU | 3.5e-3 | | 0.99980 | 1.8e-7 | 1.6e-7 |

Findings:

- fp32 in the browser is numerically the reference profile: both backends are within the binding `onnx-fp32-cpu` tolerance (embedding max abs 1e-5) by a factor of 50.
- Quantization drift dominates and is a property of the graph: for fp16 and int8-embed the browser equals native ORT to 4e-5 and 2e-7.
- Dynamic int8 is the exception: its MatMulInteger and DynamicQuantizeLinear kernels differ between native x86 and WebAssembly (0.017 to 0.026 between them), and because activation scales are computed per tensor over the whole batch, a row's embedding depends on what else is in the batch and on padding (0.027 to 0.041). That breaks batch independence, which plan 10.1 asks to test and which the e5 profile bounds (`padded_batch_vs_single_embedding_max_abs: 1.0e-5`).
- On WebGPU the dynamic int8 graphs run, but ORT places the integer ops on the CPU (placement warning in the console) and they are 10 to 30 times slower than fp32 on the GPU.

### 3.3 Load, latency and memory

Warm medians (WASM 20 samples, WebGPU 10, after 3 warm-up runs; raw samples in `browser-e5-runs.json`). JS = `session.run` only; e2e = from C#: owned tokenizer, interop, run, pooling. Load = session creation after the fetch (localhost fetch time listed separately; it says nothing about a network). RSS = peak resident memory of the renderer process; "page" = `performance.measureUserAgentSpecificMemory()` after the run.

| Variant | Backend | Fetch / create, ms | First run, ms | passage b1 JS (e2e), ms | short b8 JS (e2e), ms | long512 b1, ms | Renderer RSS / page, GB |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| fp32 | native ORT CPU (6 threads) | – / 486 | | 8.5 | 42.0 | 136 | |
| fp32 | WASM (6 threads) | 412 / 1,457 | 84 | 27.2 (20.6) | 153 (172) | 309 | 2.44 / 1.45 |
| fp32 | WASM, interpreter app | 437 / 1,446 | 86 | 19.5 (30.4) | 150 (162) | 314 | 2.40 / 1.45 |
| fp32 | WebGPU | 513 / 2,039 | 698 | 17.0 (17.8) | 16.7 (20.3) | 16.1 | 3.21 / 1.83 |
| fp16 | native ORT CPU | – / 422 | | 9.1 | 43.9 | 109 | |
| fp16 | WASM | 270 / 1,633 | 76 | 31.4 (34.7) | 160 (160) | 307 | 1.50 / 0.79 |
| int8-dynamic | native ORT CPU | – / 167 | | 6.3 | 35.9 | 91 | |
| int8-dynamic | WASM | 176 / 1,234 | 109 | 20.2 (23.8) | 98 (98) | 220 | 1.11 / 0.46 |
| int8-dynamic | WebGPU | 161 / 1,679 | 399 | 268 (262) | 500 (498) | 762 | 1.32 / 0.55 |
| int8-dynamic-pc | WASM | 151 / 1,249 | 100 | 23.3 (20.8) | 92 (107) | 216 | 1.10 / 0.46 |
| int8-embed | native ORT CPU | – / 210 | | 7.2 | 39.5 | 90 | |
| int8-embed | WASM | 210 / 1,299 | 102 | 29.2 (27.4) | 149 (162) | 295 | 1.30 / 0.59 |
| int8-embed | WASM, 1 thread, no COOP/COEP | 216 / 1,190 | 82 | 79.4 (82.7) | 791 (780) | 1,493 | 1.26 / – |
| int8-embed | WebGPU | 200 / 1,652 | 320 | 17.7 (16.6) | 16.8 (18.5) | 17.6 | 1.47 / 0.59 |

Readings:

- WASM with 6 threads is 2.3x to 3.7x slower than native ONNX Runtime with 6 threads on the same CPU. Without cross-origin isolation onnxruntime-web runs one thread and is another 2.7x to 5x slower; hosting with COOP/COEP headers is therefore part of the deployment contract.
- WebGPU has a floor of about 17 ms per call on this setup (batch 1 or 8, 34 or 512 tokens all cost the same), so it wins 9x on batches and long texts and 1.6x on short single texts. First run 0.3 to 0.7 s (shader compilation).
- The C# side costs little: e2e and JS-only medians are within run-to-run noise, and the interpreter app runs inference as fast as the AOT app (the work is inside onnxruntime-web). AOT matters for the tokenizer only.
- Memory: the fp32 graph costs 2.4 GB of renderer RSS on WASM (the model bytes exist in the JS heap and in the wasm heap during creation) and 3.2 GB with WebGPU. int8-embed halves that. wasm32 caps the wasm heap at 4 GB.

### 3.4 Proposed quality budgets (values only, for the maintainer)

Proposed as separate variant profiles (plan 10.1); nothing here edits `model-pipeline/protected/` or any `tolerances.*` file. Observed worst values in brackets.

| Profile (proposed name) | embedding max abs | min cosine | padded batch vs single | Other |
| --- | --- | --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-wasm` and `-webgpu` | 1e-5 (2.1e-7) | report only (1.0000000) | 1e-5 (1.5e-7) | same discrete rules as `onnx-fp32-cpu` (token ids, masks, offsets exact) |
| `multilingual-e5-small/onnx-fp16-wasm` | 5e-4 (1.25e-4) | 0.99999 (0.9999997) | 1e-4 (3.5e-5) | browser vs native same graph 1e-4 (4.4e-5) |
| `multilingual-e5-small/onnx-int8embed-wasm` and `-webgpu` | 1e-2 (3.5e-3) | 0.9995 (0.99980) | 1e-5 (1.6e-7) | browser vs native same graph 1e-5 (1.8e-7); a task metric before any preview (retrieval top-10 overlap with fp32 at least 0.95 on a held-out query set; not measured) |
| int8 dynamic (either) | not proposed | | | fails batch independence (0.027 to 0.041) and drifts to cosine 0.963 / 0.985; also backend-dependent |

Recommended browser default for e5: **int8-embed** (109 MB brotli, cosine 0.9998, batch-independent, identical in browser and native, works on both backends). fp16 is the closer variant (cosine 0.9999997) but is 216 MB brotli (fp16 weights barely compress) and does not run on WebGPU here. fp32 is viable on desktop browsers that accept a 249 MB brotli download and 2.4 GB of memory.

## 4. Inference route

| Route | Assessment |
| --- | --- |
| **onnxruntime-web via JS interop** (tested) | Works today on both backends with the same graphs and the same numerics as native ORT (except dynamic int8). Pinned npm package (MIT), about 150 lines of JS glue, `byte[]` interop, no change to the .NET build. The browser downloads onnxruntime-web separately (3.9 MB brotli WebGPU-capable, 2.3 MB WASM-only). Threads need COOP/COEP. Recommended. |
| Native ONNX Runtime compiled to wasm and linked into `dotnet.native.wasm` (`NativeFileReference`), driven by `Microsoft.ML.OnnxRuntime.Managed` P/Invoke | Not tested. No browser-wasm native asset is published for Microsoft.ML.OnnxRuntime; ORT would have to be built as a static library with exactly the Emscripten version .NET 10 uses (3.1.56), which need not match the emsdk ORT's own web build is tested with; threads would need the experimental .NET wasm threading (`WasmEnableThreads`), and WebGPU would need ORT's JS bridge anyway. It would make the existing C# session code reusable, but at a high build and maintenance cost for each ORT and .NET update. Not recommended now. |
| Managed inference (C# kernels) | Not viable for this plan: no BLAS-class kernels in WebAssembly .NET, and a second numeric implementation to qualify. |
| transformers.js | Uses onnxruntime-web underneath and its own tokenizer; it would bypass the owned tokenizer and the C# contracts. |
| WebNN | Not tested; backend availability in browsers is still limited. |

A browser package would keep the C# contract (tokenizer, pooling, options, `IEmbeddingGenerator`) and replace only the session: today `E5Encoder` calls `Microsoft.ML.OnnxRuntime` directly, so the encoder would need a small session abstraction with an onnxruntime-web implementation over `IJSRuntime` or `[JSImport]`.

## 5. Headless automation and GPU pinning

### 5.1 Harness

`browser/spike/harness/serve.py` serves the published `wwwroot` (precompressed `.br`/`.gz` when accepted), onnxruntime-web, the fixtures, the assets and the graphs, with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` (so `crossOriginIsolated` is true and onnxruntime-web gets threads; `--no-isolation` drops them). `run_spike.py` starts it, launches Chromium through Playwright with `launch_persistent_context` on a fresh directory under `/tmp` (identifiable process tree for the RSS sampler; never the user's profile), waits for the app to render, calls the entry point and writes JSON with the result, the bytes served per path, console messages and peak RSS. cua-driver and the user's browser were not used.

### 5.2 WebGPU in headless Chromium on Linux

With `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan` alone, `navigator.gpu.requestAdapter()` returned the SwiftShader adapter (vendor `google`, architecture `swiftshader`), and an e5 run on it did not finish in 10 minutes. Adding `--use-vulkan=native --disable-vulkan-surface` gives the NVIDIA adapter. These are now the harness defaults for `--backend webgpu`.

### 5.3 Pinning to GPU 1

`CUDA_VISIBLE_DEVICES` does not apply to Vulkan. The Vulkan loader's device filter does: both the system loader (1.4.357) and the `libvulkan.so.1` bundled with Chrome for Testing read `VK_LOADER_DEVICE_ID_FILTER`. `browser/spike/harness/vk_devices.py` (ctypes, no vulkan-tools) lists the RTX 3090 (`0x2204`) and RTX 4060 Ti (`0x2805`) without the filter and only the 4060 Ti with `VK_LOADER_DEVICE_ID_FILTER=0x2805`, through either loader.

In Chromium (`gpu-pinning.json`): with the filter the WebGPU adapter is `nvidia`/`lovelace` (RTX 4060 Ti); without it, `nvidia`/`ampere` (RTX 3090). The no-filter check was one adapter request, no work. During every e5 WebGPU run `nvidia-smi` showed Chrome's GPU process (about 570 MiB) on GPU 1 only, GPU 1 at up to 1.4 GB used, and GPU 0 unchanged at 377 MiB before, during and after. WebGPU work stayed at e5-small, about 1 minute per variant, five variants plus two repeats.

`run_spike.py` sets the filter (default `0x2805`, `--vk-device-filter`) for every WebGPU run. This is a property of this machine's two-NVIDIA setup; another machine needs its own device id.

fp16 on WebGPU: the adapter does not expose the `shader-f16` feature (also not with `--enable-dawn-features=allow_unsafe_apis`, one attempt), so ORT cannot compile fp16 shaders. Status: unsupported on this setup; not investigated further.

Reliability note: in the first (high-load) series, two of the three int8 WebGPU runs failed with an exception from inside the .NET runtime whose message was not captured; each passed on a single rerun, and all WebGPU runs of the final series passed.

## 6. Catalog models: memory and download

Load-only probes (`run_probes.sh`, WASM backend, AOT app; session creation from the graph and its external-data file, no inference; `probes.json`):

| Model (graph) | Graph + data | Result | Create | Renderer peak RSS / page memory |
| --- | ---: | --- | ---: | --- |
| e5-small fp32 (section 3) | 470 MB | loads, runs, exact | 1.5 s | 2.4 GB / 1.45 GB |
| EmbeddingGemma 2 text fp32 (`eg2-text-dynamo-sdpa`) | 4.8 MB + 1,084 MB | **loads** | 2.2 s | 2.9 GB / 2.4 GB |
| GLiNER2 fp32 (route A, dynamo) | 3.0 MB + 829 MB | **loads** | 2.6 s | 2.5 GB / 2.0 GB |
| Qwen3.5-0.8B int4 (ORT GenAI CPU export) | 0.36 MB + 542 MB | **fails**: `Could not find an implementation for GatherBlockQuantized(1)` | – | 1.3 GB |
| Qwen3.5-0.8B fp32 | 0.3 MB + 3,077 MB | **fails**: the 3.08 GB data file cannot be fetched into one buffer (`Failed to fetch`); it would not fit the 4 GB wasm32 heap with the runtime either | – | 3.4 GB |

| Model | Browser feasibility | Main constraints |
| --- | --- | --- |
| multilingual-e5-small | **Feasible now** (measured). | Download: 109 MB brotli (int8-embed) to 249 MB (fp32). Memory 1.3 to 2.4 GB. |
| EmbeddingGemma 2, text | **Plausible, untested beyond load.** fp32 loads in WASM. | 1.08 GB download in fp32 is impractical; needs a reduced variant (embedding table is 134 M of 271 M parameters: int8 embeddings alone would give about 0.68 GB, fp16 about 0.54 GB, fp16 plus int8 embeddings about 0.4 GB), its own drift budget, and a check of the 32 MB tokenizer (3.5 MB brotli, loads in 0.29 s AOT). Long inputs (8k) multiply activation memory. |
| GLiNER2 | **Plausible, untested beyond load.** fp32 loads in WASM. | 829 MB fp32; DeBERTa embedding table 98 M parameters, so int8 embeddings would give about 0.45 GB. Extraction thresholds make drift visible as changed spans: a quantized variant needs a task-level budget, not only tensor tolerances. |
| Qwen3.5-0.8B | **Not feasible for a preview.** | No graph we have loads: the ORT GenAI int4 export uses `GatherBlockQuantized` (no WASM kernel in 1.30.0), fp32 exceeds browser limits, fp16 (1.57 GB) would need WebGPU with `shader-f16` (not exposed here). ORT GenAI has no web build, so the generation loop, KV cache and recurrent state handling would be new code in C#/JS. A WebGPU-oriented 4-bit export (MatMulNBits with plain Gather) would be a separate research spike. |

## 7. Effort estimate: "Starlings in the browser" preview (e5 only)

| Work | Days |
| --- | ---: |
| Session abstraction in the e5 encoder and an onnxruntime-web implementation (`[JSImport]` or `IJSRuntime`, `byte[]` transfer, cancellation, disposal) in a new package (for example `Starlings.Runtime.OnnxWeb`) | 4 to 6 |
| onnxruntime-web as a static web asset of that package: npm pin with integrity, PROVENANCE and THIRD-PARTY-NOTICES, COOP/COEP guidance, single-thread fallback, WASM-only vs WebGPU binary choice | 2 to 3 |
| Browser asset acquisition: fetch with SHA-256 verification, Cache Storage or OPFS caching, progress | 2 to 3 |
| int8-embed (and optionally fp32) as registered variant profiles: manifest, protected fixtures and tolerances set by the maintainer, a retrieval task metric, hosting of the graph | 3 to 5 |
| CI: wasm-tools workload install, AOT publish, headless Chromium (Playwright) running the tokenizer suite and the e5 fp32 and int8-embed drift on the WASM backend (hosted runners have no WebGPU; WebGPU stays local evidence) | 2 to 4 |
| Sample app, README, release drill (G6) for a browser consumer, G4 checks (reload, dispose, memory plateau in a tab) | 2 to 3 |
| **Total** | **15 to 24** |

Not included: EmbeddingGemma 2 text in the browser (reduced variant, budget, memory checks: about 8 to 12 more days), GLiNER2 (about 8 to 12), Qwen3.5 (research spike first), non-Chromium browsers and mobile (untested).

## 8. Capability status (proposed, explicit)

| Capability | Status | Evidence |
| --- | --- | --- |
| Owned tokenizer in browser-wasm (three profiles), interpreter and AOT | experimental | 0 mismatches on all fixture cases (section 2) |
| e5 fp32, onnxruntime-web WASM backend | experimental | drift 1.4e-7 |
| e5 fp32, onnxruntime-web WebGPU backend | experimental (one NVIDIA GPU, Linux, Chromium 153) | drift 2.1e-7 |
| e5 fp16, WASM | experimental | drift 1.2e-4 |
| e5 fp16, WebGPU | unsupported on the tested setup (no `shader-f16`); untested elsewhere | section 5.3 |
| e5 int8-embed, WASM and WebGPU | experimental | drift 3.5e-3, cosine 0.9998 |
| e5 int8 dynamic (either) | not proposed | batch dependence, cosine 0.963 to 0.989 |
| Multi-threaded WASM without COOP/COEP | unsupported (falls back to one thread) | section 3.3 |
| EmbeddingGemma 2 text, GLiNER2 in the browser | untested (load-only probes succeed) | section 6 |
| Qwen3.5-0.8B in the browser | unsupported (graphs fail to load) | section 6 |
| Firefox, Safari, Windows/macOS browsers, mobile, non-NVIDIA GPUs | untested | |
| Any support badge for WebAssembly | none (plan 13.1) | |

## 9. Reproducing

```sh
export DOTNET_ROOT=$HOME/.dotnet PATH=$HOME/.dotnet:$PATH
dotnet workload install wasm-tools                       # user-local SDK, no sudo; also needed per SDK feature band
npm ci --prefix browser/spike                            # onnxruntime-web 1.30.0 into browser/spike/node_modules
dotnet publish browser/spike/BrowserSpike -c Release -o $OUT/publish-interp
dotnet publish browser/spike/BrowserSpike -c Release -p:RunAOTCompilation=true -o $OUT/publish-aot   # about 5 min
dotnet build browser/spike/BrowserSpike.Native -c Release

uv venv -p 3.11 $OUT/venv
VIRTUAL_ENV=$OUT/venv uv pip install --require-hashes -r model-pipeline/reference/encoder/requirements-browser.txt
$OUT/venv/bin/playwright install chromium                # or download the same Chrome for Testing zip by hand
$OUT/venv/bin/python model-pipeline/reference/encoder/quantize_e5_browser.py \
  --model <e5 snapshot>/onnx/model.onnx --out $OUT/e5
ln <e5 snapshot>/onnx/model.onnx $OUT/e5/model-fp32.onnx  # a hard link: the server refuses symlinks out of its root

PYTHON=$OUT/venv/bin/python browser/spike/harness/run_all.sh "$PWD" $OUT/publish-interp $OUT/publish-aot \
  <chrome> $STARLING_MODELS $OUT/e5 $OUT/run
PYTHON=$OUT/venv/bin/python browser/spike/harness/run_probes.sh $OUT/publish-aot <chrome> <models root> $OUT/run
browser/spike/harness/compressed_sizes.sh 9 $OUT/e5/*.onnx
python3 browser/spike/harness/collect_evidence.py $OUT/run $OUT/run $OUT/e5/variants.json docs/evidence/browser-wasm-spike
```

WebGPU runs set `VK_LOADER_DEVICE_ID_FILTER` (default `0x2805`); check the adapter first with `run_spike.py gpu --backend webgpu ...` and pass the right device id for the machine.

## 10. Decisions needed

1. **Target.** Is WebAssembly a planned target (plan 13.1)? If yes, for which models first. Recommendation: e5 only, WASM backend as the qualified path, WebGPU as experimental local evidence.
2. **Variant.** Which e5 variant is the browser default: int8-embed (recommended), fp16, or fp32 only.
3. **Registry.** Do quantized variants join the registry as separate profiles (for example `multilingual-e5-small/onnx-int8embed-wasm`) with protected fixtures and tolerances set by the maintainer (proposed values in section 3.4)? Which task metric and evaluation set gates them?
4. **Hosting.** Where quantized graphs live (no upstream ONNX variants exist): a Starling Hub repository, release assets, or users run `quantize_e5_browser.py` themselves; e5 is MIT, so redistribution needs only the notice.
5. **Dependency.** Accept onnxruntime-web (npm, MIT) as a redistributed static web asset, with its own pin and notices, and decide WASM-only vs WebGPU-capable binary per package.
6. **CI.** Whether CI installs the wasm-tools workload and runs headless Chromium (adds a few minutes per run); WebGPU cannot run on hosted runners.
7. **Build mode.** AOT as the recommended consumer setting (+2.1 MB brotli at startup, 3x to 4x faster tokenizer).
8. **Spike code.** Keep `browser/spike/` as the starting point of a browser package, or remove it once a package exists.
