# e5 browser packages: conformance evidence (2026-10-23)

This tests the shipping code: `Starlings.Models.E5.Browser` on `Starlings.Runtime.OnnxWeb`, with the voluntary `Starlings.Runtime.OnnxWeb.WebGpu`. It runs in headless Chrome against the committed browser tolerances. It follows the 2026-10-11 spike ([docs/m0/browser-wasm-spike.md](../../m0/browser-wasm-spike.md)), which measured a prototype. The harness is in [`browser/conformance/`](../../../browser/conformance/README.md). This is exploratory evidence from one workstation, not a qualified benchmark, and it grants no support badge (plan 13.1). All four browser profiles pass every evaluated bound: fp32 and int8-embed on WASM and on WebGPU.

## Setup

| Item | Value |
| --- | --- |
| .NET | SDK 10.0.100, `wasm-tools` workload (browser-wasm runtime pack 10.0.12, Emscripten 3.1.56), `Microsoft.AspNetCore.Components.WebAssembly` 10.0.12 |
| onnxruntime-web | 1.30.0, npm integrity `sha512-q0y+JrrtukXSzsBWEMcc…ReKYQ==` (`ai-dotnet/src/Shared/OnnxWeb/package-lock.json`). WASM runs use the package's WASM-only bundle (`ort.wasm.min.mjs`, `ort-wasm-simd-threaded.wasm`); the spike used the WebGPU-capable asyncify bundle for every run |
| Tokenizer | `Starlings.Tokenizers` 0.3.2 (the central pin) |
| Browser | Chrome for Testing 153.0.8010.12 (headless), Playwright 1.63.0, fresh profile per run, COOP/COEP from `browser/spike/harness/serve.py` unless stated |
| Reference | protected fp32 PyTorch embeddings of `multilingual-e5-small/onnx-fp32-cpu` (46 cases); Python ONNX Runtime 1.30.0 (CPU) on the same graph for `browser_vs_native_same_graph` |
| Graphs | fp32 `onnx/model.onnx` at `614241f6` (SHA-256 `ca456c06…8665`); int8-embed from `quantize_e5_browser.py` (SHA-256 `374a3cc0…fb19`) |
| Machine | i7-5820K (6 cores, 12 threads, AVX2), 94 GB RAM, Linux x64. Load average 3 to 10 and 32 to 33 GB available during the runs (`load-average.txt`) |
| Threads | onnxruntime-web's default: 4 of 12 logical cores. The spike set 6 explicitly, so its latencies are not directly comparable |

## Results against the committed tolerances

Every WASM run passes every bound in its profile's `tolerances.yaml` (`report.json`). The numbers are the same for both build modes, for 1 or 4 threads, and with the proxy worker, and they repeat exactly across runs:

| Profile | Runs | Token ids, masks, offsets | Embedding max abs (limit) | Min cosine (limit) | Padded batch vs single (limit) | Browser vs native, same graph (limit) | NaN/Inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `onnx-fp32-wasm` | AOT, interpreter | 0 mismatches (46 e5 inputs; 46 tokenizer cases, UTF-16 offsets) | 1.4e-7 (1e-5) | 1.0000000 (report only) | 1.0e-7 (1e-5) | 2.0e-7 (no bound) | none |
| `onnx-int8embed-wasm` | AOT ×3, interpreter, 1 thread without isolation, proxy worker | 0 mismatches | 3.51e-3 (1e-2) | 0.9998038 (0.9995) | 1.0e-7 (1e-5) | 1.3e-7 (1e-5) | none |

- `task_metric_before_preview.retrieval_top10_overlap_vs_fp32` (int8-embed) is **not evaluated**. It needs a P-030 held-out query set, and it gates Preview, not Experimental.
- The text path, `E5BrowserEmbeddingGenerator` over all 46 texts in one call, matches the id path within 1.0e-7. Its rows run as padded batches, so it is not bit for bit.
- The library code itself is checked separately. `NativeParityTests` runs `E5BrowserEncoder` on native ONNX Runtime and gets the same result as `E5Encoder` bit for bit, single rows and padded batches. So every difference above comes from the backend.

## Load, latency, memory

Warm medians in milliseconds (`runs/*.json` hold every sample). "Passage" is the 34-token fixture case `e5-passage-prefix`; "text" is the same passage end to end through the generator, including tokenization.

| Run | Session creation | Passage, batch 1 | Short batch of 8 | One 512-token input | Text through generator | Renderer peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32, AOT | 1,416 | 25.5 | 217 | 439 | 28.7 | 2.30 GB |
| fp32, interpreter | 1,406 | 25.7 | 217 | 418 | 26.3 | 2.29 GB |
| int8-embed, AOT (run 3) | 952 | 25.3 | 215 | 414 | 26.6 | 1.44 GB |
| int8-embed, interpreter | 903 | 25.7 | 216 | 418 | 26.9 | 1.41 GB |
| int8-embed, AOT, proxy worker | 944 | 25.3 | 216 | 417 | 26.7 | 1.53 GB |
| int8-embed, AOT, 1 thread (no COOP/COEP) | 926 | 80.9 | 797 | 1,481 | 80.0 | 1.40 GB |

- **Repeatability.** Runs 1 and 2 of int8-embed AOT each had one row 3x to 4x slower than run 3 (110 ms passage in run 1; 719 and 1,128 ms in run 2), with the same drift. That's background load on a shared machine, so read single medians with care.
- **Threads.** Without cross-origin isolation, onnxruntime-web runs one thread, which here is 3.2x to 3.7x slower.
- **Proxy worker.** It costs nothing measurable and keeps the page's main thread free during inference.
- **Build mode.** The interpreter build loads the 17 MB tokenizer in about 2.5 s, against 0.8 s for AOT. That's the time from the start to the end of the tokenizer stage, including the localhost fetch. Inference is the same in both modes.
- **Memory.** int8-embed uses about 0.9 GB less renderer memory than fp32.

## Cache Storage

`runs/cache-int8embed-wasm-aot.json`: two consecutive `LoadAsync` calls with Cache Storage on. The server sent `tokenizer.json` (17.1 MB) and the graph (181.7 MB) exactly once across both loads, and the second load returned a bit-identical embedding. Load times were 3.3 s, then 2.4 s.

## The sample (`ai-dotnet/samples/E5BrowserSearch`)

The sample was built from packages packed from this branch (a local feed with isolated package caches), then driven through its UI by `browser/conformance/smoke_sample.py`:

- **int8-embed, self-hosted** (`smoke-sample-int8embed.json`): Load to ready in 11.2 s, including the 182 MB download and embedding 28 passages (2.1 s). Queries took 80 to 117 ms on the WASM bundle with 4 threads.
- **fp32 from the Hugging Face Hub** (`smoke-sample-fp32-hub.json`): the zero-setup default. The page fetched the graph (470 MB) and the tokenizer from `huggingface.co` at the pinned revision, under COEP `require-corp`, which works because the Hub sends CORS headers on the redirect and the CDN response. It verified both SHA-256s and was ready in 20.2 s. Queries took 35 to 71 ms.
- **Rankings.** Both variants rank the 28 passages identically. Six of seven candidate example queries (`smoke-candidates.json`) put the passage they are about first, mostly across languages: English to Japanese, Spanish and Italian, and French to English. The seventh, "what to drink with dates", missed the Arabic coffee passage. The sample's example list keeps the queries that rank as intended. A final run of the rebuilt sample (`smoke-sample-final.json`, `sample-screenshot.png`) ranks the intended passage first for all three default smoke queries.

## WebGPU

A rerun on the same day, on the conformance app rebuilt from merged `main` (`634fb790`), passed. The harness pins WebGPU to GPU 1 (RTX 4060 Ti, `VK_LOADER_DEVICE_ID_FILTER=0x2805`), as in the spike. `nvidia-smi-before-webgpu.csv` and `nvidia-smi-after-webgpu.csv` show GPU 1 at 502 MiB used before the runs and 510 MiB after, with GPU 0 (RTX 3090) unchanged at 298 MiB. Load average was 4 (`load-average-webgpu.txt`). The results are in `report-webgpu.json` and `runs/*-webgpu-aot.json`.

| Profile | Embedding max abs (limit) | Min cosine (limit) | Padded batch vs single (limit) | Browser vs native, same graph (limit) | Text through generator vs ids |
| --- | --- | --- | --- | --- | --- |
| `onnx-fp32-webgpu` | 2.05e-7 (1e-5) | 1.0000000 (report only) | 1.5e-7 (1e-5) | 2.1e-7 (no bound) | 1.5e-7 |
| `onnx-int8embed-webgpu` | 3.51e-3 (1e-2) | 0.9998038 (0.9995) | 1.6e-7 (1e-5) | 1.7e-7 (1e-5) | 1.6e-7 |

Token ids, masks and offsets: 0 mismatches. `task_metric_before_preview` is not evaluated, as for WASM.

| Run | Session creation | First run (shader compilation) | Passage, batch 1 | Short batch of 8 | One 512-token input | Renderer peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32, WebGPU | 2,731 ms | 234 ms | 18.3 ms | 17.3 ms | 17.8 ms | 3.19 GB |
| int8-embed, WebGPU | 1,871 ms | 261 ms | 18.6 ms | 18.5 ms | 18.9 ms | 1.51 GB |

- **A flat floor.** As in the spike, WebGPU costs about 18 ms per call whatever the batch or length. Against 4 WASM threads it is 1.4x faster on one short passage (18 vs 25 ms), 12x on a batch of 8 and 23x on a 512-token input.
- **ORT's placement warning.** onnxruntime-web logs that some nodes run on the CPU instead (`VerifyEachNodeIsAssignedToAnEp`), as in the spike.
- **The earlier attempt.** An attempt earlier the same day failed with `VK_ERROR_OUT_OF_DEVICE_MEMORY`, because another process held 15.6 of 16.4 GB of GPU 1 at the time. Those runs were replaced by these.

## Harness pitfalls found on the way

- **Stale IL-stripped assemblies.** An interpreter publish that reused an `obj/` left by an AOT publish shipped IL-stripped assemblies, and the Mono interpreter aborted in `Starlings.Tokenizers.Tokenizer:FromUtf8Json` (`NIY encountered`, then `Assertion: should not be reached at interp.c:4135`). Builds are now published from a clean `obj/` per mode (`browser/conformance/README.md`). The renderer crashes seen in an earlier matrix at load average about 55 did not reproduce in five int8-embed runs at normal load, three of them in exactly the crashed configuration.
- **Lock-file rewrites.** Publishing the browser app rewrites the referenced libraries' `packages.lock.json` (`Microsoft.NET.ILLink.Tasks` 10.0.12 instead of 10.0.0). The committed lock files are the solution restore's.

## Files

| File | Content |
| --- | --- |
| `report.json` | `evaluate_e5_browser.py` output: every bound with observed value, limit and status; statistics; load stages; latency medians; memory |
| `runs/*.json` | `run_conformance.py` results without per-case embeddings and tokenizer outputs (`collect_evidence.py`): runtime info, load stages, latency samples, bytes served per path, console |
| `smoke-*.json` | sample UI runs and the example-query check |
| `sample-screenshot.png` | the sample after a search (int8-embed, WASM, 4 threads) |
| `load-average.txt` | load average and available memory at the start of each run |
| `report-webgpu.json` | the evaluator's output for the two WebGPU runs |
| `nvidia-smi-before-webgpu.csv`, `nvidia-smi-after-webgpu.csv` | GPU memory before and after the WebGPU runs |
| `load-average-webgpu.txt` | load average and available memory during the WebGPU runs |
