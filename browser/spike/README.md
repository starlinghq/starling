# Browser (WebAssembly) feasibility spike

Exploratory code for the browser feasibility spike of 2026-10-11. The report, with results, pins, the machine and the open decisions, is [`docs/m0/browser-wasm-spike.md`](../../docs/m0/browser-wasm-spike.md); raw results are in [`docs/evidence/browser-wasm-spike/`](../../docs/evidence/browser-wasm-spike/).

Nothing here ships. The projects are not packable and are in no solution that CI builds, because the AOT build needs the `wasm-tools` workload, which the CI runners do not install (`Directory.Build.props`). They still import the repository `Directory.Build.props` (central versions, locked restore, warnings as errors). WebAssembly has no support status: the report grants no badge (plan section 13.1), and there is no browser registry profile.

## What the spike tested

- **The owned tokenizer in the browser.** A Blazor WebAssembly app runs the protected tokenizer fixtures of three profiles (e5 Unigram, Qwen3.5 chat byte-level BPE, EmbeddingGemma 2 Gemma BPE) in headless Chrome, under the Mono interpreter and under AOT, and times load and encode against the same code on CoreCLR.
- **e5 inference in the browser.** `multilingual-e5-small` runs through onnxruntime-web 1.30.0, called from C# over `IJSRuntime`, on the WASM backend and on WebGPU, in fp32 and in quantized variants, compared with the reference embeddings.
- **Other catalog graphs.** Load-only probes of EmbeddingGemma 2 text, GLiNER2 and Qwen3.5-0.8B on the WASM backend.

Results are in sections 1 to 6 of the report, and the proposed capability statuses in section 8.

## Layout

| Path | Contents |
| --- | --- |
| `BrowserSpike/` | Standalone Blazor WebAssembly app (`net10.0`). `SpikeEntryPoints.cs` holds the `[JSInvokable]` methods the harness calls; `wwwroot/spike-ort.js` is the onnxruntime-web glue, an ES module that imports onnxruntime-web from `/ort/`. Trim warnings are turned on, so a trim-unsafe change fails the build. |
| `BrowserSpike.Native/` | Console app that runs the same tokenizer suite on CoreCLR, for the browser-versus-native comparison. Usage: `BrowserSpike.Native <repo root> <models root> [encode seconds] [warm loads]`; prints the same JSON document as the browser suite. |
| `Shared/TokenizerSuite.cs` | The conformance and timing code, compiled into both apps. |
| `harness/` | Python and shell scripts that serve the app and drive it headlessly (below). |
| `package.json`, `package-lock.json` | Pin `onnxruntime-web` 1.30.0 (npm). `node_modules/` is ignored by Git. |
| `Directory.Packages.props` | `Microsoft.AspNetCore.Components.WebAssembly` 10.0.12 (the patch of the `wasm-tools` browser-wasm runtime pack) and `Starlings.Tokenizers` `[0.3.0]`. |

Both apps reference the published `Starlings.Tokenizers` package (`PackageReference`, pinned exactly). The report, written on 2026-10-11, describes a project reference to the tokenizer source (`tokenizers-dotnet/src/Starlings.Tokenizers`); the tokenizer is now released from `starlinghq/tokenizers` and consumed as a package.

### Harness

| Script | Purpose and usage |
| --- | --- |
| `serve.py` | Static server: the published app's `wwwroot` at `/`, onnxruntime-web at `/ort/`, pinned `tokenizer.json` assets at `/data/tok/`, `model-pipeline/protected/` at `/data/fixtures/`, graphs at `/data/models/`. Sends `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` (needed for onnxruntime-web threads) unless `--no-isolation`, serves precompressed `.br`/`.gz` files, and reports bytes sent per path at `/__stats`. |
| `run_spike.py` | Starts `serve.py`, launches Chromium (Playwright's pinned Chrome for Testing, or `--chrome <path>`) with a fresh profile, calls the app's entry points and writes one JSON file of raw results. Suites: `tokenizer`, `e5`, `probe`, `gpu`. `--backend webgpu` adds the Chromium flags headless WebGPU needs and sets `VK_LOADER_DEVICE_ID_FILTER` (default `0x2805`, `--vk-device-filter`) to pin the GPU. |
| `run_all.sh` | The timing and drift runs of the report, in sequence: `run_all.sh <repo> <publish-interp> <publish-aot> <chrome> <tok models> <graphs dir> <out dir>`. `PYTHON` and `DOTNET` can be set. |
| `run_probes.sh` | Load-only probes of the other catalog graphs on the WASM backend: `run_probes.sh <publish dir> <chrome> <models root> <out dir>`. The graph paths inside are those of the M0 workstation. |
| `compressed_sizes.sh` | Raw, `gzip -9` and brotli sizes of downloaded files: `compressed_sizes.sh <quality> <file>...`. |
| `collect_evidence.py` | Copies the raw results into `docs/evidence/browser-wasm-spike/`, without embeddings and user agents: `collect_evidence.py <run dir> <probe/gpu dir> <variants.json> <evidence dir>`. |
| `summarize.py` | One line per variant and backend from a `drift_e5_browser.py browser` output file. |
| `vk_devices.py` | Lists the Vulkan devices a process sees, to check GPU pinning: `[VK_LOADER_DEVICE_ID_FILTER=0x2805] python vk_devices.py [libvulkan.so.1]`. |

The quantized e5 variants and the drift comparison come from `model-pipeline/reference/encoder/quantize_e5_browser.py` and `drift_e5_browser.py`, with the hash-locked environment `model-pipeline/reference/encoder/requirements-browser.txt`.

## Building and running

Build steps, from the project files:

```sh
dotnet workload install wasm-tools                     # needed for the AOT build
npm ci --prefix browser/spike                          # onnxruntime-web 1.30.0 into browser/spike/node_modules
dotnet publish browser/spike/BrowserSpike -c Release                            # Mono interpreter (default)
dotnet publish browser/spike/BrowserSpike -c Release -p:RunAOTCompilation=true  # AOT
dotnet build browser/spike/BrowserSpike.Native -c Release
```

The full reproduction (Python environment, Chrome for Testing, the quantized graphs, `run_all.sh`, `run_probes.sh` and evidence collection) is section 9 of the report. WebGPU device pinning is machine-specific: check the adapter with `run_spike.py gpu --backend webgpu ...` first and pass the right device id.

## Status

The report's eight decisions (section 10) went to the maintainer. `docs/plan/embeddinggemma2-follow-up.md` records them as approved on 2026-10-12 (D1: e5 only, WASM qualified and WebGPU experimental, int8-embed default, keep `browser/spike/` as the package seed) and the browser package as work item W6. The packages now exist: `Starlings.Runtime.OnnxWeb` (onnxruntime-web, WASM), `Starlings.Runtime.OnnxWeb.WebGpu` (the voluntary WebGPU adapter) and `Starlings.Models.E5.Browser` under `ai-dotnet/src/`, with the sample `ai-dotnet/samples/E5BrowserSearch` and the conformance harness `browser/conformance/` (evidence: `docs/evidence/browser-e5-package-2026-10-23/`). The four browser registry manifests and the evaluator checks are the rest of W6. This spike code stays as the record of the 2026-10-11 measurements; with SDK 10.0.100 and its ILLink, its `TrimmerSingleWarn=false` setting now surfaces trim warnings inside `Microsoft.AspNetCore.Components` and `Microsoft.JSInterop` 10.0.12, so its publish fails as committed (the conformance app collapses those per assembly instead).
