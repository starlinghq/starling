# e5 browser conformance

Evidence for the browser packages (`Starlings.Runtime.OnnxWeb`, `Starlings.Runtime.OnnxWeb.WebGpu`, `Starlings.Models.E5.Browser`). The tests here exercise the code that ships, unlike `browser/spike/`, which measured a prototype. Results are in [`docs/evidence/browser-e5-package-2026-10-23/`](../../docs/evidence/browser-e5-package-2026-10-23/README.md).

| Path | Contents |
| --- | --- |
| `E5Conformance/` | Blazor WebAssembly app with project references to the packages. `ConformanceEntryPoints.Run` loads one variant on one provider through `E5BrowserEmbedder.LoadAsync`, then runs the checks below and returns JSON. `RunCacheCheck` loads twice with Cache Storage on. |
| `run_conformance.py` | Serves a publish with `../spike/harness/serve.py` (COOP/COEP, precompressed files, bytes per path) and drives the app in headless Chromium through Playwright. It writes one JSON result, plus `<out>.console.log`, which streams the browser console as it happens, so a crash shows the last stage reached. |
| `evaluate_e5_browser.py` | Checks results against the profile's committed `tolerances.yaml`. Every bound is reported as passed, failed or not evaluated. |
| `smoke_sample.py` | Drives the `E5BrowserSearch` sample through its UI: Load, a few queries, the top hits and a screenshot. |

`Run` performs these checks:
- the e5 tokenizer against the ids of the protected embeddings
- the owned tokenizer against `tokenizer.jsonl` (ids, attention mask, UTF-16 offsets)
- embeddings of the 46 fixture id sequences, one per call
- the padded batch of the 8 short cases
- the same texts end to end through `E5BrowserEmbeddingGenerator`
- warm latency

`evaluate_e5_browser.py` also runs the same graph on Python ONNX Runtime (CPU) for `browser_vs_native_same_graph`.

## Build

Publish each build mode from a clean `obj/`:

```sh
app=browser/conformance/E5Conformance
rm -rf $app/obj $app/bin && dotnet publish $app -c Release -p:RunAOTCompilation=true -o $OUT/conf-aot
rm -rf $app/obj $app/bin && dotnet publish $app -c Release -o $OUT/conf-interp
```

An AOT publish leaves IL-stripped assemblies in `obj/Release/net10.0/wasm/for-publish/stripped/` (`WasmStripILAfterAOT`). A later interpreter publish from the same `obj/` can ship those, and the Mono interpreter then aborts on the first stripped method (`NIY encountered in method Starlings.Tokenizers.Tokenizer:FromUtf8Json`). That happened once during this evidence run. It's an SDK build-cache effect, not a package defect, but a consumer who switches build modes in one project directory can hit it too.

Publishing the app also re-restores the referenced libraries under `ai-dotnet/src/` in the context of a browser-wasm app, and rewrites their `packages.lock.json` files: `Microsoft.NET.ILLink.Tasks` becomes 10.0.12 instead of the SDK's 10.0.0. CI restores those projects in locked mode against the solution restore. So after a conformance build, restore the lock files with `git checkout -- 'ai-dotnet/**/packages.lock.json'`, or regenerate them with `dotnet restore ai-dotnet/AI.slnx`, before committing.

## Run

```sh
python run_conformance.py --publish $OUT/conf-aot --label aot --variant fp32 --provider wasm \
    --models-root ~/.cache/starlings/models --graphs <dir with model-fp32.onnx, model-int8-embed.onnx> \
    --chrome <Chrome for Testing 153> --out $OUT/run/fp32-wasm-aot.json
python evaluate_e5_browser.py --graphs <same dir> --out $OUT/run/report.json $OUT/run/*.json
```

Options:
- `--provider webgpu` adds the headless WebGPU flags and pins the Vulkan device (`--vk-device-filter`, default `0x2805`, the RTX 4060 Ti of the M0 workstation).
- `--no-isolation` drops COOP/COEP, so one thread.
- `--proxy` uses onnxruntime-web's worker.
- `--cache-check` runs the reload check.

The Python environment is the spike's hash-locked `model-pipeline/reference/encoder/requirements-browser.txt` (numpy, onnxruntime 1.30.0, Playwright 1.63.0). The int8-embed graph is `quantize_e5_browser.py`'s output (SHA-256 `374a3cc0…fb19`).
