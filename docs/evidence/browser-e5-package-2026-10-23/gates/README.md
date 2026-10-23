# e5 browser profiles: G1 and G2 with the protected evaluator (2026-10-23)

The conformance runs of this evidence set ([`../README.md`](../README.md)), evaluated by `model-pipeline evaluate`, the evaluator that qualifies the native profiles. Each profile is checked against its own protected `tolerances.yaml` and the protected fixtures of `multilingual-e5-small/onnx-fp32-cpu`, which are its reference. The evaluator verified the protected files against Git (`protected_evidence: verified`, base `634fb790`, the merge base with `origin/main`). Nothing under `model-pipeline/protected/` changed. These are the development fixtures, not the P-030 release sets.

How the browser run maps onto the evaluator's stages:

| Stage | Producer | Content |
| --- | --- | --- |
| 1, tokenizer | `model-pipeline browser-results` from the run's `e5_tokenizer` | the E5 pipeline tokenizer in the browser (owned tokenizer, truncation to 512): ids and attention mask on the 46 fixture cases |
| 2, export | `model-pipeline/reference/encoder/stage_e5_browser_native.py` | the profile's graph on Python ONNX Runtime 1.30.0 (CPU), batch 1 per case, plus two padded batches, with the library's float64 pooling. There is no hidden-state stage: the browser tolerances bind no `float.last_hidden_state_max_abs`, so that check is reported as not applicable |
| 3, .NET | `model-pipeline browser-results` from the run's `embeddings`, `batch8` and `generator_embeddings` | `Starlings.Models.E5.Browser` on onnxruntime-web: one call per fixture id sequence, the padded batch of 8, and all 46 texts through `E5BrowserEmbeddingGenerator` in one call. The text rows are padded batches, so they are judged against the reference and against the same stage's single rows from the same ids |

`browser_vs_native_same_graph.embedding_max_abs` compares stage 3 with stage 2. Both stages must declare the same graph digest. The fp32 tolerance files set no bound for it, so for fp32 it is measured and reported only. `task_metric_before_preview.retrieval_top10_overlap_vs_fp32` (int8-embed) gates Preview, not G1 or G2. It is reported as **not run** in each int8-embed report. It needs browser embeddings of a P-030 held-out query set (the profile's budgeted G3 retrieval set), evaluated with `model-pipeline evaluate --g3 retrieval --baseline <fp32 embeddings>`, and no browser run has produced those.

## Results

Every run gives G1 passed and G2 passed, exit code 0 with `--require-gates G1,G2`, and first divergence none.

| Run | Profile | G1 | G2 | Embedding max abs (bound) | Min cosine (bound) | Text path max abs | Padded batch vs single, ids / text (bound 1e-5) | vs native, same graph (bound) | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32, AOT | `onnx-fp32-wasm` | passed | passed | 1.4e-7 (1e-5) | 1.0000000 (report only) | 1.35e-7 | 1.02e-7 / 1.02e-7 | 2.0e-7 (report only) | [report](fp32-wasm-aot/report.md) |
| fp32, interpreter | `onnx-fp32-wasm` | passed | passed | 1.4e-7 (1e-5) | 1.0000000 (report only) | 1.35e-7 | 1.02e-7 / 1.02e-7 | 2.0e-7 (report only) | [report](fp32-wasm-interp/report.md) |
| int8-embed, AOT | `onnx-int8embed-wasm` | passed | passed | 3.51e-3 (1e-2) | 0.9998038 (0.9995) | 3.51e-3 | 1.0e-7 / 1.0e-7 | 1.34e-7 (1e-5) | [report](int8embed-wasm-aot/report.md) |
| int8-embed, AOT run 2 | `onnx-int8embed-wasm` | passed | passed | 3.51e-3 (1e-2) | 0.9998038 (0.9995) | 3.51e-3 | 1.0e-7 / 1.0e-7 | 1.34e-7 (1e-5) | [report](int8embed-wasm-aot-r2/report.md) |
| int8-embed, AOT run 3 | `onnx-int8embed-wasm` | passed | passed | 3.51e-3 (1e-2) | 0.9998038 (0.9995) | 3.51e-3 | 1.0e-7 / 1.0e-7 | 1.34e-7 (1e-5) | [report](int8embed-wasm-aot-r3/report.md) |
| int8-embed, interpreter | `onnx-int8embed-wasm` | passed | passed | 3.51e-3 (1e-2) | 0.9998038 (0.9995) | 3.51e-3 | 1.0e-7 / 1.0e-7 | 1.34e-7 (1e-5) | [report](int8embed-wasm-interp/report.md) |
| int8-embed, AOT, 1 thread without isolation | `onnx-int8embed-wasm` | passed | passed | 3.51e-3 (1e-2) | 0.9998038 (0.9995) | 3.51e-3 | 1.0e-7 / 1.0e-7 | 1.34e-7 (1e-5) | [report](int8embed-wasm-aot-noiso/report.md) |
| int8-embed, AOT, proxy worker | `onnx-int8embed-wasm` | passed | passed | 3.51e-3 (1e-2) | 0.9998038 (0.9995) | 3.51e-3 | 1.0e-7 / 1.0e-7 | 1.34e-7 (1e-5) | [report](int8embed-wasm-aot-proxy/report.md) |
| fp32, WebGPU, AOT | `onnx-fp32-webgpu` | passed | passed | 2.05e-7 (1e-5) | 1.0000000 (report only) | 2.05e-7 | 1.5e-7 / 1.5e-7 | 2.06e-7 (report only) | [report](fp32-webgpu-aot/report.md) |
| int8-embed, WebGPU, AOT | `onnx-int8embed-webgpu` | passed | passed | 3.51e-3 (1e-2) | 0.9998038 (0.9995) | 3.51e-3 | 1.6e-7 / 1.6e-7 | 1.73e-7 (1e-5) | [report](int8embed-webgpu-aot/report.md) |

Stage 2 on its own (Python ONNX Runtime vs PyTorch): fp32 embedding max abs 2.32e-7, int8-embed 3.51e-3 with min cosine 0.9998038. So the int8-embed drift comes from quantization, not from the browser backend. The numbers match the harness's own `report.json` and `report-webgpu.json`.

The reports name the evaluator commit `a886b5da…-dirty`: the evaluator code for the browser path was not yet committed when they were made. The machine `omarchy` is the workstation of `../README.md` (i7-5820K; the WebGPU runs used the RTX 4060 Ti).

## Commands

The run files with per-case outputs are kept outside Git. `runs/*.json` are the same runs without them. The SHA-256 of each full file is recorded in its report (`inputs.results.*.environment.run_file_sha256`).

```sh
FX=model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu
G=<dir with model-fp32.onnx, model-int8-embed.onnx>          # quantize_e5_browser.py output
P=multilingual-e5-small/onnx-int8embed-wasm                  # each of the four profiles
MAN=model-pipeline/registry/$P/manifest.yaml

# Stage 2: native ONNX Runtime on the same graph (requirements-browser.txt environment)
PYTHONPATH=model-pipeline/src python model-pipeline/reference/encoder/stage_e5_browser_native.py \
  --graph $G/model-int8-embed.onnx --graph-path onnx/model-int8-embed.onnx \
  --fixtures $FX --profile $P --out <results> --intra-op-threads 4
#   (fp32 profiles: --graph $G/model-fp32.onnx --graph-path onnx/model.onnx)

# Stages 1 and 3 from one conformance run (run_conformance.py output with per-case outputs)
model-pipeline browser-results <run>.json --manifest $MAN --fixtures $FX --out <results> --machine "$(hostname)"

model-pipeline evaluate $MAN --results <results> --out <report> --require-gates G1,G2
```
