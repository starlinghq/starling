# G1/G2 rerun at Starlings.Tokenizers 0.3.0-preview.1

Workstream R4 (`docs/plan/production-readiness.md`). The tokenizer pin moved from `0.1.0-preview.4` to the published `Starlings.Tokenizers` `0.3.0-preview.1`, and the manifests said "Model gates were not rerun". This directory reruns the protected staged evaluator at the new pin, on CPU, for the profiles below.

**These are development fixtures, not P-030 release sets.** Every case comes from the agent-written development corpora the maintainer accepted as development fixtures (e5: M0 corpus; GLiNER2: PR #19; EmbeddingGemma 2: PR #29 and PR #42). A pass here says the new tokenizer pin reproduces the earlier development evidence; it is not release qualification. The manifests' gate entries are **unchanged**: recording a gate result is the maintainer's act (plan M10). Whether a tokenizer bump needs a model rerun at all is open decision M8.

Each `report.md` / `report.json` is the unedited output of

```sh
model-pipeline evaluate model-pipeline/registry/<profile>/manifest.yaml --results <dir> --out <dir> --require-gates G1,G2
```

run on 2026-10-13 at `2f53409` (branch `r4-manifests-gates`; the manifests at that commit differ from `fb27658` only in comments and gate entries the evaluator does not read) with the protected evidence verified against Git, the binding tolerances and the accepted development fixtures of each profile. Nothing under `model-pipeline/protected/` was changed.

## Results

| Profile | Exit | G1 | G2 | First divergence | Key numbers (bound) | Same as the earlier evidence |
| --- | ---: | --- | --- | --- | --- | --- |
| [`multilingual-e5-small/onnx-fp32-cpu`](e5-onnx-fp32-cpu/report.md) | 0 | passed | passed | none | ids and masks exact 46/46; C# from text vs reference embedding max abs 2.32e-7 (1e-5); C# vs Python ORT 1.0e-7 (1e-6); Python ORT vs PyTorch 2.3e-7 | yes: equal to `docs/releases/e5-0.1.0-preview.2/fidelity-report.md` |
| [`gliner2-base-v1/onnx-fp32-cpu`](gliner2-onnx-fp32-cpu/report.md) | 0 | passed | passed | none | processor outputs exact 58/58; span logits relative 2.4e-5 (1e-4); scores 3.64e-6 (2e-5); entities 58/58 with 0 exceptions; text-path confidence 1.81e-6 (2e-5); C# vs Python ORT span logits 0 (1e-6) | yes: equal to `docs/evidence/p015-gliner2-dotnet/` |
| [`embeddinggemma-2/onnx-fp32-cpu`](embeddinggemma2-onnx-fp32-cpu/report.md) | 0 | passed | passed | none | ids exact 138/138; C# from text embedding max abs 3.7e-7 (5e-6), min cosine 0.999999999995; C# vs Python ORT 4.2e-7 (1e-6); token embeddings per case 6.2e-6 (1e-4) | yes: equal to `docs/evidence/embeddinggemma2-dotnet/` |
| [`embeddinggemma-2/image-fp32-cpu`](embeddinggemma2-image-fp32-cpu/report.md) | 0 | passed | passed | none | ids exact 42/42; decoded RGB, resized pixels, pixel values and positions exact 35/35; embedding max abs 2.24e-6 (1e-5), min cosine 0.999999999893; soft tokens row-relative 1.57e-4 (1e-3); token embeddings 4.54e-5 (1e-3); both batch layouts 0 against single rows | yes: equal to `docs/evidence/embeddinggemma2-images/` (stage 3 only; `--reference-tensors` = the spike's `ref-clean-tensors`, outside Git, checked against the protected `tensors.json`) |

No mismatch was found, and nothing was adjusted.

## Qwen3.5-0.8B `ortgenai-fp32x-cpu` (no evaluator task)

`model-pipeline evaluate` has no generation task yet (`docs/evidence/p016-qwen35/README.md` section 10; workstream R5), so this profile was rerun the way its P-016 evidence was produced: `ai-dotnet/tools/Starlings.Qwen35.Fidelity -- --variant fp32x --device cpu --modes parity` (Release, `Starlings.Tokenizers/0.3.0-preview.1` in its `deps.json`) on the 22 prompts of `prompts-v1.jsonl`, with the bundle from the acquisition cache, then `model-pipeline/reference/generation/compare.py` against the protected `generation.jsonl`. The full reference step-0 vectors are outside Git; the copies used (`p016/ref-cpu/step0`) were checked against the 22 SHA-256 digests in the protected `environment.json` first (22/22 equal). Bounds from `model-pipeline/protected/qwen3.5-0.8b/ortgenai-fp32x-cpu/tolerances.yaml`, checked by hand against `compare.json`:

| Check | Bound | Result |
| --- | --- | --- |
| Template rendering (owned C# renderer) | exact | 21/21 equal; the `tools` prompt is rejected as unsupported, as before |
| Prompt ids (owned tokenizer on the rendering) | exact | 21/21 equal |
| Greedy ids | exact except listed near-ties | 22/22 exact, token match 1.000 |
| Finish reason | equal | 22/22 |
| Step-0 argmax | equal | 22/22 |
| Step-0 log-probability max abs | 5e-3 | 1.38e-3 |
| Step-0 top-5 log-probability max abs | 2e-3 | 6.2e-4 |
| Streaming reassembly and usage | exact | ok on 22/22 (`run.log`) |

Equal to the P-016 CPU numbers (`docs/evidence/p016-qwen35/compare-cpu-fp32x.json`). Semantics mode (G4 checks) was not rerun: it does not depend on the tokenizer beyond the text-path check that parity already covers. Files: `qwen35-ortgenai-fp32x-cpu/compare.json`, `environment.json`, `run.log`.

## Stage results

| Stage | Producer |
| --- | --- |
| 1 tokenizer / processor, 3 .NET | `ai-dotnet/tools/Starlings.Fidelity.Tool` (Release) at `2f53409`, resolving `Starlings.Tokenizers/0.3.0-preview.1` (its `deps.json` and `packages.lock.json`): `-- e5 --hidden`, `-- gliner2 --threads 1`, `-- embeddinggemma2`, `-- embeddinggemma2-images`. Model files from the acquisition cache (`$STARLING_MODELS`), every digest checked against the manifests before the run. |
| 2 export (tokenizer-independent; rerun so the reports are complete) | e5: `model-pipeline/reference/encoder/stage_e5_export.py` in a fresh environment from `encoder/requirements.txt` (Python 3.11.16, torch 2.14.1+cpu, onnxruntime 1.30.0, sentence-transformers 6.1.0, transformers 5.18.0). GLiNER2: `reference/gliner2/stage_gliner2.py export --graph-asset-prefix onnx/gliner2-entities-fp32 --threads 1` on the hosted graph (`0ba2db95...`) in the P-014 CPU environment (transformers 4.57.6). EmbeddingGemma 2: `reference/embeddinggemma2/stage_embeddinggemma2_export.py` on the release text graph (`02cc7ecc...`) in the spike environment (transformers 5.19.0). PyTorch weights from local snapshots at the pinned revisions (`model.safetensors` digests equal to `origin.files`). |

One workstation: Intel i7-5820K (12 threads, AVX2), 94 GB RAM, Linux x64, .NET 10.0.12, ONNX Runtime 1.30.0 CPU, shared with other jobs.

## Run notes

- The first EmbeddingGemma 2 image run of the fidelity tool was killed (exit 137) after embedding every row, before writing `dotnet.results.json`, while the machine was short of memory; the evaluator reported G2 `not-run` (exit 4). The unchanged second run completed; the report here is from it.
- The first EmbeddingGemma 2 stage 2 run ended without writing `export.results.json` and without an error message while the machine was short of memory (likely killed). The evaluator then reported G2 `not-run` for the missing export checks (exit 4), as designed. The second run, unchanged, completed; the report here is from it.
- The machine ran other jobs throughout (load average 18 to 21 during the Qwen run); timings in `run.log` are not benchmarks.
- Not rerun: CUDA profiles (no evaluator CUDA mode for EmbeddingGemma 2, workstream R7 W2; GPU time is maintainer item M6) and `embeddinggemma-2/video-fp32-cpu` (not in this workstream's scope; its earlier evaluator run is `docs/evidence/embeddinggemma2-video/`).

## Reproduce

Commands per model: `model-pipeline/reference/README.md`. Stage results and tensors (hundreds of MB) stay outside Git.
