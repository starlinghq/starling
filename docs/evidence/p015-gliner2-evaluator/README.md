# Staged fidelity evaluation, entity extraction: gliner2-base-v1/onnx-fp32-cpu, stages 1 and 2 (P-015)

`report.json` and `report.md` are the unedited output of `model-pipeline evaluate` on real model output, 2026-10-10, in a measurement-only configuration: no protected tolerances file exists for this profile, so every bound is unset, both gates are `not-run`, and the exit code is 4. This is not gate evidence and changes no manifest.

Development evidence from one workstation, against the fixtures proposed in PR #19 (`model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/extraction.jsonl`, 58 agent-written cases; not yet accepted, not the P-030 release set).

## How it ran

The fixtures are not on `main` and the draft manifest pins no ONNX graph, so the evaluator ran in throwaway Git repositories under `.models/` (not committed) holding a copy of the draft manifest, the PR #19 `extraction.jsonl`, and, for the third run only, a temporary tolerances file. Nothing was written to `model-pipeline/protected/`. The report's `evaluator.repository_commit` and `protected_evidence.base` are the throwaway repository's commit, not this repository's.

| Stage | Producer | Environment |
| --- | --- | --- |
| 1 processor | `model-pipeline/reference/gliner2/stage_gliner2.py tokenizer` (GLiNER2's own Python `ExtractorCollator`; reproduces the fixture processor path, not the owned C# processor) | Python 3.11, gliner2 at `55656fbf`, transformers 4.57.6, tokenizers 0.22.2 (P-014 `venv-cpu`) |
| 2 export | `stage_gliner2.py export`: Python ONNX Runtime 1.30.0 CPU, 1 thread, on the P-014 route A dynamo graph (`gliner2-A-dynamo.onnx`, declared as `onnx/gliner2-entity-route-a.onnx[.data]`) with the fixture inputs; PyTorch hidden states recomputed with torch 2.14.1+cpu | same |
| 3, 4 .NET | not run (the C# package is in progress on `claude/p015-gliner2`) | |

All on one Intel i7-5820K (Linux x64), checkpoint `fastino/gliner2-base-v1@f9634218`. Stage 2 took about 3 minutes (PyTorch hidden states dominate); the evaluator takes about 5 s.

| Run | Manifest | Tolerances | Result |
| --- | --- | --- | --- |
| A | draft manifest as on `main` | none | exit 1. `export.identity` invalid: the stage's graph is not in the manifest's `origin.files` (no graph is pinned; decision 1 of docs/m0/gliner2-p014.md). First divergence `export`, `packaging/harness`. G1 not-run, G2 failed. |
| B (committed here) | draft manifest + the graph's two digests added to `origin.files` | none | exit 4. Every bound-dependent check `not-run`, metrics measured; G1 and G2 not-run. |
| C | as B | temporary file with the proposed values below | exit 4 with `--require-gates G1,G2`. **G1 passed**, G2 not-run only because stages 3 and 4 have no results; every stage 2 check passed, 0 entity exceptions. |

## Numbers (58 cases; run C's bounds are the P-014 proposals, not maintainer-set)

| Check | Proposed bound | Max abs | Mean abs | p99 abs | Max rel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Processor: normalized text, words, maps, ids, mask, word and schema positions | exact | 58/58 exact | | | |
| Hidden state, ORT vs PyTorch | 1e-4 | 2.29e-5 | 3.7e-7 | 2.6e-6 | |
| Count logits | 5e-4 | 4.8e-5 | 1.1e-5 | 3.8e-5 | |
| Gate decision | exact | 58/58 | | | |
| Span logits (valid spans), relative | 1e-4 | 7.76e-3 | 9.3e-5 | 7.0e-4 | 2.40e-5 |
| Scores | 2e-5 | 3.6e-6 | 7.7e-10 | 2.5e-11 | |
| Entities (rule) | exact + exceptions 2e-5 | 58/58 exact, 0 exceptions | | | |
| Confidence of matched entities | 2e-5 | 1.8e-6 | | | |
| Padded batches (by label count, up to 8; 56 rows) vs own singles, span logits | 1e-4 rel | 0 | 0 | 0 | 0 |
| Padded batches vs reference, span logits | 1e-4 rel | 7.76e-3 | 8.8e-5 | 6.5e-4 | 2.07e-5 |

These match the P-014 parity read (docs/m0/gliner2-p014.md, section 5). The stage's PyTorch rerun reproduced the fixture's hidden state bit for bit on 58/58 cases (`reference_rerun_bit_identical_to_fixture_cases`, by the fixture's `tensor_sha256`).

## The entity rule on the P-014 C# smoke outputs

The throwaway C# smoke from P-014 wrote only entities (no logits or batches), so it cannot be a stage 3 results file. Its entity outputs were run through `evaluate_extraction.compare_entities` with the proposed bounds (2e-5, 2e-5):

| Output | Violations | Allowed exceptions |
| --- | --- | --- |
| CPU, 1 thread | 0 | 0 |
| CUDA, `use_tf32=0` | 0 | 0 |
| CUDA, `use_tf32=1` | 1: `threshold-near-below-patient` drops `patient` (reference score 0.437065095, threshold 0.437, margin 6.5e-5 > 2e-5) | 6 order-swap pairs in `long-over-1024-tokens` (person and location items whose reference confidences differ by at most 3.6e-7) |

So under the proposed rule the TF32 read fails on the near-threshold flip the corpus was built to catch, and its reorder is tolerated and listed.

## Reproduce

Stage commands: `model-pipeline/reference/README.md` (GLiNER2 section). The throwaway-repository script is not committed: it copies the draft manifest (adding the graph digests for runs B and C), the PR #19 `extraction.jsonl` and, for run C, a temporary `tolerances.yaml` into a new Git repository, commits, and calls `model_pipeline.evaluate.evaluate(..., protected_base="HEAD", require_gates=("G1", "G2"))`.
