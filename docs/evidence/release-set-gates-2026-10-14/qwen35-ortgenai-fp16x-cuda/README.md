# qwen3.5-0.8b `ortgenai-fp16x-cuda` on the P-030 release sets

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. The tool was the CUDA build (ONNX Runtime GenAI 0.17.1), with `use_tf32: 0`. The bounds come from `protected/qwen3.5-0.8b/ortgenai-fp16x-cuda/`.

| Gate | Set | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- | --- |
| G1 | tokenizer release (232) | **passed** | ids exact 232/232 | [`g1-tokenizer-release/report.md`](g1-tokenizer-release/report.md) |
| G1 | release prompts (32) | **passed** | template and ids exact 32/32 | [`g1-g2-release-prompts/report.md`](g1-g2-release-prompts/report.md) |
| G2 | release prompts (32) | **not-run** | Greedy ids exact on 30/32. Two divergences are allowed near ties under `equal_until_first_divergence_at_margin_below_0.05`: `oasst1-496a8740` at step 24 (margin 0.00249) and `oasst1-b3321bbb` at step 17 (0.00655). Finish reason and step-0 argmax 32/32. Step-0 top-5 0.0273 (0.06). Step-0 on the reference top 64: 0.0414 (0.15). The runtime-semantics keys and the full-vocabulary bound did not run, as for fp32x. | same |
| G2 | release prompts (32) | **passed** (rerun below) | full-vocabulary step 0 0.0478 (0.15); cached vs uncached 0.0531 (0.1); chunked vs one-call prefill 0.0401 (0.1); concurrency equal; greedy 30/32 exact with 2 allowed near-tie divergences | [`g2-release-prompts-fixture-semantics/report.md`](g2-release-prompts-fixture-semantics/report.md) |
| G3 | GSM8K, 50 problems | **passed** | exact match **0.64** (32/50; minimum 0.56); no row's correctness differs from the transformers fp32 reference | [`g3-gsm8k/report.md`](g3-gsm8k/report.md) |

Wall clock: release-prompt parity 16.7 s; GSM8K 120 s.

## G2 on the release prompts, rerun (2026-10-14)

**G2 passed** on the release prompts, exit 0, at `e360188` (branch `qwen-release-semantics`). The two reasons for the first run's not-run are resolved:

1. **Full step-0 reference vectors.** `run_reference.py` regenerated them in the environment the fixture records (`release-prompts/reference/environment.json`): torch 2.14.1+cpu, transformers 5.18.0, tokenizers 0.23.2, numpy 2.4.6, safetensors 0.8.0, no flash-linear-attention, Python 3.11.16, CPU, `--threads 6`. All 32 `step0/<id>.f32` are byte-identical to the SHA-256 the fixture records per case (`step0_sha256`, and `step0_full_vectors.sha256`), and the evaluator checked each file against them (`--reference-tensors`). The vectors stay outside Git (`~/.cache/starling-release-refs/qwen-release-t6`). The first run's mismatch came from the torch +cu130 wheel at 4 threads.
2. **Runtime-semantics keys.** `Starlings.Qwen35.Fidelity --modes fixture-semantics` runs the three checks that feed G2 (cached vs uncached, chunked vs one-call prefill, concurrency) on any generation fixture, with the cases chosen from the fixture by a fixed rule that `semantics.json` records under `selection`: cached vs uncached on the case with the largest smallest reference top-2 margin over its first 12 steps (`oasst1-68a06244`); concurrency on the six cases with the largest smallest margin over 24 steps (four in parallel on two threads, two interleaved); chunked prefill on the longest prompt (`oasst1-330fc1cf`, 92 tokens: chunks 64 and 32) and the second longest (`oasst1-3c5b52fd`, 83 tokens: chunk 7). These checks compare with the reference's greedy prefix, so the rule avoids near ties; the greedy-ids check keeps every case. The development-only checks of `--modes semantics` (max-token boundary, stop sequences, cancellation, sampling, text path) bind no tolerance key and are not part of this run. The evaluator now refuses a `semantics.json` that names cases outside the evaluated fixture, so a development-prompt run cannot stand in for the release prompts.

| Check (binding key) | Measured (bound) | Status |
| --- | --- | --- |
| Template rendering, prompt ids (G1) | 32/32 exact | passed |
| Greedy ids (`equal_until_first_divergence_at_margin_below_0.05`) | 30/32 exact; `oasst1-496a8740` diverges at step 24 (reference margin 0.00249) and `oasst1-b3321bbb` at step 17 (0.00655), both allowed and listed | passed |
| Finish reason (`equal_when_ids_equal`), step-0 argmax | 32/32 | passed |
| Step-0 top-5 log-probability | 0.0273 (0.06) | passed |
| Step-0 log-probability, full vocabulary | 0.0478 (0.15); top 64 0.0414 | passed |
| Cached vs uncached (12 steps, `oasst1-68a06244`) | 0.0531 (0.1), prefix equal | passed |
| Chunked vs one-call prefill (`oasst1-330fc1cf` chunks 64 and 32, `oasst1-3c5b52fd` chunk 7) | 0.0401 (0.1), tokens equal | passed |
| Concurrency (4 parallel on 2 threads, 2 interleaved) | equal to the reference prefix | passed |
| Streaming reassembly, usage, provider options `use_tf32: 0` | exact | passed |

Bounds from `protected/qwen3.5-0.8b/ortgenai-fp16x-cuda/tolerances.yaml`; nothing under `protected/` changed. Report: [`g2-release-prompts-fixture-semantics/report.md`](g2-release-prompts-fixture-semantics/report.md) (its date is UTC, 2026-10-15).

```sh
RP=model-pipeline/protected/qwen3.5-0.8b/release/release-prompts
<venv: torch==2.14.1+cpu transformers==5.18.0 tokenizers==0.23.2 numpy==2.4.6 safetensors==0.8.0>/bin/python model-pipeline/reference/generation/run_reference.py \
  --model-dir <Qwen/Qwen3.5-0.8B@2fc06364 snapshot> --revision 2fc06364715b967f1860aea9cf38778875588b17 --prompts $RP/prompts-release-v1.jsonl --out <ref> --device cpu --threads 6
dotnet .../Starlings.Qwen35.Fidelity.dll --bundle <snapshot> --variant fp16x --device cuda --prompts $RP/prompts-release-v1.jsonl \
  --reference $RP/reference --out <run> --modes parity,fixture-semantics --threads 4
model-pipeline evaluate model-pipeline/registry/qwen3.5-0.8b/ortgenai-fp16x-cuda/manifest.yaml --fixtures $RP/reference --results <run> \
  --reference-tensors <ref> --out <report> --require-gates G1,G2
```

Wall clock (shared machine, not a benchmark): reference regeneration 307 s; runner 23 s.

