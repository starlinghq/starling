# qwen3.5-0.8b `ortgenai-fp32x-cuda` on the P-030 release sets, and the development-fixture G2 rerun

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. `Starlings.Qwen35.Fidelity` was built with `-p:UseCuda=true` (ONNX Runtime GenAI 0.17.1 CUDA), with provider options `use_tf32: 0`, which the evaluator checks.

| Gate | Set | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- | --- |
| G1 | tokenizer release (232) | **passed** | ids exact 232/232 (device-independent; evaluated against this manifest) | [`g1-tokenizer-release/report.md`](g1-tokenizer-release/report.md) |
| G1 | release prompts (32) | **passed** | template and ids exact 32/32 | [`g1-g2-release-prompts/report.md`](g1-g2-release-prompts/report.md) |
| G2 | release prompts (32) | **not-run** | greedy ids exact 32/32; finish reason and step-0 argmax 32/32; step-0 top-5 4.73e-4 (2e-3); step-0 on the reference top 64: 6.06e-4 (5e-3). The runtime-semantics keys and the full-vocabulary step-0 bound did not run, for the reasons in the CPU README ([`../qwen35-ortgenai-fp32x-cpu/`](../qwen35-ortgenai-fp32x-cpu/README.md)). | same |
| G3 | GSM8K, 50 problems | **passed** | exact match **0.64** (32/50; minimum 0.60); no row's correctness differs from the transformers reference | [`g3-gsm8k/report.md`](g3-gsm8k/report.md) |

## Development-fixture G2 rerun (not a release set)

The earlier CUDA evidence (`docs/evidence/cuda-qualification-2026-10-14/`) left G2 `not-run`, because this profile's tolerances had no `float.chunked_vs_one_call_prefill_logprob_max_abs`. The key now exists: it was accepted on 2026-10-14 with the fp32x CPU value 1.0e-4. I reran on the development fixtures (`prompts-v1.jsonl`, `protected/qwen3.5-0.8b/reference`, `--modes parity,semantics`), with the full step-0 reference vectors from the P-016 CPU reference run, which the evaluator checked against the fixture digests: **G1 passed, G2 passed**, exit 0. Step-0 log-probability over the full vocabulary 1.35e-3 (5e-3); cached vs uncached 4.67e-5 (2e-4); chunked vs one-call prefill 5.91e-5 (1e-4); concurrent requests equal to sequential. The report is at [`g1-g2-development-fixtures-rerun/report.md`](g1-g2-development-fixtures-rerun/report.md). These are development fixtures, so the manifest's G2 entry stays `checklist`, with a note.

Wall clock: release-prompt parity 23.5 s; GSM8K 353 s; development rerun 51.8 s.
