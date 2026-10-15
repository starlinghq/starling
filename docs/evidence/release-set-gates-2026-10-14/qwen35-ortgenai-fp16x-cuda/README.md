# qwen3.5-0.8b `ortgenai-fp16x-cuda` on the P-030 release sets

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. The tool was the CUDA build (ONNX Runtime GenAI 0.17.1), with `use_tf32: 0`. The bounds come from `protected/qwen3.5-0.8b/ortgenai-fp16x-cuda/`.

| Gate | Set | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- | --- |
| G1 | tokenizer release (232) | **passed** | ids exact 232/232 | [`g1-tokenizer-release/report.md`](g1-tokenizer-release/report.md) |
| G1 | release prompts (32) | **passed** | template and ids exact 32/32 | [`g1-g2-release-prompts/report.md`](g1-g2-release-prompts/report.md) |
| G2 | release prompts (32) | **not-run** | Greedy ids exact on 30/32. Two divergences are allowed near ties under `equal_until_first_divergence_at_margin_below_0.05`: `oasst1-496a8740` at step 24 (margin 0.00249) and `oasst1-b3321bbb` at step 17 (0.00655). Finish reason and step-0 argmax 32/32. Step-0 top-5 0.0273 (0.06). Step-0 on the reference top 64: 0.0414 (0.15). The runtime-semantics keys and the full-vocabulary bound did not run, as for fp32x. | same |
| G3 | GSM8K, 50 problems | **passed** | exact match **0.64** (32/50; minimum 0.56); no row's correctness differs from the transformers fp32 reference | [`g3-gsm8k/report.md`](g3-gsm8k/report.md) |

Wall clock: release-prompt parity 16.7 s; GSM8K 120 s.
