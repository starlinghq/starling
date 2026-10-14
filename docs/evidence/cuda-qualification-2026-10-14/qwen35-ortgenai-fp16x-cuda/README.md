# qwen3.5-0.8b/ortgenai-fp16x-cuda

Evaluator exit 0, first divergence none, **G1 passed, G2 passed** on development fixtures (`prompts-v1.jsonl`, 22 prompts). This is not the P-030 release set. Hardware and commands are in [`../README.md`](../README.md).

The runner was `Starlings.Qwen35.Fidelity -p:UseCuda=true -- --variant fp16x --device cuda --modes parity,semantics` at `c33d63c` (ONNX Runtime GenAI 0.17.1, `use_tf32: 0`, serialized native calls). The fp16x bundle was downloaded for this run from `Starlinghq/qwen3.5-0.8b-ortgenai` at `95a690c1774f144957cb7b76986a9126e8b1bb5f` without a token. Its digests equal `SHA256SUMS` and the lock: `genai_config.json` `ded164b6…`, `model.onnx` `26895023…`, `model.onnx.data` `47f6ee15…`. `--reference-tensors` was the P-016 CPU reference `step0/`.

| Check | Bound | Result | Status |
| --- | --- | --- | --- |
| Template rendering, prompt ids | exact | 21/21 each; `tools` excluded by contract | passed |
| Greedy ids | equal until the first divergence, which must be at a reference margin below 0.05 | 19/22 exact. Divergences: `code-sql` step 112 (margin 0.0019), `special-think-tool` step 95 (0.0027), `thinking-on` step 76 (0.0004) | passed |
| Finish reason | equal when ids equal | 22/22 | passed |
| Step-0 argmax | equal | 22/22 | passed |
| Step-0 log-probability, full vocabulary | 0.15 | 0.061 (worst `thinking-on`) | passed |
| Step-0 top-5 log-probability | 0.06 | 0.041 | passed |
| Cached vs uncached | 0.1 | 0.061 | passed |
| Chunked vs one-call prefill | 0.1 | 0.049 | passed |
| Concurrent requests, streaming, usage, NaN/Inf, provider options | | ok | passed |

The near-tie set differs from P-016 (RTX 4060 Ti: `ru-de-mix`, `special-think-tool`, `special-broken`, `thinking-on`). Here `ru-de-mix` and `special-broken` match the reference, and `code-sql` diverges at margin 0.0019. Every divergence is below the bound's margin. fp16 accumulation order differs between GPUs, so near-tie divergences are expected to vary with hardware. This is why the profile binds "first divergence at a small margin" rather than exact ids.

`compare.py` (`compare.json`): 19/22 exact, token match 0.955, step-0 0.061, top-5 0.041. Files: `report.md`, `report.json` (unedited), `environment.json`, `semantics.json`, `compare.json`, `run.log`.
