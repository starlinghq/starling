# qwen3.5-0.8b/ortgenai-fp32x-cuda

Evaluator exit **4**, first divergence none: **G1 passed, G2 not-run**. Development fixtures (`prompts-v1.jsonl`, 22 prompts); not the P-030 release set. Hardware and commands: [`../README.md`](../README.md).

Runner: `Starlings.Qwen35.Fidelity -p:UseCuda=true -- --variant fp32x --device cuda --modes parity,semantics` at `c33d63c` (ONNX Runtime GenAI 0.17.1, provider options `use_tf32: 0`, native calls serialized), bundle from the acquisition cache at `2fc06364715b967f1860aea9cf38778875588b17`. `--reference-tensors`: the P-016 CPU reference `step0/` (22 vectors, checked against the fixture digests).

| Check (tolerance key) | Bound | Result | Status |
| --- | --- | --- | --- |
| Template rendering (`discrete.template_rendering`) | exact | 21/21; `tools` rejected as unsupported (excluded by contract) | passed |
| Prompt ids (`discrete.prompt_ids`) | exact | 21/21 | passed |
| Greedy ids (`discrete.greedy_ids`) | exact except listed margins below 1e-3 | 22/22 exact | passed |
| Finish reason | equal | 22/22 | passed |
| Step-0 argmax | equal | 22/22 | passed |
| Step-0 log-probability, full vocabulary | 5e-3 | 1.35e-3 (worst `tools`) | passed |
| Step-0 top-5 log-probability | 2e-3 | 5.9e-4 | passed |
| Cached vs uncached | 2e-4 | 4.7e-5 | passed |
| Concurrent requests | equal to sequential | equal | passed |
| Streaming reassembly, usage, NaN/Inf, provider options | exact / fail / `use_tf32: 0` | ok | passed |
| Chunked vs one-call prefill (`float.chunked_vs_one_call_prefill_logprob_max_abs`) | **not in the protected tolerances file** | 5.9e-5 measured (`long-license` chunks 100 and 128, `multi-turn` chunks 7) | **not-run** |

**Why G2 is not-run.** `protected/qwen3.5-0.8b/ortgenai-fp32x-cuda/tolerances.yaml` does not bind the chunked-prefill key. The fp32x CPU file (1e-4) and the fp16x CUDA file (0.1) do bind it. When a measured check has no bound, the evaluator reports it, and therefore G2, as not-run, and exits 4. That is correct behavior, and nothing was adjusted. Closing the gap needs a maintainer change to protected evidence. In P-016 the fp32x CUDA one-call prefill of the 1,666-token prompt did not fit next to another process on a 16 GB GPU. On this 24 GB GPU it fit, so this run also measured the comparison for the long prompt.

Reproducer: run the commands in `../README.md` for this profile. The evaluator prints `G1 passed, G2 not-run` and names `dotnet-text.chunked_prefill`, so a run with no GPU is enough to inspect the gap (`report.md`, ".NET runtime semantics").

`compare.py` against the P-016 CPU reference (`compare.json`) agrees: 22/22 exact, token match 1.000, step-0 1.35e-3, top-5 5.9e-4, rendering and ids 21/21. These equal the P-016 fp32x CUDA numbers (1.36e-3 / 6.0e-4). Files: `report.md`, `report.json` (unedited), `environment.json`, `semantics.json`, `compare.json`, `run.log`.
