# R8: Qwen3.5-0.8B long context above 1,666 tokens (fp32x, CPU)

This is the evidence for the capability `long_context.over_1666_tokens`, which is untested today (`Qwen35Capabilities`, profile `qwen3.5-0.8b/ortgenai-fp32x-cpu`), and for workstream R8 of the production-readiness plan. It was produced on 2026-10-13 from repository commit `fb27658` plus the R8 working tree. No gate, maturity or tolerance was changed. The tolerances quoted below are read from `model-pipeline/protected/qwen3.5-0.8b/ortgenai-fp32x-cpu/tolerances.yaml`.

## 1. Setup

- Host: the P-016 workstation (i7-5820K, 12 threads, 94 GB RAM, Linux x86-64), CPU only. The machine was shared with other heavy jobs: the load average was 15 to 44, and `systemd-oomd` was active, with other processes holding 57 to 83 GB. All timings are exploratory.
- Prompts: `model-pipeline/reference/generation/prompts-long-v1.jsonl` (SHA-256 `bb1eb0b0...`), written by `make_long_prompts.py`. Each prompt is one user message: a prefix of the repository's own documentation at `fb27658` (`docs/plan/fast-faithful-ai-dotnet.md`, then `docs/m0/generation-backend-check.md`, `p023-tokenizer-source-audit.md`, `browser-wasm-spike.md`, `gliner2-p014.md`), cut at a line boundary, followed by one question. The three prompts have 4,112, 8,213 and 16,391 tokens. Each generates 32 greedy tokens.
- Reference: `run_reference.py` (transformers 5.18.0, fp32, CPU, 6 threads, greedy) on the pinned snapshot `2fc06364...`, in the environment of `requirements.txt`. The run's environment is in `environment-reference.json`; the run was killed before it wrote that file, so it was rebuilt afterward with the runner's own fields (see section 4). Reference time per prompt: 89 s, 145 s and 378 s.
- Candidate: `ai-dotnet/tools/Starlings.Qwen35.Fidelity` with `--modes parity,longcontext --prefill-chunk 128 --threads 6`, on the fp32x bundle (the `ortgenai-fp32x` files of the P-016 build, digests as in `docs/evidence/p016-qwen35/README.md` section 2). The environment is in `environment-cpu-fp32x.json`. The candidate is fed the reference prompt ids, and the owned template and tokenizer are checked separately. A prefill chunk of 128 keeps the logits buffer at 127 MB; P-016 showed chunked and one-call prefill to be bit-identical on CPU.
- Proposed fixture: `model-pipeline/proposed/qwen3.5-0.8b/long-context/` (`PROPOSED.md` there). The CPU test `CpuModelTests.LongContextGreedyAndLogitsEqualTheReference` reads it.

## 2. Results

`compare-cpu-fp32x.json` (from `compare.py`) and `longcontext-cpu-fp32x.json` (from the runner):

| Prompt | Tokens | Owned rendering and ids | Greedy 32 tokens | Smallest reference margin | Step-0 argmax | Step-0 max abs dlogp (all / top 5) | Chosen-token logprob vs reference (max abs, 32 steps) | Cached vs uncached (max abs dlogp; steps 2, 16, 32) |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
| `long-4096` | 4,112 | equal | exact | 0.038 | equal | 6.1e-4 / 2.2e-4 | 2.0e-4 | 4.0e-5, 5.9e-5, 4.6e-5 |
| `long-8192` | 8,213 | equal | exact | 0.042 | equal | 6.6e-4 / 1.7e-4 | 1.9e-4 | 5.5e-5, 1.2e-4, 1.1e-4 |
| `long-16384` | 16,391 | equal | exact | 0.129 | equal | 7.6e-4 / 2.2e-4 | 1.7e-4 | 1.2e-4, **4.1e-4**, 2.8e-4 |

Against the binding fp32x CPU tolerances:

| Key | Bound | 4,112 | 8,213 | 16,391 |
| --- | --- | --- | --- | --- |
| `discrete.template_rendering`, `discrete.prompt_ids` | exact | pass | pass | pass |
| `discrete.greedy_ids`, `discrete.finish_reason`, `discrete.step0_argmax` | exact or equal | pass | pass | pass |
| `float.step0_logprob_max_abs` | 5e-3 | 6.1e-4 | 6.6e-4 | 7.6e-4 |
| `float.step0_top5_logprob_max_abs` | 2e-3 | 2.2e-4 | 1.7e-4 | 2.2e-4 |
| `float.cached_vs_uncached_logprob_max_abs` | 2e-4 | 5.9e-5 | 1.2e-4 | **4.1e-4: exceeds** |
| `discrete.streaming_reassembly`, `discrete.usage` | exact | pass | pass | pass |

- Streaming reassembly and usage hold on all three prompts (`stream_text_equals_decode`, `usage_consistent` in the runner's results).
- **Cached vs uncached at 16,391 tokens exceeds the protected bound.** The worst difference is 4.1e-4 at step 16, against a bound of 2e-4. In every check, the uncached argmax equals the cached token. At 16,391 tokens, the chosen token's log-probability agrees with the reference within 1.6e-5 uncached (1.0e-4 over all nine checks at the three lengths) and within 1.7e-4 cached, over all 32 steps. So the largest differences are in the low-probability tail, and the selected tokens are unaffected. The difference grows with length: the maximum over the checked steps is 5.9e-5 at 4,112 tokens, 1.2e-4 at 8,213 and 4.1e-4 at 16,391. P-016 measured 3.4e-5 at 12 steps of a 22-token prompt. The cause was not investigated. Plausible causes are fp32 accumulation order in the full-attention layers (128-token chunks for the cached prompt, one chunk boundary layout for each uncached prefix) or in the linear-attention recurrent state. The bound itself is the maintainer's to keep or change.
- Exploratory timings on the loaded machine, fp32x CPU with 6 threads and 128-token chunks: prefill 41 s, 71 s and 159 s. In the longcontext mode, with the second process gone, decode of 32 tokens took 3.9 s, 5.9 s and 4.9 s. The parity-mode timings in the runner's `results.jsonl` (108 s to 351 s prefill) overlapped with a second runner process (section 4) and are not representative.

- **Memory with default options (finding, not fixed here).** On CPU, `Qwen35ChatOptions.PrefillChunkTokens` defaults to a one-call prefill. GenAI computes fp32 logits for every prompt position, about 1 MB per token, so a default request with a 16,391-token prompt allocates about 16 GB of logits, and a 32k prompt about 32 GB. The runs above set 128-token chunks explicitly. A CPU default chunk size (or a cap) is proposed for the maintainer. It would change only memory use, not results: P-016 measured chunked and one-call prefill to be bit-identical on CPU.

## 3. Coverage and limits

- Covered: CPU fp32x, prompts of 4,112 to 16,391 tokens, 32 greedy tokens after the prompt. The pinned `genai_config.json` allows `max_length` 262,144, so the limit here was CPU time and memory on the shared machine, not the model.
- Not covered: CUDA (no GPU was used, as instructed), fp16x, prompts above 16,391 tokens, outputs longer than 32 tokens, and sampling at long context.
- 32,844 tokens: no reference exists. The transformers reference was killed twice (exit 137, SIGKILL) on a 32,844-token prompt while it held 6 to 8 GB, under machine-wide memory pressure from other jobs. Following the stop rule, no third attempt was made. The prompt was dropped from `prompts-long-v1.jsonl`. It can be regenerated with `make_long_prompts.py --targets 32768`, adding `docs/m0/embeddinggemma2-*.md` as further sources if needed.

## 4. Failures during the runs

1. Reference, first run: the four-prompt run was killed with SIGKILL on the fourth (32,844-token) prompt after it had written the other three results. `run_reference.py` writes `environment.json` only at the end, so `environment-reference.json` and the fixture's environment were rebuilt from the same fields: packages read from the same environment, and snapshot digests recomputed and equal to the P-016 reference. Thread count 6 and the kill are recorded in its `note`.
2. Reference, second run: the 32,844-token prompt alone was killed the same way.
3. Candidate, first run: with `--prefill-chunk 512`, the `dotnet run` host was killed (exit 137). Its child process kept running and was found only when the second run started, so for a few minutes both runners shared the CPU and wrote to the same log, before the orphan was stopped. The second run, with 128-token chunks, completed. Every number above comes from the second run's output files. The parity prefill timings overlapped with the orphan.

## 5. Proposed capability status

The evidence supports moving `long_context.over_1666_tokens` from `untested` to `experimental` for fp32x CPU only, with the stated range (prompts up to 16,391 tokens). The exception is that cached versus uncached logits exceed the protected bound at 16,391 tokens. That limitation should be stated wherever the status changes. The manifest and `Qwen35Capabilities` are not changed in this pull request; the decision is the maintainer's (see the PR description). CUDA and fp16x stay `untested`.

## 6. Files

- `compare-cpu-fp32x.json`: `compare.py` output (reference `<r8>/ref-long`, candidate `<r8>/cs-long`).
- `longcontext-cpu-fp32x.json`: per prompt, the greedy ids, chosen log-probabilities against the reference, the cached-versus-uncached checks and timings.
- `environment-cpu-fp32x.json`, `environment-reference.json`: the candidate and reference environments.
- Step-0 vectors are not in the repository. Their SHA-256 digests (reference) are: `long-4096` `0fe1104f...bc12`, `long-8192` `934662fb...418f` and `long-16384` `abe7ee7d...fc49`. The full digests are in the proposed fixture's `environment.json`.
