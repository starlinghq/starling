# CPU prefill in 1,024-token chunks against one call (backlog B2)

Evidence for moving the CPU default of `Qwen35ChatOptions.PrefillChunkTokens` from one call to 1,024-token chunks (`Qwen35ChatOptions.DefaultCpuPrefillChunkTokens`). With that value a prompt of at most 1,024 tokens is still prefilled in one call, so only longer prompts take a different execution path. Produced on 2026-10-26 on Linux x64 (Ubuntu 24.04.5, Intel Xeon at 2.80 GHz, 4 vCPUs shared with other processes) in a cloud container with no GPU, with `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 (CPU), .NET 10.0.0 and `MALLOC_MMAP_THRESHOLD_=65536`. The fidelity runner was built in Release from branch `postponed/genai-guidance` at `c3b8e06` (the `source_revision` in each `environment.json`); the runner passes `--prefill-chunk` straight to `GenerationOptions.PrefillChunkTokens`, so the library change in this PR does not alter what it ran.

**Model files.** The fp32x variant, the tokenizer and the chat template, downloaded anonymously from the pinned hosting revision `95a690c1774f144957cb7b76986a9126e8b1bb5f`; all five SHA-256 digests equal the lock in `Qwen35ModelAssets`. No weights are committed.

## Result

| Comparison | Prompts | Generated ids and finish reason | Step-0 log-softmax (248,320 floats) |
| --- | --- | --- | --- |
| Development fixture (G2), one call against `--prefill-chunk 1024`, full `max_new_tokens` | 22 (one above 1,024 tokens: `long-license`, 1,666) | equal on 22/22, and equal to the reference on 22/22 | byte-identical on 22/22 |
| Proposed long-context fixture, `long-4096` (4,112 tokens, 32 new tokens; not a G2 fixture) | 1 | equal, and equal to the reference | byte-identical |
| Alternating timing series, `long-license` with 8 new tokens, one call, 1,024 and 512 (4 runs each) | 1 | equal on all 12 runs | byte-identical on all 12 runs |

The comparison is exact, so no tolerance is used for it. The profile's binding key for this comparison is `float.chunked_vs_one_call_prefill_logprob_max_abs` (1e-4); the 1,024-token chunks are at 0 on both prompts that exercise them.

The 21 development prompts and all 32 release prompts are at most 286 and 92 tokens, so with the new default they run in one call as before.

## G2 with the proposed default

`model-pipeline evaluate` on the runs with `--prefill-chunk 1024` (reports in `gate-reports/`):

| Fixture | G1 | G2 | Notes |
| --- | --- | --- | --- |
| Release prompts (32), with the hosted full step-0 reference vectors (`--reference-tensors`, every file checked against `hosted-reference.json`) | passed | passed | greedy ids 32/32 exact; full-vocabulary step 0 9.19e-4 (bound 5e-3); cached vs uncached 2.86e-5 (2e-4); chunked prefill 2.1e-5 (1e-4); concurrency equal to sequential |
| Development prompts (22) | passed | not-run | every check that ran passed: greedy ids 22/22 exact, step-0 top 5 6.2e-4 (2e-3), cached vs uncached 3.4e-5, chunked prefill 2.1e-5, concurrency equal. `float.step0_logprob_max_abs` is not-run because the development fixture's full reference vectors are not hosted; the 64 reference ids give 1.01e-3 (5e-3). |
| Development prompts, one call (baseline, parity only) | passed | not-run | the same parity numbers; no `semantics.json` in this run |

## Tests on the real model with the new default

With `STARLING_QWEN35_SNAPSHOT` set to the downloaded directory, on the same host after the library change: `tests/Starlings.Models.Qwen35.Tests` 402 passed, 8 skipped, 0 failed (8 min 13 s); `tests/Starlings.Models.Qwen35.Extensions.AI.Tests` 251 passed, 2 skipped, 0 failed (8 min 8 s). These go through `Qwen35ChatOptions` defaults, so prompts above 1,024 tokens in them now prefill in chunks.

## Smaller chunks are not bit-identical on this host

The runtime-semantics checks in the same runs use small chunks. On this host they were not all bit-identical, unlike the P-016 run on the M0 workstation (`docs/evidence/p016-qwen35/README.md`, "bit-identical for chunks of 128, 100 and 7") and the release-set run of 2026-10-14 (chunked prefill 0):

| Run | Case / chunk | Step-0 max abs log-probability difference against one call | Tokens equal |
| --- | --- | ---: | --- |
| development | `long-license` / 128 | 0 | yes |
| development | `long-license` / 100 | 2.10e-5 | yes |
| development | `multi-turn` / 7 | 0 | yes |
| release | `oasst1-330fc1cf` / 64 | 2.10e-5 | yes |
| release | `oasst1-330fc1cf` / 32 | 2.10e-5 | yes |
| release | `oasst1-3c5b52fd` / 7 | 1.72e-5 | yes |

All are within the 1e-4 tolerance. Bit-identity therefore depends on the chunk size and the host; the claim this evidence supports is only that 1,024-token chunks were bit-identical on the two long prompts measured here. Library docs that said CPU chunks are bit-identical in general now say this.

## Prefill time and memory (exploratory; shared host, not a benchmark)

`timing-long-license.log`: one process per run, alternating modes, the 1,666-token prompt, 8 new tokens.

| Mode | Runs | Median prefill | Min | Max | Median / one call |
| --- | ---: | ---: | ---: | ---: | ---: |
| one call | 4 | 20,075 ms | 18,378 ms | 20,246 ms | 1.000 |
| 1,024-token chunks | 4 | 18,199 ms | 18,107 ms | 19,475 ms | 0.907 |
| 512-token chunks | 4 | 18,801 ms | 17,934 ms | 20,898 ms | 0.937 |

In the full parity runs the same prompt took 18,629 ms (one call) and 18,839 ms (1,024). The 4,112-token prompt took 62,048 ms in one call and 48,412 ms in 1,024-token chunks. On this host 1,024-token chunks were not slower than one call; P-016's "chunks were slower" came from smaller chunks (the 7-token chunks above took 2.2 s and 3.0 s against 0.9 s and 1.0 s). G5's `long-prefill` workload (`ai-dotnet/benchmarks/workloads/qwen35-v1.json`, the same 1,666-token prompt) goes through the library default, so it now uses two chunks; its time-to-first-token budget (ratio 1.10 to Python GenAI, `performance-budgets.yaml`) was not rerun here, because G5 needs the quiet qualification host.

Memory follows from GenAI computing logits for every position it is given: prompt tokens x 248,320 x 4 bytes. One call needs 1.65 GB for 1,666 tokens, 4.08 GB for 4,112 and 8.14 GB at the 8,192-token context; 1,024-token chunks bound it to 1.02 GB. Resident memory was not measured. The 8,213- and 16,391-token long-context prompts were not run in one call, because their logits buffer does not fit in this container next to the weights.

## Files

- `runs/<run>/`: raw runner output. `stdout.log` (the runner's per-prompt lines, plus a last line with the modes, the UTC start and end and the exit code), `environment.json`, `results.jsonl`, `semantics.json` where the run had a semantics mode, and `step0.sha256` (SHA-256 of each `step0/<id>.f32`; the vectors themselves, 1 MB each, are not committed). Paths in them are local to the container. Runs: `dev-one-call`, `dev-chunk1024` (with `--modes semantics`), `lc4096-one-call`, `lc4096-chunk1024`, `release-chunk1024` (with `--modes fixture-semantics`).
- `runs/timing/`: `stdout.log` of the 12 timing runs.
- `compare-dev.log`, `compare-long-4096.log`: per-case comparison of a one-call and a chunked run (`scripts/compare_runs.py`).
- `timing-long-license.log`: `scripts/timing_summary.py` over the timing runs.
- `gate-reports/`: `model-pipeline evaluate` reports and console logs.

## Commands

```sh
B=<snapshot dir>; F=ai-dotnet/tools/Starlings.Qwen35.Fidelity/bin/Release/net10.0/Starlings.Qwen35.Fidelity.dll
DEV=model-pipeline/protected/qwen3.5-0.8b/reference; RP=model-pipeline/protected/qwen3.5-0.8b/release/release-prompts
dotnet $F --bundle $B --variant fp32x --device cpu --prompts model-pipeline/reference/generation/prompts-v1.jsonl \
  --reference $DEV --out runs/dev-one-call --modes parity
dotnet $F ... --out runs/dev-chunk1024 --modes parity,semantics --prefill-chunk 1024
# long-4096: the fixture row of model-pipeline/proposed/qwen3.5-0.8b/long-context/generation.jsonl as both prompts and reference
dotnet $F ... --out runs/lc4096-one-call --modes parity
dotnet $F ... --out runs/lc4096-chunk1024 --modes parity --prefill-chunk 1024
# timing: the long-license row of prompts-v1.jsonl with max_new_tokens 8; per round: one call, --prefill-chunk 1024, 512
dotnet $F --bundle $B --variant fp32x --device cpu --prompts $RP/prompts-release-v1.jsonl --reference $RP/reference \
  --out runs/release-chunk1024 --modes parity,fixture-semantics --prefill-chunk 1024
model-pipeline evaluate model-pipeline/registry/qwen3.5-0.8b/ortgenai-fp32x-cpu/manifest.yaml --fixtures $RP/reference \
  --results runs/release-chunk1024 --reference-tensors <hosted step0 vectors> --out <report> --require-gates G1,G2
python3 -I scripts/compare_runs.py runs/dev-one-call runs/dev-chunk1024 $DEV/generation.jsonl
```

## Scope

- CUDA: unchanged (512-token chunks), and no GPU was used.
- G5 was not rerun (see above), and no resident-memory measurement was made.
- The matched Python G5 runner (`model-pipeline/reference/bench/bench_qwen35_matched.py`) and `docs/benchmarks/PROTOCOL.md` follow the C# default (1,024-token chunks on CPU), so a G5 run keeps comparing the same execution path.
- The QwenChat profiles (`Starlings.Models.QwenChat`) keep their one-call default; this change is limited to `Qwen35ChatOptions`.
