# Batching sweep on RunPod, 2026-10-18

Phase 0 of the batching investigation (`docs/plan/batching-investigation.md`): the batching sweep of e5, GLiNER2 and EmbeddingGemma 2 text, on CUDA and CPU.

- **Machine:** one on-demand RunPod pod, NVIDIA GeForce RTX 3090 (24 GB, driver 580.126.20, CUDA 13.0) in EU-CZ-1.
- **CPU:** AMD EPYC 7H12, 256 visible CPUs under a CFS quota of 31 (`environment.txt`). The host was shared, with a load average of 40 to 80 during the run. CFS throttling read 0 before and after each provider.
- **Tooling:** `tools/ci/runpod-batching-bench.sh` → `ai-dotnet/benchmarks/run-batching-sweep.sh`. Summaries come from `model-pipeline bench sweep`.
- **Threads:** CUDA ran with intra-op 4. The main CPU grid ran with intra-op 16; the thread sweep covered 1, 2, 4, 8, 16 and 31.

## Runs

| Directory | Commit | What |
| --- | --- | --- |
| `cuda/` | `16b02ef` | Full CUDA grid and batching configurations. 26 of 27 runs passed. `eg2/cuda t4` (the EmbeddingGemma 2 baseline grid) aborted with an ORT CUDA arena allocation failure at long b32 c16 (512 rows of up to 512 tokens in 16 concurrent runs, beyond 24 GB). Cells measured before the abort are kept. |
| `cpu/` | `16b02ef` | Full CPU grid, thread sweep and configurations for e5 and GLiNER2. EmbeddingGemma 2 CPU has the main grid (complete) and the start of the thread sweep. The pod run's 240-minute cap ended it before the remaining EmbeddingGemma 2 thread values and configurations. |
| `eg2-cuda-rerun/` | `8c9b816` | Repeat of the EmbeddingGemma 2 CUDA baseline and default-MaxBatchSize grids with `SWEEP_MAX_INFLIGHT_CUDA_EG2=256`, which skips the cells that exhausted the arena. |

## Files

- `*/summary-*.md` and `.json`: per model × provider × configuration tables, best-configuration lines, the "Batching configurations" comparison and the intra-op thread table.
- `*/raw/*.jsonl.gz`: the `starlings-bench-sweep/1` records. Run `gunzip -k` before `model-pipeline bench sweep`.
- `*/sweep-status.tsv`: per-run exit status.
- `environment.txt`: GPU, CPU and memory inventory of the pod.

## Quality

Every timed output was compared with the same item run alone, against the profile's existing padded-batch tolerance:
- e5 and EmbeddingGemma 2: `padded_batch_vs_single_embedding_max_abs`.
- GLiNER2: span logits relative, plus identical entities. For micro-batched GLiNER2, entities and confidence, because `ExtractAsync` does not expose logits.

Every measured cell passed, on CUDA and CPU, for every configuration: sorted, token-budgeted and micro-batched. The worst CUDA embedding difference was 2.6e-7 against 5e-6 (EmbeddingGemma 2) and 1e-5 (e5).

## Caveats

- **Sample sizes:** cells with large batches have few calls (5 to 16), so their p95 is not reported and the medians are indicative.
- **CPU noise:** the CPU figures come from a shared host and are noisier than CUDA. Use their ratios between configurations, not their absolute values.
- **GPU scope:** the results cover the RTX 3090 only, the GPU the CUDA contracts cover.
- **Cost:** about $2.11 in pod time ($2.03 for the 243-minute main run, $0.08 for the 9-minute rerun), plus $0.34 for an earlier quick smoke whose results were not kept.
