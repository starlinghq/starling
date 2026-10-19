# Batching investigation

Status: merged through PR #126, with the RunPod measurement of 2026-10-18 in section 5 (`docs/evidence/batching-sweep-2026-10-18/`). Everything new is opt-in: with default options every model's outputs are unchanged bit for bit. No capability status was raised, no profile maturity changed, and nothing under `model-pipeline/protected/` or in a tolerances file was edited.

## 1. The plan

This is the reviewed plan, reproduced as given, with light formatting only.

### Option contract shared across workstreams

- `E5EncoderOptions` / BERT path: `MaxBatchTokens` (int?, null = no token cap; padded cells rows x longest row), `SortByLength` (bool, default false).
- `Gliner2ExtractorOptions`: `MaxBatchTokens` (int?, null), `SortByLength` (bool, default false).
- `EmbeddingGemma2EncoderOptions` (+ `MultimodalOptions` text path): `SortByLength` (bool, default false).
- Micro-batcher: `Starlings.Runtime.Onnx.Batching.MicroBatcher<TItem,TResult>` and `MicroBatchingOptions { MaxWait (TimeSpan), MaxRows, MaxTokens (int?), QueueCapacity }`; per-model opt-in property `MicroBatching` (`MicroBatchingOptions?`, null = off) on the E5 and EmbeddingGemma 2 Extensions.AI generator options and on `Gliner2ExtractorOptions`.

Defaults keep today's outputs bit for bit; new behavior is opt-in until benchmarks justify it.

### Phase 0: measure

Add CUDA benchmarks + concurrency sweeps (1/4/16) for e5, EmbeddingGemma 2 text and GLiNER2; mixed-length workloads; batch 1/4/8/16/32; record padding efficiency, peak VRAM/RSS, queue vs inference time; CPU IntraOpNumThreads vs concurrency sweep.

Evidence today (e5 only):

- CPU short: b1 189 items/s, b8 226, b32 142 (regression under default MaxBatchSize 32); b1 at concurrency 4 = 344 (concurrency beats batching on CPU).
- CUDA short: b1 c4 1431, b8 2652, b32 2819 (saturates at b8, ~1.9x over concurrency).
- CUDA long512: b1 150, b8 128, b32 119 (batching loses).

### Phase 1: explicit-batch fixes (no scheduler)

- Per-provider defaults (CPU e5 32 -> ~8, decided after measurement).
- Token-budget chunking (`MaxBatchTokens`) for E5/BERT and GLiNER2 as EG2 has.
- Length-sorted chunking within one call with scatter-back, tested with mixed neighbours.
- CUDA plumbing to evaluate later (IO binding, attention fusion, arena/gpu_mem_limit, keep use_tf32=0).
- Gate: existing padded_batch_vs_single tolerances.

### Phase 2: cross-request micro-batching, opt-in, only adopted if it beats concurrency 4/16

- Bounded queue, max wait, row/token budget.
- Targets `IEmbeddingGenerator` single-string callers and GLiNER2 on CUDA keyed by label set (its CUDA calls are serialized today).
- Scheduler-owned cancellation token; caller cancel drops before dispatch or discards after; never terminates the shared run.
- Outputs depend on neighbouring traffic (within tolerance, not bit-reproducible).
- Experimental capability until a G4 soak passes.

### Phase 3: Kev batched question rows (CPU spike)

- Uncached batched path first (reference `kev_ort.py` 292-305, tolerance `padded_batch_vs_single_probability_max_abs` 1e-5).
- Then cached path with tiled state cache, right padding only, all-ones mask over past+width because GroupQueryAttention derives past length from mask sum.
- Row/token budget (4B: ~0.54 GB KV + 50 MB recurrent state per row at 8k tokens).
- `batched_rows` stays Unsupported pending maintainer decision.

### Phase 4: Qwen 3.5 batched generation (research note only)

GenAI generators already interleave under one lock (`GenAIModel.cs:197`). Static batch issues: finished rows burn compute, one random_seed per batch, per-request cancellation/stop sequences/chunked prefill don't map, left padding through LinearAttention unsafe, no protected batch tolerance.

## 2. What each phase delivered

Five branches were merged with merge commits: `batching/qwen-note`, `batching/kev`, `batching/explicit`, `batching/scheduler`, `batching/bench`. Two files conflicted when `batching/scheduler` was merged on top of `batching/explicit`, `E5Encoder.cs` and `Gliner2ExtractorOptions.cs`. Both sides were kept: the chunk plan with the public `Validate(EncoderInput)`, and the `MaxBatchTokens` check with the `MicroBatching` check. The micro-batchers call the encoders' and extractor's planned paths, so a micro-batched batch is still split by `MaxBatchSize`, `MaxBatchTokens` and `SortByLength`.

### Phase 0: batching sweep (`batching/bench`, wired in the integration commit)

- A `sweep` command in the e5, GLiNER2 and EmbeddingGemma 2 (text) harnesses and their `.Cuda` builds, sharing one engine (`ai-dotnet/benchmarks/Core/BatchSweep.cs`, `SweepPrimitives.cs`, `SweepWorkloads.cs`).
- The grid is workload set (short / long / mixed, `workloads/batching-sweep-v1.json`) x batch 1/4/8/16/32/64 x concurrency 1/4/16. Each cell records items/s, latency p50/p95, padding efficiency, peak RSS and peak VRAM.
- Every output of every timed call is checked against the same item run alone, within the profile's padded-batch tolerance.
- The output format is `starlings-bench-sweep/1` (`model-pipeline/schemas/benchmark-sweep.schema.json`), summarized by `model-pipeline bench sweep`.
- Scripts: `ai-dotnet/benchmarks/run-batching-sweep.sh` and `tools/ci/runpod-batching-bench.sh` (RTX 3090 pod).
- **Integration:** the contract flags are now mapped to the real options:
  - `--sort-by-length` maps to `SortByLength`.
  - `--max-batch-tokens` maps to `MaxBatchTokens`.
  - Any of `--microbatch-wait-ms`, `--microbatch-max-rows` or `--microbatch-max-tokens` turns micro-batching on. Each item is then issued as its own call through the micro-batched entry point: e5 and EmbeddingGemma 2 `GenerateAsync` with one value, GLiNER2 `ExtractAsync`.
- Each run records its configuration label as `sweep.config` (`baseline`, `sorted`, `tokens4096`, `sorted+tokens4096`, `micro-w0-r8`, ...). A micro-batched cell also records the batcher's counters: batches, mean rows, padding, queue time per item and run time per batch.
- The summarizer keeps one table per configuration. It adds a "Batching configurations" table that sets each configuration's best cell beside its single-item callers at each concurrency.
- `run-batching-sweep.sh` (and through it the pod script) now runs these configurations after the baseline.
  - The full grid runs `sorted`, `tokens`, `sorted-tokens` and `micro`.
  - `sorted` runs at 8 rows per ORT run beside the same grid unsorted, because sorting changes nothing when a call is a single run.
  - `micro` runs at MaxWait 0 / MaxRows 8 and at 2 ms / 16, next to a plain batch-1 baseline.
  - `--quick` runs `sorted`, `tokens` and `micro` at 0 ms / 8.
  - On CPU, GLiNER2 has its own in-flight memory guard (32 items).

### Phase 1: explicit-batch fixes (`batching/explicit`)

- A shared internal chunk planner (`ai-dotnet/src/Shared/Batching/BatchPlan.cs`), compiled into E5, BERT, GLiNER2 and EmbeddingGemma 2. It sorts rows by length (stable) when asked and packs chunks greedily within `MaxBatchSize` rows and an optional padded-cell budget. Results are scattered back to input order. Without sorting or a budget, the chunks are the same as before.
- `E5EncoderOptions.MaxBatchTokens` / `SortByLength`, `Gliner2ExtractorOptions.MaxBatchTokens` / `SortByLength`, `EmbeddingGemma2EncoderOptions.SortByLength` (and `EmbeddingGemma2MultimodalOptions.SortByLength` for the text encoder it builds).
- BERT: `BertTextEmbedder.Create` and `BertReranker.Create` take `maxBatchTokens` and `sortByLength`. This is source-compatible but changes the binary signature.
- New capabilities, all Experimental: `batch.token_budget` (E5, GLiNER2) and `batch.sort_by_length` (E5, GLiNER2, EmbeddingGemma 2), and for BERT `BertCapabilities.BatchTokenBudget` and `BatchSortByLength`.
- Not applied: the CPU default change for e5 (`MaxBatchSize` 32 to 8). The RunPod sweep did not support it (section 5).
- Not done: the CUDA plumbing (IO binding, attention fusion, arena / `gpu_mem_limit`). It is still to be evaluated on the pod.

### Phase 2: cross-request micro-batching (`batching/scheduler`)

- `Starlings.Runtime.Onnx.Batching.MicroBatcher<TItem,TResult>` and `MicroBatchingOptions { MaxWait = 0, MaxRows = 8, MaxTokens = null, QueueCapacity = 1024 }`.
  - One dispatcher thread runs one batch at a time, grouped by an optional key.
  - Backpressure: callers wait when the queue is full.
  - Cancellation: the batch runs under a token the batcher owns. A caller that cancels before dispatch is removed; after dispatch, its result is discarded. The batch's token is cancelled only when every caller in it has cancelled.
  - A failure in a batch faults only that batch's callers.
  - Statistics cover batches, items, padding, queue time and run time.
- Opt-in `MicroBatching` on `E5EmbeddingGeneratorOptions`, `EmbeddingGemma2EmbeddingGeneratorOptions` and `Gliner2ExtractorOptions`, plus `Gliner2EntityExtractor.ExtractAsync`.
  - Only one-value `GenerateAsync` calls and `ExtractAsync` calls queue.
  - Each caller's input is validated on its own thread first.
  - GLiNER2 groups by schema (labels in order plus descriptions). The threshold is applied per caller at decode.
- Capabilities `extensions_ai.micro_batching` (e5, EmbeddingGemma 2) and `micro_batching` (GLiNER2), both Experimental.
- New public API, all additions: `E5Encoder.Validate`, `EmbeddingGemma2Encoder.Validate`, `E5TextEmbedder.Tokenizer`, `Gliner2EntityExtractor.ExtractAsync` (2 overloads) and `MicroBatchingStatistics`.

### Phase 3: Kev batched question rows (`batching/kev`)

- An internal, never-default API: `KevBatchOptions`, `KevBatchPlanner`, `KevDecisionModel.RunUncachedBatched` and `RunBatched`. `Decide`, `Score` and `Run` are unchanged, and `batched_rows` stays Unsupported.
- The uncached batched path matches: at most 1.49e-8 from the uncached single runs and 2.83e-7 from the default cached run. The bound is 1e-5.
- ONNX Runtime 1.30.0's GroupQueryAttention refuses the cached batched layout whenever more than one row runs: "batch_size must be 1 when sequence_length > 1 and past context is given". `RunBatched` reports this as `NotSupportedException`, and a test pins the limit.
- On CPU, batching never beat the default cached sequential path (`docs/evidence/kev-batched-rows/README.md`).

### Phase 4: Qwen 3.5 batched generation (`batching/qwen-note`)

- Research note `docs/plan/qwen35-batched-generation.md`, with evidence in `docs/evidence/qwen35-batched-generation/`.
- The managed GenAI 0.17.1 API offers static batching only.
- The qualified graphs feed the attention mask only to GroupQueryAttention. Pad tokens therefore enter the recurrent state, and only equal-length batches are correct.
- Recommendation: no product change, and `batch` stays Unsupported. Run a time-boxed CUDA Engine-API spike on a paged export only after measuring serialized concurrency on RunPod.

## 3. Evidence so far

All numbers below are from a shared i7-5820K workstation (12 logical CPUs) under heavy load from other jobs (load average about 4 to 65). Only the ratios between variants are usable; absolute numbers are not evidence.

**Correctness of the explicit-batch options, CPU, real models, against the protected bounds:**

| Model | Configurations | Against single calls | Against protected | Bound |
| --- | --- | --- | --- | --- |
| e5 | 46 cases × 3 orders × 3 configurations (sorted 8 rows, 1,024-token budget, 16 rows + 2,048 sorted) | max abs 0.0 | 2.33e-7 | 1e-5 |
| GLiNER2 | 108 rows, sorted with a 2,048-token budget | span logits relative 0.0, entities identical | – | 1e-4 |
| EmbeddingGemma 2 | 46 rows of 2 to 1,743 tokens × 2 orders, sorted | max abs 1.49e-7 | 1.94e-7 | 5e-6 |

Padding on the e5 fixture (random order, `MaxBatchSize` 8): sorting cut padded cells from 9,248 to 3,640 for 1,689 real tokens. Padding efficiency rose from 18.3% to 46.4%, with the same number of runs.

**Micro-batching correctness, CPU, real models:**

| Model | Calls | Batches formed | Padding efficiency | Against single calls | Against protected |
| --- | --- | --- | --- | --- | --- |
| e5 | 46 concurrent one-value calls (MaxRows 8, MaxWait 20 ms) | 6 | 38.9% | bit-equal | 2.33e-7 |
| EmbeddingGemma 2 | 24 query rows | 3 | 48.3% | max abs 1.12e-7 | – |
| GLiNER2 | 54 fixture cases | 27, grouped by schema | 92.2% | all 173 entities identical, confidence diff 0 | – |

**Phase 0 sweep smokes, CPU, development numbers only.** On this loaded CPU, concurrency was at least as good as batching in nearly every set. For example, e5 mixed at intra-op 4 ran 32.7 items/s at b1 c4 against 16.2 at b8 c4. Batched-vs-single differences were exactly 0 for e5 and GLiNER2 at every batch size.

**Kev-0.8B batched rows, CPU** (8 mixed questions, faster of 2 runs):

| State tokens | Cached sequential (default) | Uncached batched, 4 rows | Uncached batched, 8 rows |
| ---: | ---: | ---: | ---: |
| 46 | 3.28 s | 6.58 s | 10.13 s |
| 530 | 6.45 s | 31.91 s | 33.47 s |

The cached batched layout is refused by ORT 1.30.0.

**Qwen3.5-0.8B static batch, CPU:**

- Equal-length pairs were bit-identical to single runs.
- With unequal lengths, the shorter row was wrong from step 0 with left padding (max abs 17.3) and from step 1 with right padding.

**Integration smoke of the new configurations (this branch), CPU:** see section 4.

## 4. Integration smoke (local CPU, quick grid)

These numbers come from `run-batching-sweep.sh cpu <out> --quick` on the shared i7-5820K, with `SWEEP_THREADS=2`, the thread sweep skipped and `SWEEP_ARGS="--pool 12 --min-seconds 0.5 --min-calls 3"`. They are development numbers from a loaded machine: they show that the configurations run end to end and give the direction, not the size, of each effect. Raw files are in `/tmp/bsweep-integ`, `/tmp/bsweep-integ2` and `/tmp/bsweep-integ-eg2` (not committed).

**Every run passed its quality check, with one exception.** GLiNER2's first quick main run was killed for memory at long b32 c4. Its long b32 c1 cell had already peaked at 10.2 GB RSS. The CPU in-flight guard for GLiNER2 now defaults to 32 items (`SWEEP_MAX_INFLIGHT_GLINER2`), and the rerun passed.

Batched-vs-single differences:

| Model | Max difference | Notes |
| --- | --- | --- |
| e5 | 0 | every configuration |
| GLiNER2 | 0 | every configuration; entities identical |
| EmbeddingGemma 2 | at most 1.34e-7 | bound 5e-6 |

**Length sorting.** Mixed set, calls of 32 items split into 8-row runs (`--max-batch-size 8`), items/s:

| Model | Unsorted c1 | Sorted c1 | Unsorted c4 | Sorted c4 | Padding efficiency |
| --- | ---: | ---: | ---: | ---: | --- |
| e5 | 9.6 | 21.9 | 16.6 | 31.9 | 0.402 to 0.724 |
| GLiNER2 | 1.82 | 2.26 | skipped by the guard | skipped by the guard | 0.518 to 0.822 |
| EmbeddingGemma 2 | 0.36 | 0.82 | 0.67 | 1.44 | 0.228 to 0.490 |

**Token budget 4096**, mixed set, one ORT run per call apart from the budget's own splits:

- e5: b32 c4 14.6 items/s against 15.0 for the baseline. Padding is unchanged (0.355): the pool's 12 mixed items are short enough that the budget rarely splits a call.
- EmbeddingGemma 2: padding efficiency 0.43 to 0.44 at b8 and b32. Its baseline at the library default (16 rows, 8,192 cells) was 0.226.

**Micro-batching, MaxWait 0 / MaxRows 8**, single-item callers against plain concurrency (baseline b1), items/s:

| Model, set | Concurrency | Plain | Micro-batched | Mean rows per batch | Queue ms per item |
| --- | ---: | ---: | ---: | ---: | ---: |
| e5 short | 4 | 148 | 96 | 2.0 | 15 |
| e5 short | 16 | 222 | 112 | 7.2 | 63 |
| e5 mixed | 4 | 46 | 19 | – | – |
| e5 mixed | 16 | 64 | 11 | – | – |
| GLiNER2 short | 4 | 14.7 | 4.5 | – | – |
| GLiNER2 short | 16 | 17.4 | 4.6 | 5.7 | 1,022 |
| EmbeddingGemma 2 short | 4 | 27.7 | 9.7 | – | – |
| EmbeddingGemma 2 short | 16 | 30.2 | 10.9 | – | – |

On CPU, micro-batching loses clearly to plain concurrency, as the e5 evidence predicted. The batcher runs one batch at a time on one dispatcher thread, while plain callers run in parallel on the intra-op pool. This is not a decision: the adoption gate is CUDA (section 5).

**Library default `MaxBatchSize`, mixed b32 c1:**

| Model | Items/s | Padding efficiency |
| --- | ---: | ---: |
| e5 | 9.5 | 0.352 |
| GLiNER2 | 2.0 | 0.518 |
| EmbeddingGemma 2 | 0.4 | 0.226 |

## 5. RunPod measurement (2026-10-18)

One RTX 3090 pod in EU-CZ-1, CPU quota 31 on an AMD EPYC 7H12, commit `16b02ef`. Raw records, summaries, environment and caveats are in `docs/evidence/batching-sweep-2026-10-18/`. Every measured cell, on CUDA and CPU and in every configuration, stayed within the profile's padded-batch tolerance against the same item run alone. Figures are items/s; "plain concurrency" means single-item calls from 1, 4 or 16 concurrent callers with no batching.

### CUDA (RTX 3090, intra-op 4)

| Workload | e5 | GLiNER2 | EmbeddingGemma 2 |
| --- | --- | --- | --- |
| short, best explicit batch vs best plain concurrency | 8,507 (b64 c4) vs 1,938: 4.4× | 603 (b64 c4) vs 227: 2.7× | 1,279 (b64 c4) vs 203: 6.3× |
| mixed, unsorted 8-row runs vs **sorted** 8-row runs | 482 vs **986** | 110 vs **214** | 84 vs **175** |
| mixed, best plain concurrency | 858 (c16) | 147 (c1) | 142 (c16) |
| long, best batch vs best plain concurrency | 364 sorted vs 351 | 76 vs 61 | 80 sorted vs 69 |
| micro-batching vs plain concurrency, short c4 / c16 | 786 vs 1,319 / 3,902 vs 1,938 | **373 vs 129 / 568 vs 130** | 167 vs 173 / 576 vs 203 |
| micro-batching vs plain concurrency, mixed c4 / c16 | 428 vs 731 / 399 vs 858 | 139 vs 105 / 98 vs 106 | 78 vs 131 / 67 vs 142 |
| micro-batching vs plain concurrency, long c4 / c16 | 262 vs 316 / 298 vs 349 | **69 vs 60 / 74 vs 59** | 59 vs 67 / 65 vs 69 |

### CPU (31-CPU quota, intra-op 16, shared host: ratios only)

| Workload | e5 | GLiNER2 | EmbeddingGemma 2 |
| --- | --- | --- | --- |
| short, best batch vs best plain concurrency | 1,141 (b16 c16) vs 776: 1.5× | plain concurrency best (97 at c16) | 150 (b8 c16) vs 92: 1.6× |
| mixed, best batch vs best plain concurrency | 101 sorted+tokens vs **109** | 12 sorted vs **25** | 8.5 vs **14.6** |
| long, best batch vs best plain concurrency | 58 sorted+tokens vs 36 | 6.5 tokens vs **8.1** | 9.5 vs 6.9 |
| micro-batching vs plain concurrency, c16 | slower: 5.3× short, 3.6× mixed, 1.5× long | slower: 4× short, 5.3× mixed, 2× long | not measured (run cap) |

Intra-op threads against concurrency (e5 mixed, single-item callers): with 16 callers, 8 threads gave 140 items/s, 16 threads 109 and 31 threads 63 (oversubscribed). With one caller, 31 threads were best (42). GLiNER2 shows the same shape.

### What the evidence supports

1. **`SortByLength` is the effective explicit-batch technique for mixed-length traffic on CUDA.** It doubles throughput over unsorted chunking for all three models, and with large calls it beats plain concurrency (e5 1.15×, GLiNER2 1.45×, EmbeddingGemma 2 1.23×). The cost is latency, because the gain needs large calls: p50 of 0.25 s to 5.8 s at b64. Recommendation: keep it opt-in (Experimental) and document it for offline and bulk embedding on CUDA. Whether it becomes a default is a maintainer decision; this run is its first CUDA quality check (all cells within tolerance).
2. **`MaxBatchTokens` alone does not pay.** It was neutral everywhere, and with sorting it was slightly worse than sorting alone (except GLiNER2 mixed). Keep it as a memory bound, not a throughput tool.
3. **Micro-batching passes the adoption gate (beat plain concurrency at both c4 and c16) only for GLiNER2 on CUDA, on short (2.9× / 4.4×) and long (1.15× / 1.25×) inputs.** GLiNER2's CUDA calls are serialized today, and micro-batching turns that queue into batches. On mixed lengths it wins at c4 and loses narrowly at c16. For e5 and EmbeddingGemma 2 it helps short inputs at c16 only, and it loses on mixed lengths (about 0.5×) and on every CPU workload. Recommendation: keep `MicroBatching` opt-in and Experimental; document GLiNER2-on-CUDA as the use case; run a G4 soak with it before any status change.
4. **Why micro-batching loses on mixed lengths:** it queues in arrival order, so a batch pads to its longest item (padding efficiency about 0.6). Its single dispatcher also serializes work that concurrent callers otherwise overlap. The next iteration is length-aware queueing (sort or bucket the pending queue) and more than one batch in flight.
5. **e5 CPU default `MaxBatchSize` stays 32.** With one caller on this host, 32 matched or beat 8 (long b32 26.9 vs 19.5; mixed 27.6 vs 28.8). The earlier i7 figure (b32 25% below b1) did not reproduce, so the change is not justified.
6. **CPU: concurrency, not batching, for mixed traffic.** Size intra-op threads × concurrent callers to the cores: 16 callers at 8 threads beat 16 at 16 threads on a 31-CPU quota.
7. **GPU memory has no admission control.** EmbeddingGemma 2's long b32 c16 cell (16 concurrent runs of 32 rows × 512 tokens) exhausted the 3090's 24 GB in the ORT CUDA arena and aborted the process. The library throws `OnnxRuntimeException`; the harness crashed because the exception escaped a worker thread. An arena limit (`gpu_mem_limit`) or a token-cell admission bound per session (the plan's Phase 1 CUDA plumbing) is the follow-up. The sweep now guards these cells (`SWEEP_MAX_INFLIGHT_CUDA_EG2`), and the EmbeddingGemma 2 CUDA baseline was rerun with the guard (`eg2-cuda-rerun/`, commit `8c9b816`). Every cell passed, and the results agree with the first run: short b64 c4 1,274 vs 221 for plain concurrency (5.8×), long b8 c4 75 vs 70, mixed best with plain concurrency at c16 (144). Only the guarded b32 and b64 cells at c16 were skipped.

### Still open

- **Micro-batcher iteration:** length-aware queueing and more than one batch in flight, then re-measure. Also a G4 soak with micro-batching for GLiNER2 on CUDA.
- **CUDA memory admission:** `gpu_mem_limit` or token-cell admission, measured against the long-input cells that failed.
- **Not measured on the pod:** Kev on CUDA (whether GroupQueryAttention has the same batch limit), Qwen3.5 serialized concurrency and the batch experiment on CUDA, EmbeddingGemma 2 CPU thread sweep and CPU configurations (run cap), and the Phase 1 CUDA plumbing (IO binding, attention fusion).
- **EmbeddingGemma 2 image and video:** the tests skipped locally because their assets are absent; run them where the assets are present.
