# Kev batched question rows: CPU spike, 2026-10-17

Phase 3 of the batching investigation: can several of a request's question rows run as one graph run on CPU, and does that pay? Capability `batched_rows` stays **Unsupported** (`KevCapabilities`). This note gives the maintainer the evidence to decide.

## What was built

Internal API on `KevDecisionModel`, off the default path (`Decide`, `Score` and `Run` are unchanged):

- `KevBatchOptions` (internal): `MaxRows` (default 8), `MaxTokens` (padded cells per run, rows x (past + longest width), default 32,768, null for no cap; a single row over the cap runs alone), `SortByLength` (default false; results are scattered back to question order either way). `KevBatchPlanner` groups rows under that budget.
- `RunUncachedBatched`: each question as one full causal row (state then branch), right-padded with `<|endoftext|>` under a zero mask at position 0, as `model-pipeline/reference/kev/kev_ort.py` (lines 292 to 305) builds its padded batches.
- `RunBatched`: the state runs once; its cache is tiled along the batch dimension (ONNX Runtime does not broadcast a batch-1 cache); branches are right-padded only, pads take continuing positions, and the attention mask is all ones over past + width, because GroupQueryAttention derives each row's past length from the mask sum. Never left-padded: a leading pad would enter the DeltaNet convolution and recurrent state before the real tokens.

Tiled-cache size per row (why the token budget exists): Kev-0.8B 24 KiB of attention cache per past token (6 attention layers x K and V x 2 heads x 256 x fp32) plus 18.9 MB of DeltaNet recurrent state; Kev-4B 64 KiB per past token plus 50.3 MB. At an 8,192-token state that is 201 MB (0.8B) and 537 MB (4B) of attention cache per row.

## Results

Setup: Kev-0.8B, the locked bundle (graph digest checked against the lock), ONNX Runtime 1.30.0 CPU EP, .NET 10.0.100, Linux x64, Intel Core i7-5820K (6 cores, 12 threads), `IntraOpNumThreads` 4. The machine was shared with other jobs (load average 9 to 65 during the runs), so timings are **contended**; the ratios between variants measured back to back are what matter. Kev-4B was not run (memory).

### 1. Cached batched rows do not run on ONNX Runtime 1.30.0

With more than one row per run, the graph fails in the first full-attention layer:

```
GroupQueryAttention node '/model/layers.3/attn/GroupQueryAttention':
batch_size must be 1 when sequence_length > 1 and past context is given.
```

GroupQueryAttention accepts a non-empty past with more than one new token only at batch 1 (a "subsequent prompt"), and cached batched rows are exactly that shape. Nothing in the inputs avoids it: the alternatives either recompute the state (the uncached layout), run one token per step at batch B (each step reads all weights for one token per row, so it is slower than one prefill per row), or pack branches into one batch-1 sequence (branches would see one another through attention and the DeltaNet state, breaking question isolation). `RunBatched` turns the error into `NotSupportedException`; with `MaxRows = 1` it is the sequential path bit for bit. `CachedBatchedRowsAreRefusedByGroupQueryAttention` pins the limit, so a runtime that lifts it shows up as a failing test. CUDA EP was not checked.

### 2. Uncached batched rows match single rows

`BatchedRowsTests.BatchedRowsMatchSingleRows`: 40 records (the 9 request goldens, the 30 records of `model-pipeline/reference/corpus/kev-decisions-v1.jsonl`, one record that puts a 2-option question next to one with 8 long options), 81 questions, 23 records with several rows; three budgets (all rows in question order; 4 rows sorted by length; 1,024-token cap sorted).

| Check | Bound | Max abs probability |
| --- | --- | --- |
| Uncached batched vs `RunUncached` (single rows) | `padded_batch_vs_single_probability_max_abs` 1e-5 | 1.49e-8 |
| Uncached batched vs `Run` (the default cached path) | `cached_vs_uncached_probability_max_abs` 1e-5 | 2.83e-7 |
| Short row next to the long row vs without it, and vs alone | 1e-5 | 0 |

Bounds from `model-pipeline/protected/kev-0.8b/onnx-fp32/tolerances.yaml` (read only).

### 3. Batching is slower on CPU

8 questions of mixed width per record (2-option, 4-option, 5-option and 6 long options; branches 16 to 155 tokens, 432 in total), each variant run twice after a warm-up, the faster run reported ([`latency-kev-0.8b-cpu.json`](latency-kev-0.8b-cpu.json)):

| State tokens | Cached sequential (`Run`, default) | Uncached sequential | Uncached batched, 4 rows sorted | Uncached batched, 8 rows sorted |
| ---: | ---: | ---: | ---: | ---: |
| 46 | **3.28 s** | 5.10 s | 6.58 s | 10.13 s |
| 134 | **3.86 s** | 9.51 s | 10.87 s | 14.74 s |
| 530 | **6.45 s** | 28.97 s | 31.91 s | 33.47 s |

On this CPU the model is compute-bound even at these sizes (the cached path runs about 480 tokens in 3.3 s, about 150 tokens/s on 4 threads, with no batching to amortize), so a batch saves no per-run overhead worth having and pays for every padded cell: 8 rows at a 46-token state are 1,608 padded tokens for 800 real ones, and take twice as long. The uncached layout also recomputes the state once per question, which the default path does once per request. Neither batched layout beats the default cached sequential path anywhere measured.

## Recommendation

Keep `batched_rows` **Unsupported**. On CPU, batching cannot pay: the cached layout is refused by the runtime and the uncached layout is slower than the default path at every size measured. Batched rows could matter on CUDA, where per-run overhead is larger relative to compute. That needs a CUDA EP check of the GroupQueryAttention limit first (RunPod), and an uncached-batched-vs-cached-sequential comparison there. A different attention export (for example MultiHeadAttention with an explicit past and a per-row mask) would also remove the limit, but it is a new graph with its own fidelity evidence.

## Reproduce

```sh
export STARLING_KEV_08B_BUNDLE=/path/to/kev-0.8b-bundle
dotnet build ai-dotnet/tests/Starlings.Models.Kev.Tests
cd ai-dotnet/tests/Starlings.Models.Kev.Tests/bin/Debug/net10.0
STARLING_KEV_BATCH_BENCH=/tmp/latency.json ./Starlings.Models.Kev.Tests -class Starlings.Models.Kev.Tests.BatchedRowsTests -showLiveOutput
```

The latency test runs only when `STARLING_KEV_BATCH_BENCH` names an output file; the parity tests skip without a bundle.
