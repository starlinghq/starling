# Qwen 3.5 batched generation: research note (batching investigation, phase 4)

Status: research only, 2026-10-17. Nothing in this note changes product code, a capability status or a tolerance. `batch` stays Unsupported on every Qwen3.5 profile (`Qwen35Capabilities`). The CPU spike behind the numbers is in `docs/evidence/qwen35-batched-generation/`.

## Summary

- **Static batching on the qualified bundle gives wrong results.** With ONNX Runtime GenAI 0.17.1 and the qualified `ortgenai-fp32x` bundle on CPU, a batch of 2 is bit-identical to single runs only when both prompts have the same length (step-0 logits max abs 0, all greedy ids equal). When lengths differ, the shorter row is wrong from step 0 with left padding (step-0 logits off by up to 17) and from step 1 with right padding. The longer row stays exact. The fp32x and fp16x graphs feed the attention mask only to GroupQueryAttention: the Gated DeltaNet ops (`CausalConvWithState`, `LinearAttention`) have no mask or length input, so pad tokens enter the recurrent state. GenAI 0.17.1 also sets the decode-step mask to all ones.
- **What GenAI 0.17.1 offers.** The managed 0.17.1 binding (the only one Starling references) offers static batching only, through `batch_size` on one `Generator`. A batched generator gets one prefill call, one `max_length`, one RNG and one cancellation unit for the whole batch. Finished rows keep running. The native 0.17.1 library also exports a continuous-batching Engine C API. It has per-request turns, per-turn seed, stop strings, max tokens and cancellation. The managed binding does not wrap it. For a hybrid model like Qwen3.5, it needs a different, paged export (PagedAttention plus varlen Gated DeltaNet). The 0.17 builder only produces that export for the CUDA provider.
- **Recommendation: do not batch Qwen3.5 in this version.** On CPU, keep concurrent generators. Concurrency is already Supported and equal to sequential, and a batch of 2 equal-length prompts was only about 7% faster than two concurrent requests on a loaded host. On CUDA, where generators are serialized and concurrency gives no parallelism, batching is worth a time-boxed RunPod spike of the Engine API on a paged CUDA export (section 6). Adopt it only if it beats serialized concurrency at 4 and 16 by a clear margin. It would ship as a new Experimental profile with its own maintainer-set batch tolerances, and the Stable profiles would not change.

## 1. Starting point

- `TextGenerator` (Starlings.Generation.OnnxGenAI) creates one `GenAISequence`, which is one GenAI `Generator` with batch size 1, per request. Concurrent requests share the weights and nothing else.
- On CPU, generators run concurrently (`concurrent_requests.cpu` Supported; tolerance `concurrent_requests: equal_to_sequential`).
- On CUDA, `GenAIModel` runs every native call of every sequence under one lock (`NativeCallLock`, taken at `GenAIModel.cs:199` for generator creation and in `GenAISequence.Invoke` for each step). The reason is a GenAI 0.17.1 defect: concurrent CUDA generators on one model produced wrong tokens (`docs/evidence/p016-qwen35/README.md` section 5). Requests interleave step by step, so CUDA throughput does not grow with concurrency. On a loaded GPU, the exploratory decode rate was 16.9 tokens/s for fp32x and 52.1 for fp16x.
- Features a batched path would have to keep or explicitly drop:
  - per-request `SamplingOptions.Seed`, passed to GenAI's `random_seed` at `GenAIModel.cs:189`;
  - per-request `MaxNewTokens`, stop token ids and stop sequences (`StopSequenceMatcher`, applied to text from the owned tokenizer);
  - cancellation before the prefill, between prefill chunks (`PrefillChunkTokens`) and between decode steps;
  - streaming, usage and timings.

## 2. What ONNX Runtime GenAI 0.17.1 offers

Sources: the managed surface was read by reflection from `microsoft.ml.onnxruntimegenai.managed/0.17.1/lib/netstandard2.0/Microsoft.ML.OnnxRuntimeGenAI.dll`. The native exports come from `libonnxruntime-genai.so` (linux-x64, 0.17.1), and the C API from the packaged `build/native/include/ort_genai_c.h`. That header is identical, apart from line endings, to the one at the upstream tag `v0.17.0` (`1d67be30e5b1ed0c264f6f41c598f85157e8288b`). Implementation details below were read from `v0.17.0` sources and may differ in the 0.17.1 patch. These are upstream code, read as data.

### 2.1 Static batching through the managed `Generator` (available)

| Feature | 0.17.1 behavior |
| --- | --- |
| Batch | `GeneratorParams.SetSearchOption("batch_size", n)`, then one `AppendTokens` with a flat `n x width` id array. The caller pads. Positions equal to `pad_token_id` get mask 0 and position 0, and `rope_deltas` shift later positions (`Qwen2VLPositionInputs`, used for `qwen3_5_text`). `AppendTokenSequences(Sequences)` exists, but `Sequences` has no public constructor (only `Tokenizer.EncodeBatch`). |
| Prefill | `AppendTokens` may be called **once** when `batch_size > 1` ("AppendTokens can only be called once for batch_size > 1"). Chunked prefill, cancellation between prefill chunks and continued decoding are therefore not available. `RewindTo` accepts only 0. |
| Length | One `max_length` for the whole batch: the padded width plus the largest new-token budget. |
| Stopping | GenAI marks a row done only at an EOS id (`GreedySearch_Cpu::SetNextToken`). After that, the row is fed the pad id until every row is done or `max_length` is reached. The row stays in every forward pass ("finished rows burn compute"). Stop sequences and additional stop ids would be applied by Starling per row, but a stopped row cannot leave the batch. |
| Seed | `random_seed` is a `GeneratorParams` option. There is one `std::mt19937` per generator, and rows draw from it in row order. A row's sampled tokens therefore depend on its position and on its neighbours, so per-request seeded reproducibility (`sampling.seeded`, Supported) cannot be kept. |
| Cancellation | Only the whole generator can be cancelled, by stopping the step loop and disposing it. There is no per-row cancel, so a cancelled request either keeps burning its row or tears down its neighbours. |
| Logits | `GetOutput("logits")` returns the logits for all rows. `GetNextTokens()` returns one id per row. |

### 2.2 Continuous batching through the Engine C API (native only in 0.17.1)

Exported by the native library but **not wrapped by the 0.17.1 managed binding**: `OgaCreateEngine`, `OgaEngineRun`, `OgaEngineCreateRequest`, `OgaRequestBeginTurn`, `OgaRequestCancelTurn`, `OgaRequestClose`, `OgaTurnOptionsSet{MaxGeneratedTokens, DoSample, Temperature, TopK, TopP, Seed, StopStrings, Guidance, ...}`, the event-buffer getters (token, finish reason, matched stop-string index, usage) and `OgaEngineGetCapabilities`. Using it from .NET means writing our own P/Invoke layer over a preview C API, kept inside Starlings.Generation.OnnxGenAI.

| Feature | Engine behavior (from the header) |
| --- | --- |
| Scheduling | `engine.dynamic_batching` in `genai_config.json` (`max_batch_size` 16, `max_scheduled_tokens` 2048, `block_size` 256 by default, prefix caching) or `engine.static_batching` (`max_batch_size` 4). One owner thread calls `OgaEngineRun`, and each call executes at most one model transaction. |
| Per request | A Request holds a session token limit. Each Turn has its own sampling settings, max and min generated tokens, stop strings (at most 16 entries, 16 KiB), guidance and seed. |
| Seed | `OgaTurnOptionsSetSeed` gives a per-request stream, but it "requires an Engine configured for dynamic batching". |
| Cancellation | `OgaRequestCancelTurn` is synchronous and owner-thread only. It produces one `Cancelled` event and does not stop the rest of the batch. |
| Hybrid models | Dynamic batching on a model with fixed conv and recurrent state uses `HybridDecoderIO` and a `FixedStatePool`. It requires `decoder.state_groups` with exactly one `paged_kv` group, and the graph must be exported with `use_paged_attention`: PagedAttention, packed `input_ids` without `attention_mask`, `VarlenCausalConvWithState` and varlen Gated DeltaNet. The 0.17 builder rejects that export unless the provider is CUDA ("GatedDeltaNet exports require the CUDA execution provider"). The static Engine path (`StaticBatchDecoderIO`) binds 2-D position ids and no recurrent state, so it does not fit the Qwen3.5 graph. This conclusion comes from reading the code and has not been executed. |

So there is no continuous batching for Qwen3.5 on CPU in 0.17.1. On CUDA it would need a new bundle, a new graph and a new binding.

## 3. Can the qualified graph run batch > 1 with padding?

Graph inspection (`graph-fp32x.txt`, and the same for fp16x): `attention_mask` feeds only a `ReduceSum` and `Shape` subgraph that produces `seqlens_k` and `total_sequence_length` for the 6 GroupQueryAttention layers. The 18 `CausalConvWithState` and 18 `LinearAttention` (gated_delta) nodes take the activations and `past.N.conv` / `past.N.recurrent` (`[batch, 6144, 3]` and `[batch, 16, 128, 128]`), with no mask or per-row length. Any pad token in a row therefore passes through that row's convolution window and delta-rule state:

- **Left padding.** The pads run before the prompt, so the state the real tokens start from is not zero. GroupQueryAttention takes the row's length from the mask sum. That treats the first `length` positions as valid, which is a right-padding convention, so on left-padded rows it attends to pads and drops the end of the prompt. The observed symptom fits this: the shorter row re-emitted the prompt's trailing `<think>\n\n</think>\n\n` block.
- **Right padding.** The pads run after the prompt, so the final recurrent state of the shorter row includes them. During decode, GenAI 0.17.1 sets an all-ones mask over past plus new positions (`Qwen2VLPositionInputs::UpdateAttentionMask`), and new tokens are placed after the pads.
- **Equal lengths.** No padding is needed. Rows are independent and bit-identical to single runs on CPU.

Experiment (CPU, greedy, 32 new tokens; `agreement-cpu.txt`):

| Pair (prompt tokens) | Padding | Shorter row vs single | Longer row |
| --- | --- | --- | --- |
| 22 + 22, same prompt | none | identical, step-0 max abs 0 | identical, 0 |
| 24 + 24, different prompts | none | identical (EOS at 8), 0 | identical, 0 |
| 22 + 40 | left | wrong from step 0 (max abs 17.3) | identical, 0 |
| 22 + 40 | right | 1 token equal, then EOS (max abs 21.4) | identical, 0 |
| 24 + 80 | left | wrong from step 0 (max abs 16.8) | identical, 0 |
| 24 + 80 | right | 1 token equal, then EOS (max abs 15.6) | identical, 0 |

Making padding safe in this graph would need either a mask or length input on the two Gated DeltaNet ops, which is a graph change and a new export, or per-row state resets. GenAI has neither. The paged/varlen export of section 2.2 solves the problem by packing tokens without padding.

## 4. Risks

| Risk | Static `batch_size` (qualified graph) | Engine, dynamic (paged CUDA export) |
| --- | --- | --- |
| Padding through linear attention | Wrong output whenever lengths differ (measured). Only exact-length groups are safe. | No padding (packed varlen). Untested on Qwen3.5. |
| Per-request seed | Lost: one RNG per batch, so samples depend on row order and neighbours. `sampling.seeded` would be Unsupported in batched mode. | Per-turn seed. Reproducibility holds only for the same scheduling path when drafts are used. Needs a test that seeded output is independent of neighbouring traffic. |
| Stop sequences | Applied by Starling per row with the owned detokenizer, but a stopped row keeps running. | Native stop strings are matched on GenAI's own decoding. `stop.sequences` is qualified on the owned detokenizer, so either keep matching in Starling and cancel the turn (one step late), or qualify the native matcher separately. |
| Cancellation | Whole batch only. Cancelling one caller either wastes its row or kills its neighbours. | Per turn, but owner-thread only, so cancellation has to be marshalled to the scheduler thread. It must never end the shared run (the same rule as the phase 2 micro-batcher). |
| Finished rows | Burn compute until the longest row ends. In the spike, one row stopped at 8 tokens and the batch still ran 32 steps. | Finished requests leave the schedule. |
| Chunked prefill and prefill cancellation | Unavailable: one `AppendTokens` per batched generator. | `search.chunk_size` is supported with dynamic batching. |
| Context and length | One `max_length` per batch. Rows longer than 8,192 tokens stay rejected (`long_context.over_8192_tokens` Unsupported). | Per-request session limit, capped by `max_request_length`. |
| CUDA concurrency defect | One generator with batch > 1 is a different code path from concurrent generators. Untested on CUDA. | A single owner thread replaces concurrent generators and may avoid the defect. Untested. |
| API stability | Stable managed API. | Preview C API with no managed binding. We would own the P/Invoke layer and its lifetime rules (event buffers borrowed until the next Run, requests reclaimed on the owner thread). |
| Fidelity bound | There is no protected batch tolerance for Qwen3.5. `concurrent_requests: equal_to_sequential` does not cover batching. | Same, plus a new graph that needs G1 and G2 from scratch. |

## 5. Options

| Option | Effort | Expected gain | Verdict |
| --- | --- | --- | --- |
| A. Keep today's design: CPU concurrent, CUDA serialized | none | n/a | **Default.** |
| B. Static batch of exact-length prompts only (bucket by token count) | small | CPU: about 7% over 2 concurrent requests on a loaded host (8.92 s vs 9.59 s), and it only applies when lengths match exactly, which is rare for chat. CUDA: unmeasured. | Not worth a capability. It loses per-request seed, prefill chunking and per-row cancellation for a marginal gain. |
| C. Static batch with padding on the current graph | small | n/a | **Rejected**: wrong output (section 3). |
| D. Patched export with masked Gated DeltaNet ops | large: graph surgery or upstream ORT op change | unknown | Out of scope. It also needs new ORT kernels. |
| E. GenAI Engine with dynamic batching, on a paged CUDA export of Qwen3.5 | large: P/Invoke binding, scheduler thread, new bundle and profile, full qualification | Potentially large on CUDA, where serialized concurrency gives no parallelism today. | **Spike on RunPod**, behind a new Experimental profile. |

## 6. Recommended path

1. **Now:** no product change. Keep `batch` Unsupported, CPU concurrency, and CUDA serialization. Record this note.
2. **CUDA baseline on RunPod (RTX 3090), cheap, with the existing harness:** measure serialized concurrency at 1, 4 and 16 for fp32x and fp16x (tokens/s, time to first token, p50 and p95 latency). Run the same spike program with the CUDA natives against the equal-length and padded pairs, to confirm the padding finding on CUDA and to see whether one batched generator escapes the concurrent-generator defect. The output of this step is the bar the Engine must clear.
3. **Engine spike (time-boxed, outside `src/`):** build a paged export with the 0.17.1 builder (`use_paged_attention=true`, CUDA, fp16x first), with `engine.dynamic_batching` in its `genai_config.json`. Drive it from a minimal P/Invoke prototype over `OgaCreateEngine`, `OgaEngineRun`, `OgaRequestBeginTurn` and `OgaRequestCancelTurn` on one owner thread. Measure throughput and latency against step 2 at 4 and 16 concurrent requests, with mixed prompt lengths and outputs from 8 to 512 tokens.
4. **Only if step 3 wins clearly (target: at least 1.5x serialized throughput at concurrency 4, with p95 latency no worse):** propose a new profile, for example `qwen3.5-0.8b/ortgenai-fp16x-paged-cuda`, at maturity Experimental. That means a new bundle, manifest and P-030 qualification. Add an opt-in Engine-backed generator in Starlings.Generation.OnnxGenAI, with:
   - a scheduler-owned run that a caller's cancellation never ends;
   - stop sequences still matched with the owned detokenizer (cancel the turn on a match);
   - per-turn seeds;
   - explicit statuses: `batch.dynamic.cuda` Experimental, `batch.static` Unsupported, `batch.cpu` Unsupported.

   The Stable profiles and their defaults stay as they are.

### Evidence and tolerances a batched profile would need

The maintainer sets these. Agents do not write `model-pipeline/protected/`. Suggested keys:

- `discrete.batched_vs_single: greedy_ids_exact_except_listed_margin_below_1e-3`. Each request's greedy output under mixed concurrent traffic (4 and 16, mixed lengths, random arrival) is equal to its single-request output, with the same near-tie rule as `greedy_ids`.
- `float.batched_vs_single_logprob_max_abs`: the step-0 and per-step chosen-token log-prob difference between batched and single runs. The current cached-vs-uncached bound is 2e-4. Packed varlen kernels are expected to differ from the dense graph, so this needs a measured value and a maintainer decision, not a copy.
- `discrete.batched_seeded_sampling: equal_to_single`, or an explicit Unsupported if the Engine's seeded streams turn out to depend on neighbours.
- `discrete.batched_cancellation`: a cancelled request ends with `OperationCanceledException`, and its neighbours' outputs are unchanged (equal to single).
- The usual gates on the new graph: G1 and G2 against the transformers reference (prompts-v1, long-context set), G3 GSM8K budget, G4 soak with cancellation churn and bounded VRAM, and G5 performance against serialized concurrency.

## 7. Open questions

- Does GenAI 0.17.1 batch > 1 on CUDA show the same row independence for equal lengths, and does it avoid the concurrent-generator defect? This is unmeasured and is step 2 above.
- Is the 0.17.1 builder's paged Qwen3.5 export numerically close enough to the dense graph to reuse the dense fixtures' near-tie rules?
- Will a later GenAI release add a managed Engine binding? If so, wait for it rather than own a P/Invoke layer. Any version change also needs the pinned builder, the bundle and every profile requalified together (`Directory.Packages.props` note).
