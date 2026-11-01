# Prefix state reuse for chat: design note

Status: design note, 2026-10-26 (#161; 2026-10-25 review, item B4 track b); its open questions were decided on 2026-11-01 (section 6). Nothing in this note changes product code, a capability status or a tolerance. `conversation_state_reuse` stays `unsupported` on every Qwen3.5 and QwenChat profile (`Qwen35Capabilities`, the five manifests under `model-pipeline/registry/`). Kev's state reuse is a separate track (#140, #166).

## Summary

- Every chat request renders and prefills the whole conversation in a fresh generator (`Qwen35ChatModel` remarks; `QwenChatTypes.cs`, "conversation cache state is never reused"). A consumer reports 8.1 s at p50 for a turn with 28 tools in a 3,334-token prompt on the RTX 3090 (corvid S2, quoted in #161). Most of such a prompt (the tools block and the earlier turns) was already processed by the previous turn's generator.
- Reusing that work needs three things: a rendering whose token ids share a long prefix with the previous request's, a way to return the model state to the end of that shared prefix, and fidelity evidence that the reused state gives the same outputs as a fresh prefill.
- The Qwen3.5 profiles fail the second condition on the managed ONNX Runtime GenAI 0.17.1 binding. Their Gated DeltaNet layers keep a recurrent state that cannot be rewound to an earlier position (`RewindTo(n)` with `0 < n` fails), and the native snapshot export that would allow it is not wrapped. The template makes the shared prefix end at the previous assistant turn's header, so plain appending does not help either.
- Prefix caching therefore needs **either a non-recurrent model** (standard attention only, where the key/value cache can be truncated to the shared prefix) **or GenAI support beyond the managed 0.17.1 binding** (the continuous-batching Engine API with prefix caching, or the native state snapshot). Both GenAI routes mean a P/Invoke layer over a preview C API that Starling would maintain.
- For the next generator family (the model request in #181, postponed item F3), prefix reuse is straightforward only if the family has no recurrent layers. A candidate from the same hybrid family as Qwen3.5 inherits every constraint in section 3.

## 1. Where the time goes

`TextGenerator` (`Starlings.Generation.OnnxGenAI`) creates one `GenAISequence`, one GenAI `Generator` with batch size 1, per request (`docs/plan/qwen35-batched-generation.md`, section 1). The chat models render the conversation with the owned template, tokenize it with the owned tokenizer and append the ids. Continuing a conversation means sending the whole conversation again (`Qwen35ChatModel` remarks).

For a turn `k`, the prompt holds the system block (with the tools block when tools are passed), every earlier turn, and the new user message. Turn `k` repeats all of turn `k-1`'s prompt except its last few lines. The share of repeated tokens grows with the conversation and with the size of the tools block, which is the same on every turn.

## 2. What reuse needs

| # | Condition | Why |
| --- | --- | --- |
| 1 | **A shared token prefix.** The ids of turn `k`'s rendering start with a long run equal to the ids the previous generator processed. | The cache holds state for ids, not text. The shared prefix is the longest common prefix of the two id sequences, computed after tokenization; an assistant reply that re-tokenizes to other ids than the ones generated ends the shared prefix at that point. |
| 2 | **A way back to the end of the shared prefix.** The generator's state must be returned to position `p`, the shared-prefix length, before the rest of turn `k` is appended. | Turn `k-1`'s state also contains tokens after `p` (the generation prompt and the reply as generated), which turn `k` renders differently. |
| 3 | **Equal outputs.** Reused state followed by the remaining ids must give the same greedy ids and step-0 log-probabilities as a fresh prefill of the whole prompt, within maintainer-set bounds. | Plan gates G1 and G2. The profiles already check cached against uncached decoding (`cached_vs_uncached_logprob_max_abs` in the protected tolerances of the Qwen3.5 profiles); a reuse check would need its own protected bound, set by the maintainer. |
| 4 | **Lifetime and memory.** Kept state lives beyond one request: per conversation, with an owner, a size bound and eviction. | Today a generator is disposed with its request. A kept generator pins native memory and, on CUDA, takes its turn under `GenAIModel`'s `NativeCallLock` like any other. |

## 3. Qwen3.5: why the current profiles cannot reuse state

### 3.1 The template moves the shared prefix to the previous assistant header

The owned renderer follows the pinned template (`Qwen35ChatTemplate.Render`). An assistant message after the last user query is rendered with its think block; an assistant message before it is rendered without one. The generation prompt ends with `<|im_start|>assistant\n` followed by `<think>\n` (thinking on) or an empty `<think>\n\n</think>\n\n` block (thinking off, the template default).

So for turn `k-1` the generator processed `... user k-1 <|im_start|>assistant\n<think>...</think>\n\n reply k-1`, while turn `k` renders `... user k-1 <|im_start|>assistant\n reply k-1 <|im_end|>\n user k ...`. With thinking on or off, the two diverge right after the previous `<|im_start|>assistant\n`. The shared prefix covers the system and tools block and every earlier turn, which is most of a long conversation, but it never covers everything the previous generator processed. Appending turn `k`'s new ids to the old generator is therefore not an option; the state must go back to `p` first.

Rendering earlier think blocks anyway (or omitting the empty block from the generation prompt) would make the rendering append-only, but it is not the pinned template. Outputs would change, so it would be a new profile with its own qualification, not an option on the Stable ones (plan section 14).

### 3.2 The recurrent layers cannot go back to `p`

The qualified graphs have 6 GroupQueryAttention layers and 18 Gated DeltaNet layers (`CausalConvWithState` and `LinearAttention`), whose state is `past.N.conv` (`[batch, 6144, 3]`) and `past.N.recurrent` (`[batch, 16, 128, 128]`) (`docs/plan/qwen35-batched-generation.md`, section 3). Each new token updates that state in place, so no earlier position can be recovered from it.

Measured on GenAI 0.17.1 (`docs/m0/generation-backend-check.md`, section 5, "KV/state reuse for a conversation"):

- appending the next turn's ids to the same generator equals a fresh generator on the full prompt (16 tokens);
- `RewindTo(n)` with `0 < n` fails with "RecurrentState::RewindTo(26) is not supported without a snapshot"; `RewindTo(0)` works;
- the native library exports `OgaGenerator_SnapshotState`, but the managed binding does not expose it (untested).

LLamaSharp 0.27.0, the recorded fallback backend, shows the same limit on the same model: partial `llama_memory_seq_rm` returns false on the hybrid memory, while saving and loading the whole sequence state (20.9 MB at 40 tokens) and continuing from it equals a fresh run (same section).

### 3.3 What a snapshot would cost

A snapshot at `p` is the 18 recurrent and convolution states plus the attention layers' key/value cache up to `p`. From the shapes above at fp32: 18 x 16 x 128 x 128 x 4 bytes is 18 MiB of recurrent state and 18 x 6,144 x 3 x 4 bytes is about 1.3 MB of convolution state, about 20 MB per snapshot before the key/value cache, which grows with `p`. That agrees with the 20.9 MB LLamaSharp measured at 40 tokens. The snapshot must be taken during the previous request, at the moment the prefill reaches `p`, which is before the reply exists. `PrefillChunkTokens` already splits the prefill; a reuse path would end a chunk exactly at the previous user turn's `<|im_start|>assistant\n` and snapshot there.

## 4. Routes

| Route | What it needs | Fits the current Qwen3.5 profiles | Cost and risk |
| --- | --- | --- | --- |
| A. A non-recurrent model on the managed binding | A model with standard attention only. Keep the generator per conversation, `RewindTo(p)` to the shared prefix, append the rest. | No. | No new binding. `RewindTo(n)` on an attention-only model is not yet measured in this repository; the QwenChat profiles (attention only, Experimental, CPU) are the place to measure it. The rewind point is the same as in section 3.1 for a template that drops earlier think blocks; a template without think blocks renders append-only. |
| B. The native state snapshot | P/Invoke over `OgaGenerator_SnapshotState` (and whatever restores a snapshot), kept inside `Starlings.Generation.OnnxGenAI`. Snapshot at the assistant header of each turn, as in section 3.3. | In principle, on the existing bundles. | A preview native API the managed binding does not wrap; untested. About 20 MB plus key/value cache per kept conversation. |
| C. The GenAI Engine API | P/Invoke over the continuous-batching Engine C API (`OgaCreateEngine`, `OgaEngineRun`, requests and turns), with `engine.dynamic_batching`, which lists prefix caching in its configuration (`docs/plan/qwen35-batched-generation.md`, section 2.2). | No. For a hybrid model, dynamic batching needs a paged export (PagedAttention plus varlen Gated DeltaNet) that the 0.17 builder produces only for the CUDA provider, so it is a new bundle and a new profile, and CPU is out. | Brings continuous batching as well. Whether its prefix cache covers the recurrent state of a hybrid model has not been read or run; this note makes no claim about it. Shares the time-boxed CUDA spike that section 6 of the batching note recommends. |
| D. Wait for the managed binding | Nothing now. Re-read each GenAI release for a managed Engine or snapshot wrapper. | Not until it lands. | No cost now; no reuse until then. |

Every route keeps the fidelity condition of section 2: a reuse path ships as a new Experimental capability (or a new profile for route C) with a maintainer-set bound comparing reused and fresh prefill. Stable profiles do not change.

## 5. Consequence for the next generator family

A family with standard attention only makes route A available on the existing managed binding: rewind the key/value cache to the shared prefix and append. A hybrid family with recurrent layers, such as the one Qwen3.5 belongs to, needs route B or C whatever its size. When the next generator is chosen (#181), the admission should record:

1. whether the architecture has recurrent or other state that cannot be truncated;
2. what its chat template does to earlier turns (think blocks dropped or kept, generation prompt suffix), which sets where the shared prefix ends;
3. a measured `RewindTo(p)` plus append against a fresh prefill on its bundle, before reuse is offered.

## 6. Decisions

The former open questions, decided 2026-11-01 by the maintainer's delegate (agent), revisitable (#161).

1. **Engine or wait:** wait. Do not wrap the GenAI Engine API now (route C); wait for the managed binding (route D), and revisit when the batching note's CUDA Engine spike or an upstream managed wrapper lands. Rationale: no upstream maintenance burden and no new native surface on a pre-1.0 binding.
2. **Requirement for the next generator:** prefix reuse is not a hard requirement for the family chosen in #181. The family's admission records whether it has recurrent layers and whether its chat template keeps a stable rendered prefix (section 5, items 1 and 2), so that reuse stays possible. Rationale: it keeps the choice of family open while preserving the information reuse would need.
3. **Snapshot spike:** deferred. The CPU spike of route B runs when a consumer needs latency below the current full prefill. `conversation_state_reuse` stays unsupported. Rationale: no consumer has asked for it, and route B is a preview native API.
