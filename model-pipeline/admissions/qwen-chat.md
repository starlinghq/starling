# Admission: Qwen text chat (Qwen3-0.6B, Qwen2.5-0.5B-Instruct)

Pipeline stage 1 for `Starlings.Models.QwenChat` (plan section 7; five-model onboarding, `docs/plan/five-model-onboarding.md`). Requested in #124: Roost (starlinghq/roost) item R-009 needs a published Qwen3 chat package for its first model. Evidence was gathered on 2026-10-18 from snapshots downloaded at the pinned commits and checked against the candidate contracts (`model-pipeline/candidates/qwen3/profile.json`, `model-pipeline/candidates/qwen25/profile.json`). Demand figures come from the Hugging Face Hub API on the same day.

| Field | Value |
| --- | --- |
| Candidate | `Qwen/Qwen3-0.6B` (profile `qwen3-0.6b/ortgenai-fp32-cpu`, template `qwen3-text-v1`) and `Qwen/Qwen2.5-0.5B-Instruct` (profile `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu`) |
| Code revision | No upstream code is run or adopted. The C# chat templates adapt the `chat_template` strings at the weight revisions below (`ai-dotnet/src/Starlings.Models.QwenChat/THIRD-PARTY-NOTICES.md`). The bundle is built with the ONNX Runtime GenAI 0.17.1 builder at source commit `83de55a1f3886cddbb32a5517e26adb3f9d65159`. |
| Weight revision | Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`; Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`. Every file's size and SHA-256 is in the candidate contracts and was verified on download. |
| Requested by / demand evidence | #124 (Roost R-009: a named consumer). Hub downloads on 2026-10-18: Qwen3-0.6B 247.0M (1,828 likes), Qwen2.5-0.5B-Instruct 74.8M (697 likes). |
| Proposed owner | Mark |
| Proposed maturity at first release | Experimental |
| Decision | Proposed: admitted at maturity Experimental, both profiles, CPU fp32 on Linux x64. Bundles built locally from the pinned originals, no Starling hosting (see "Open risks"). |
| Decided by and date | Pending: merging the PR that adds this record is the decision. |

## Rights (hard blockers)

| Item | License or terms | Redistribution |
| --- | --- | --- |
| Code | No upstream code adopted. The template strings are Apache-2.0 (Copyright 2024 Alibaba Cloud), recorded in the package's `THIRD-PARTY-NOTICES.md` with the full license in `LICENSE.Qwen.txt`. | Permitted with the license and attribution, which the package carries. |
| Weights | Apache-2.0: the `LICENSE` file at each pinned revision (SHA-256 `832dd9e0...e92e`, 11,343 bytes, identical in both) and the model card front matter (`license: apache-2.0`). | Apache-2.0 permits redistribution. Starling does not redistribute: the package ships no weights, and users build the bundle from the original Hub files. |
| Tokenizer assets | Apache-2.0, same repositories and revisions. | Same as the weights. |
| Evaluation data | The Starling-written corpus `model-pipeline/corpora/five-models/chat.jsonl` (14 conversations). | Repository license. |

No hard blocker.

## Score

| Criterion | Weight | Score (0 to 5) | Notes |
| --- | --- | --- | --- |
| User demand | 25% | 4 | Very high Hub usage and a named consumer (Roost R-009). |
| Backend feasibility | 20% | 4 | The same GenAI 0.17.1 backend and decode loop as the Stable Qwen3.5 profiles, with standard `qwen3` and `qwen2` GenAI model types. Development parity: see "Evidence". |
| Reusable architecture coverage | 20% | 3 | Adds two dense Qwen generations and their chat templates on the existing generation backend. No new runtime. |
| Access and licensing clarity | 15% | 5 | Apache-2.0, public, files pinned by commit and SHA-256. |
| Testability | 10% | 3 | Full reference route (transformers, Python GenAI, .NET) on a small corpus of 14 conversations with 16 new tokens each. No upstream evaluation suite is used. |
| Maintenance burden (5 = low) | 10% | 4 | Upstream revisions are old and stable (2024 and 2025). Two template contracts to maintain. |
| **Weighted total** | | **3.85** | |

## Execution route

ONNX Runtime GenAI 0.17.1, CPU fp32, through `Starlings.Generation.OnnxGenAI` (the Qwen3.5 backend). The bundle comes from the GenAI builder run on the original safetensors (`model-pipeline/reference/qwen-chat/build_bundle.py`). The decode loop is Starling's: greedy decoding, stop ids and strings, streaming, cancellation between steps, and usage. The route gives step-0 logits for fidelity, but not hidden states.

GenAI masks token id 151643 (`<|endoftext|>`) as padding. Both profiles therefore overlay `model.pad_token_id=151935`, the unused last embedding row, so a literal `<|endoftext|>` in a prompt stays attended. The original files are never modified.

## Tokenizer profile

Byte-level BPE from each repository's `tokenizer.json` through `Starlings.Tokenizers` (`Tokenizer.FromFile`, pinned package version). The digests are pinned in `QwenChatProfile`: Qwen3 `aeb13307...dae4`, Qwen2.5 `c0382117...27f`. The loader also checks the identities of `<|im_start|>` 151644, `<|im_end|>` 151645 and `<|endoftext|>` 151643, and that id 151935 is unused. No GenAI or upstream tokenizer is used at runtime.

## Feature contract

| Upstream feature | In scope | Evidence source |
| --- | --- | --- |
| Text chat with system, user and assistant roles | Yes, greedy | Development parity (prompt text and ids, step-0 logits, greedy ids, finish reason) |
| Qwen2.5 default system instruction | Yes (inserted when the first message is not a system message) | Template tests; parity corpus |
| Qwen3 thinking mode | No: `enable_thinking=false` is fixed (differs from the upstream default) | Template tests |
| Earlier assistant reasoning (Qwen3) | Dropped before the last user turn, as upstream | Template tests |
| Streaming, usage, stop ids and strings | Yes | Development parity (stream, usage) |
| Cancellation | Before prefill and between decode steps | Development parity (cancellation) |
| Fresh cache per request; request isolation | Yes | Development parity (cache, isolation) |
| Sampling | No | n/a |
| Tools, tool messages, multimodal input | No (rejected) | Template tests |
| CUDA, fp16, quantized graphs | No | n/a |

## Hardware and compute estimate

CPU only, one or two native threads (the package bounds them). Qwen3-0.6B bundle: about 3 GB of fp32 weights; Qwen2.5-0.5B: about 2.5 GB. The onboarding route runs in under an hour per profile on a desktop CPU.

## Evidence

EVIDENCE_PLACEHOLDER

## Open risks

- **Asset lock.** ASSET_LOCK_PLACEHOLDER
- **Hosting.** Starling hosts no QwenChat bundle. Users install `onnxruntime-genai` 0.17.1 and run `build_bundle.py`, so this package is harder to adopt than Kev or Qwen3.5. Public hosting would need its own decision, as P-009 was for Kev.
- **References are proposed.** The transformers references and the comparison policy (exact prompt and greedy ids, step-0 logits within 0.005 max and 0.0005 mean) are not adopted into `protected/`, and no gate (G0 to G7) has run.
- **`MaxNewTokens` defaults to 16**, the corpus setting. Applications must set it.
- **Naming.** `ai-dotnet/samples/QwenChat` is the Qwen3.5 sample, not this package's.

## What Preview would additionally need

Plan section 14: "declared scope passes meaningful tests".

1. Adopt the references and tolerances into `protected/` in a separate protected-change PR, then run G1 and G2 on them.
2. Add model-level C# tests that load a bundle: generation, streaming equal to non-streaming, cancellation, concurrent requests, and disposal. Today the package has only seven template tests.
3. G4: a soak with bounded memory, and repeated load and unload.
4. G6: a clean-consumer drill from the packages (`model-pipeline run --distribution`).
5. A larger corpus: longer completions than 16 tokens, multi-turn conversations, long prompts near the stated context, and non-English text.
6. Decide whether to keep a local-build-only route or host bundles.
