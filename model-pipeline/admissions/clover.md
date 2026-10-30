# Admission: Clover (DiffuSpace)

Model request: Clover-35B-A3B-Preview. Plan section 7, pipeline stage 1. Facts were read from the Hub on 2026-10-28 (API and files at the pinned revision); every small file was downloaded and hashed, and the safetensors digests are the Hub's LFS SHA-256.

| Field | Value |
| --- | --- |
| Candidate | `Diffuspace/Clover-35B-A3B-Preview`: a block-diffusion fine-tune of `Qwen/Qwen3.6-35B-A3B` (text only) |
| Code revision | `4b0402b65e523d0183a14c431955438714cb16ff` (`modeling_clover_moe.py`, `configuration_clover_moe.py`, `sampling_clover_moe.py` in the checkpoint repository, run by transformers 5.19.0 with `trust_remote_code`). The card names SGLang and llama.cpp implementations in github.com/DiffuSpace/Clover-35B-A3B-Preview; no commit of that repository is pinned here. |
| Weight revision | `4b0402b65e523d0183a14c431955438714cb16ff`: 26 bf16 safetensors shards and two bidirectional GDN parameter files, 69.3 GB (`CloverCheckpoint.Lock`) |
| Requested by / demand evidence | Model request in this repository. Released 2026-10-24; 105 downloads and 68 likes on 2026-10-28. The card reports agentic benchmark results on the DiffuSpace blog; none were reproduced here. |
| Proposed owner | Unassigned |
| Proposed maturity at first release | Experimental (model definition only) |
| Decision | **Pending maintainer decision** (execution route, below). Development work in the same pull request is the model definition: Experimental and source only. Nothing is registered or published. |
| Decided by and date | |

## Rights (hard blockers)

| Item | License or terms | Redistribution |
| --- | --- | --- |
| Code (`modeling_clover_moe.py` and companions) | Apache-2.0 (repository license; derived from the transformers Qwen3.5-MoE implementation, `ATTRIBUTIONS.md`) | Not redistributed and not run here. |
| Weights | Apache-2.0; upstream Qwen3.6-35B-A3B assets Copyright 2026 Alibaba Cloud (`LICENSE`, `ATTRIBUTIONS.md`) | Not redistributed: `CloverCheckpoint.Lock` pins the files, nothing downloads them. |
| Tokenizer assets | Same repository, same license | Not redistributed (fetched by `tools/fetch-tokenizer-assets.sh` and by `tools/fetch-assets.sh` in starlinghq/tokenizers). |
| Chat template | Same repository, Apache-2.0 (the file carries its own notice) | Ported to C# (`CloverChatTemplate.cs`), recorded in `ai-dotnet/THIRD-PARTY-NOTICES.md` and `ai-dotnet/PROVENANCE.md`. |
| Evaluation data | `corpora/five-models/chat.jsonl` (existing) and `corpora/clover/chat.jsonl` (written for this admission) | Repository content. |

No rights blocker.

## Score

| Criterion | Weight | Score (0 to 5) | Notes |
| --- | --- | --- | --- |
| User demand | 25% | 2 | Four days old; one request. |
| Backend feasibility | 20% | 1 | No backend Starling has runs block diffusion (below). |
| Reusable architecture coverage | 20% | 2 | The Qwen3.6 hybrid GDN MoE backbone is new to Starling (Qwen3.5-0.8B is dense); the decoder is the first diffusion one. |
| Access and licensing clarity | 15% | 5 | Ungated, Apache-2.0. |
| Testability | 10% | 2 | The reference runs remote code on a GPU with at least about 69 GB of memory for the bf16 weights; the card says its PyTorch and SGLang paths can differ in generated text. |
| Maintenance burden (5 = low) | 10% | 1 | An owned runtime for a 35B MoE diffusion decoder, against a preview checkpoint. |
| **Weighted total** | | **2.15** | |

## Execution route

None in this repository. Clover does not generate one token per step: it appends a block of 32 `<|mask|>` positions (id 151669), runs the block with bidirectional attention inside it over a causal prefix, fills every position whose confidence exceeds 0.95 (at least one per pass), and repeats until the block is resolved. The model also adds learned GDN parameters for that bidirectional pass (right convolution weights, block-size softmax scalars) and restricts expert routing per block (`moe_block_coreset_*`). What the existing routes offer:

- **ONNX Runtime GenAI** (Qwen routes): autoregressive only; no Clover model type, no block denoising, no Qwen3.6 MoE.
- **ONNX Runtime with an owned loop** (the LFM3 pattern): needs an ONNX export of the denoising forward with its GDN state. None is published. A Starling export would mean exporting a 35B MoE with custom GDN operators, and the bf16 weights (69.3 GB) exceed the RTX 3090 (24 GB) and, in fp32 (about 139 GB), the workstation's 94 GB of RAM. It would need quantization, which is a separate fidelity question.
- **llama.cpp** (DiffuSpace's own implementation, upstream of the card): a fork, not a release Starling pins. Its self-speculative mode is a different decoding algorithm from the transformers reference.

What this pull request delivers instead is the runtime-independent model definition (`ai-dotnet/src/Starlings.Models.Clover`): the chat template, the checkpoint's ids and shape as a `ModelProfile`, and the asset lock. A backend could build on that once one is chosen. Generation is listed as Unsupported in `CloverCapabilities`.

## Tokenizer profile

Byte-level BPE, the Qwen2/Qwen3 pipeline (NFC, the Qwen2 split regex without `\p{M}`, ByteLevel post-processor that adds nothing): vocabulary 248,044, 247,581 merges, 34 added tokens. Against the Qwen3.5 vocabulary, id 151669 holds the special added token `<|mask|>` instead of `ĠãģĿãĤĮ` (" それ"), and the 6 merges that built that token are removed. transformers 5.19.0 loads the Hub file as is (`Qwen2Tokenizer`), so the asset is the Hub file. It loads with the `Starlings.Tokenizers` version this repository pins (0.3.2; no new component). starlinghq/tokenizers#20 adds the profile `clover-bpe-chat` with fixtures from huggingface/tokenizers 0.23.3 (137/137 cases exact, plus a 13-case `<|mask|>` check).

## Feature contract

| Upstream feature | In scope | Evidence source |
| --- | --- | --- |
| Text chat, system/user/assistant | Yes (definition) | Template renderings and prompt ids against transformers 5.19.0 (30 cases) |
| `enable_thinking` (default true), `preserve_thinking` (default true), `add_non_thinking_prefix` (default true) | Yes (definition) | Same corpus; every switch off at least once |
| Block-diffusion generation (LowConfidence, block 32, threshold 0.95) | No | No backend (above) |
| Self-speculative decoding (SGLang, llama.cpp) | No | Same |
| Presence, frequency and repetition penalties; length and think-end biases | No | Generation-only controls |
| Tool calling (XML `<tool_call>`) | No | Later; the template's tool sections are rejected by the port |
| 128K context | Recorded | `CloverCheckpoint.ContextLength`; nothing runs it |

## Hardware and compute estimate

bf16 weights: 69.3 GB (the card asks for a GPU with that much memory plus inference state). About 3B parameters are active per token, but all experts must be resident or paged. Neither the RTX 3090 (24 GB) nor the RTX 4060 Ti fits the bf16 checkpoint. The workstation's 94 GB of RAM would hold it for a CPU run of the PyTorch reference, which was not attempted. The template and tokenizer evidence needed no model weights and ran on a 4-vCPU container in seconds.

## Open risks

- The execution route above: admitting Clover means choosing and funding a backend, or keeping it a definition-only package.
- `tokenizer_config.json` carries a `pretokenize_regex` with `\p{M}` (the Qwen3.5/3.6 regex), but the `tokenizer.json` that transformers 5.19.0 runs uses the Qwen2 regex. Text with combining marks (Devanagari, Thai, Arabic harakat, Tamil) therefore tokenizes differently from the Qwen3.6 base. Starling reproduces what transformers runs. Whether Clover was trained on that tokenization, and whether DiffuSpace's SGLang path does the same, is unverified.
- Preview checkpoint, four days old: revisions may move quickly; everything is pinned by full SHA.
- The card states that the PyTorch and SGLang implementations can produce different text with the same settings, so there is no single reference output to qualify a runtime against.
