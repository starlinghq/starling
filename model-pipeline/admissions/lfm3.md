# Admission: LFM3 family (Liquid AI)

Model request #217. Plan section 7, pipeline stage 1. Facts were read from the Hub on 2026-10-28 (API and files at the pinned revisions); every pinned file was downloaded and hashed.

| Field | Value |
| --- | --- |
| Candidate | `LiquidAI/LFM3-1.7B` (text) through its ONNX export `LiquidAI/LFM3-1.7B-ONNX`. Later candidates of the same family: `LiquidAI/LFM3-VL-2B`, `LiquidAI/LFM3-Audio-2B`, `LiquidAI/LFM3-Omni-2B` (not in this admission). |
| Code revision | `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd` (`modeling_lfm3.py`, `configuration_lfm3.py`, `lfm3_conv.py` in the checkpoint repository; the reference runs them with `trust_remote_code`) |
| Weight revision | Checkpoint `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`; ONNX export `c680759afee1ce8310c90d6c621f4558200dee77` |
| Requested by / demand evidence | #217. Released 2026-10-27; 6,311 downloads and 155 likes on the first day. The card reports LiveClaw 45.6 (Qwen3.5-4B 19.9, Qwen3.5-9B 35.8), relevant to the tool-calling gap in #181. |
| Proposed owner | Unassigned |
| Proposed maturity at first release | Experimental |
| Decision | Admitted at maturity Experimental, to be published on nuget.org with the LFM Open License v1.0 as the package's license file and license acceptance required on install (`Starlings.Models.Lfm3`, `LICENSE.txt`); the revenue condition opens the package description and README. Registry profile `lfm3-1.7b/onnx-fp32-cpu`. |
| Decided by and date | The maintainer, 2026-10-30: "make sure that license is attached and prominent in the NuGet package for that model" |

## Rights (hard blockers)

| Item | License or terms | Redistribution |
| --- | --- | --- |
| Code (`modeling_lfm3.py` and companions) | LFM Open License v1.0 (repository license) | Not redistributed. The reference runs it; no file is copied into this repository. |
| Weights (checkpoint and ONNX export) | LFM Open License v1.0: an Apache-2.0-style grant (sections 2 and 3), except that commercial use is licensed only to a legal entity below 10 million USD annual revenue (section 5(a), 5(b)); qualified non-profits are exempt for non-commercial or research use (5(c)) | Not redistributed: the package ships an asset lock and downloads from the Hub on request (`Lfm3ModelAssets.DownloadAsync`). |
| Tokenizer assets | Same repository, same license | Not redistributed (fetched by `tools/fetch-assets.sh` in starlinghq/tokenizers and by the asset lock). |
| Chat template | Same repository, same license | Ported to C# (`Lfm3ChatTemplate.cs`), recorded in `ai-dotnet/THIRD-PARTY-NOTICES.md` and `ai-dotnet/PROVENANCE.md`. |
| Evaluation data | `corpora/five-models/chat.jsonl` (existing) and `corpora/lfm3/chat.jsonl` (written for this admission) | Repository content. |

Decided 2026-10-30: Starling offers the model, and the package states the condition where a user meets it first. The license text is the package's license file (NuGet shows it and asks for acceptance on install). The revenue condition opens the package description and the README. The ported template is named as a derivative work under that license. This goes further than the EmbeddingGemma 2 precedent, which repeats the upstream policy sentence in the README and the asset notice.

## Score

| Criterion | Weight | Score (0 to 5) | Notes |
| --- | --- | --- | --- |
| User demand | 25% | 3 | One day old; strong first-day downloads; addresses the tool-calling gap of #181 if the card's agent scores hold. |
| Backend feasibility | 20% | 4 | Liquid ships an ONNX export that runs on ONNX Runtime 1.30.0 (standard `com.microsoft` GroupQueryAttention and RotaryEmbedding). ONNX Runtime GenAI has no LFM3 model type, so a managed loop is needed (written). |
| Reusable architecture coverage | 20% | 3 | First convolution-hybrid family; the managed step loop also fits other exports with explicit caches. |
| Access and licensing clarity | 15% | 2 | Ungated, but the revenue threshold is a non-standard restriction. |
| Testability | 10% | 4 | Same-graph parity is bit-exact; a PyTorch reference exists but runs remote code. |
| Maintenance burden (5 = low) | 10% | 3 | Owned loop and template port; a new tokenizer regex in starlinghq/tokenizers. |
| **Weighted total** | | **3.15** | |

## Execution route

ONNX Runtime 1.30.0 (`Starlings.Runtime.Onnx`) over the export's single step graph: `input_ids`, `attention_mask`, `position_ids`, `logits_to_keep`, 13 convolution caches `past_conv.N` (`[1, 2048, 7]`) and 11 attention caches `past_key_values.N.key/value` (`[1, 4, past, 128]`), returning `logits` and the `present*` caches. The decode loop is owned (`Lfm3Sequence`, `Lfm3ChatModel`): greedy, stop ids, stop sequences, incremental detokenization, prefill chunks. It gives the full logits, token ids in and out, and template control. It does not give sampling, logprobs or guided decoding (Unsupported in this package). CUDA: the export's card says the CUDA provider has no fp32 GroupQueryAttention kernel, so the RTX 3090 profile is the fp16 graph (`model_fp16.onnx`, 3.6 GB; locked, untested).

## Tokenizer profile

Byte-level BPE (`ignore_merges` true, vocabulary 124,893, 293,320 merges, 140 added tokens) with the Llama 3 split regex (`\p{N}{1,3}`). `Starlings.Tokenizers` 0.3.x rejected that regex; starlinghq/tokenizers#18 adds it and the profile `lfm3-bpe-chat` (137/137 cases exact against huggingface/tokenizers 0.23.3). transformers 5.21.0 loads the Hub file as is (`TokenizersBackend`). The ONNX export's `tokenizer.json` is the checkpoint's without its final newline.

## Feature contract

| Upstream feature | In scope | Evidence source |
| --- | --- | --- |
| Text chat, system/user/assistant | Yes | Template renderings and prompt ids against transformers 5.21.0 (19 cases) |
| Thinking switch (`enable_thinking`, default true), `preserve_thinking` | Yes | Same corpus |
| Greedy generation, fp32 CPU | Yes | Same-graph parity with ONNX Runtime in Python (bit-exact); PyTorch reference pending |
| fp16 on CUDA (RTX 3090) | Planned | Needs a GPU run |
| Card sampling (`temperature 0.1, top_k 50, repetition_penalty 1.1`) | No | Unsupported in the first cut |
| Tool calling (XML `<tool_call>`), deferred tools | No | Later; the template's tool sections are rejected |
| Vision, audio (VL-2B, Audio-2B, Omni-2B) | No | Separate candidates |
| 256K context | No | `Lfm3Options.ContextLength` defaults to 32,768; longer is untested |

## Hardware and compute estimate

fp32: 7.2 GB of weights; about 7.5 GB resident on CPU; attention caches 44 KB per token. fp16: 3.6 GB, fits an RTX 3090 (24 GB) with room for long contexts. CPU throughput on a 4-vCPU container: 19 cases (313 tokens, prompts up to 368 tokens) in 95 s including model load.

## Open risks

- The export's fidelity to the checkpoint is unmeasured until `run_lfm3.py reference` runs.
- The convolution caches make prefix reuse (#161) a hybrid-family problem, as for Qwen3.5 (`docs/plan/prefix-state-reuse.md`).
- One-day-old upstream: revisions may move quickly; everything is pinned by full SHA.
