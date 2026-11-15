# Admission: Phi-4-mini-instruct (Microsoft)

Model request #109. Plan section 7, pipeline stage 1. Facts were read on 2026-11-04 from the files at the pinned revision, which were downloaded and hashed (`candidates/phi4-mini/profile.json`), and from the Hub API. The route follows `docs/design/generic-genai-chat-route.md`, accepted on 2026-11-04 for the route shape and the generic loader.

| Field | Value |
| --- | --- |
| Candidate | `microsoft/Phi-4-mini-instruct` (profile `phi4-mini-instruct/ortgenai-fp32-cpu`, template `phi4-text-v1`, route `genai-chat-cpu-v1`) |
| Code revision | No upstream modeling code is run or adopted: the reference is the `Phi3ForCausalLM` built into transformers 5.18.0 with `trust_remote_code` off, and the candidate pins no Python file. The bundle comes from the ONNX Runtime GenAI 0.17.1 builder (source commit `83de55a1f3886cddbb32a5517e26adb3f9d65159`). |
| Weight revision | `cfbefacb99257ffa30c83adab238a50856ac3083`: two bf16 safetensors shards, 7.7 GB. Every file's size and SHA-256 is in the candidate contract. |
| Requested by | #109. Hub on 2026-11-04: 340,545 downloads, 880 likes. Its triage calls it a natural step up from Qwen3.5-0.8B. |
| Proposed maturity at first release | Experimental |
| Decision | Admitted at maturity Experimental: `phi4-mini-instruct/ortgenai-fp32-cpu`, greedy text chat, CPU fp32 on Linux x64, prompt plus completion below 4,096 tokens. `Starlings.Models.Phi4` is source only: no Phi-4 package, no registry manifest and no hosted bundle. Development parity passed on all 14 corpus rows; protected G1/G2 did not run. Whether to host a bundle stays open. |
| Decided by and date | Mark, 2026-11-04 |

## Rights (hard blockers)

| Item | License or terms | Redistribution |
| --- | --- | --- |
| Code | MIT (`LICENSE` at the pinned revision, 1,084 bytes, SHA-256 `fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d`; card front matter `license: mit`). `NOTICE.md` at the revision names one third-party component, flash-attention (BSD 3-Clause), which the repository's modeling code uses. | That code is not run or redistributed here. The chat template string is ported to C# and recorded in `ai-dotnet/THIRD-PARTY-NOTICES.md` and `ai-dotnet/PROVENANCE.md`. |
| Weights | MIT, same file and revision. Ungated. | MIT permits redistribution. Nothing is redistributed in this change: no hosted bundle, and hosting stays a maintainer decision. |
| Tokenizer assets | MIT, same repository and revision. `config.json`'s `auto_map` names `Xenova/gpt-4o` for `AutoTokenizer`; it is not used: transformers loads the pinned `tokenizer.json` with remote code off. | Not redistributed: fetched by `tools/fetch-tokenizer-assets.sh`. |
| Evaluation data | Starling-written: `corpora/five-models/chat.jsonl` and `corpora/phi4/template.jsonl`. | Repository content. |

No rights blocker.

## Score

| Criterion | Weight | Score (0 to 5) | Notes |
| --- | --- | --- | --- |
| User demand | 25% | 3 | High Hub usage. |
| Backend feasibility | 20% | 4 | The GenAI builder has a Phi-3 long-rope model class and a `phi3` model type, and the decode loop is the Qwen3.5 one. fp32 is 15.3 GB. |
| Reusable architecture coverage | 20% | 4 | First Phi family model, first o200k tokenizer profile, and the first entry of a data-driven chat route meant for #110. |
| Access and licensing clarity | 15% | 5 | Ungated, MIT, files pinned by commit and SHA-256. |
| Testability | 10% | 3 | Full reference route (transformers, Python GenAI, .NET) on 14 conversations. Long context needs a separate run. |
| Maintenance burden (5 = low) | 10% | 4 | Stable upstream revision (last changed 2025-12-10) and a small template; one new route, one source-only project and a small generic loader. |
| **Weighted total** | | **3.80** | |

## Execution route

ONNX Runtime GenAI 0.17.1, CPU fp32, through `Starlings.Generation.OnnxGenAI`. The bundle comes from the GenAI builder run on the pinned safetensors with a float32 config (`reference/genai-chat/build_bundle.py`); its scratch copy drops `auto_map`, so remote code cannot be reached. Microsoft's own ONNX repositories are not used.

The builder picks `Phi3MiniLongRoPEModel` because `max_position_embeddings` (131,072) differs from `original_max_position_embeddings` (4,096). The graph switches rope caches at position 4,096. The short factors of this checkpoint are all 1.0. `partial_rotary_factor` 0.75 gives 96 of the 128 head dimensions.

GenAI masks a position whose id equals `pad_token_id`, and upstream's pad id is `<|endoftext|>` (199999), so a literal one in a prompt would lose attention. The profile overlays 200063, an unused embedding row (the tokenizer's ids end at 200028), as the Qwen profiles do. The original files are never modified.

`Starlings.Models.Phi4`, source only, holds the definition and embeds the route contract table. Generation goes through the generic `GenAIChatModel` in `Starlings.Generation.OnnxGenAI`, experimental API, which reads `Phi4Checkpoint.ChatDefinition`.

## Tokenizer profile

`phi4-bpe-chat`, first in `Starlings.Tokenizers` 0.3.5 (starlinghq/tokenizers#23; this repository pins 0.3.6): byte-level BPE with the o200k split, and `rstrip` on the role tags. The SHA-256 of `tokenizer.json` is pinned in `Phi4Checkpoint`, and `OpenTokenizer` also checks the special token ids and that row 200063 has no token.

## Feature contract

| Upstream feature | In scope | Evidence source |
| --- | --- | --- |
| Text chat with system, user and assistant roles | Yes, greedy, Experimental | Development parity; template corpus |
| Streaming, usage, stop ids and strings | Yes, Experimental | Development parity (stream, usage, token_stop, string_stop) |
| Cancellation | Before prefill and between decode steps | Development parity (cancellation) |
| Fresh cache per request; request isolation | Yes | Development parity (cache, isolation) |
| Context up to 131,072 tokens | No: prompt plus completion below 4,096 only. Longer context is `untested` (long rope factors); the loader does not refuse it | Corpus rows past the bound are refused by the reference scripts |
| Tool calling (`tools` on the system message) | `unsupported`: the template's tool section is rejected | Template tests |
| Thinking | `unsupported`: the template has no switch | Template tests |
| Multimodal | `unsupported`: text checkpoint | n/a |
| Sampling, logprobs, guided decoding | `untested` on this checkpoint | n/a |
| CUDA, fp16 and quantized bundles | `unsupported`; the loader is CPU fp32 only | n/a |

## Hardware and compute estimate

Linux x64 CPU, two threads, inside a 20 GiB limit. `model.onnx.data` is 15.3 GB. Anonymous memory, sampled every 15 s, peaked at 17.7 GiB in the Python GenAI stage and 17.4 GiB in .NET. The control group, which also counts reclaimable file cache, reached the limit with no out-of-memory kill. A smaller limit was not tried. See the evidence for memory and times. GPU runs stay under #210.

## Evidence

`docs/evidence/phi4-mini-onboarding-2026-11-04/README.md`: the `genai-chat-cpu-v1` route passed development parity on 2026-11-04 on all 14 corpus rows, CPU only, two threads, inside a 20 GiB limit. Against transformers, both Python GenAI and .NET had exact prompt and greedy ids, and initial logits within 2.14e-4 max absolute (bound 5e-3) and 4.27e-5 mean absolute (bound 5e-4). All seven .NET runtime checks held on every row. The run was one invocation with exit 0 and took about 86 minutes on a busy machine. The longest row is 366 tokens. This is development evidence, not release qualification.

## Open risks

- The route is bounded at 4,096 tokens, which is short of what the model card advertises (128K), and the longest evaluated row is 366 tokens (361 prompt, 5 completion), so nothing near the bound has been exercised. A long-context run is separate work.
- Bounds are the route's existing development policy, proposed. No fixture is protected, no tolerance is set, no gate has run.
- 15.3 GB of graph data is a heavy CPU profile. With no hosted bundle, a user must have the snapshot and build it.
- The evaluated corpus is 14 short conversations; the parity check is about faithfulness to the reference, not model quality.
- Whether to host a bundle stays open (`docs/design/generic-genai-chat-route.md`).
