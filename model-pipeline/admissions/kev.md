# Admission: Kev decision-model family

Backlog item P-027 (plan sections 5.1, 6.4, 7 and 18). Pipeline stage 1.

Evidence was gathered on 2026-10-05 from a clone of the upstream repository and from package registries. Direct downloads from huggingface.co were blocked. Text files on the Hub (configs, JSON headers, file listings with sizes, license metadata) were then read through the Hugging Face MCP connector; facts from that check are tagged "[hub, 2026-10-05]". The connector reads the current `main` and returns no commit SHAs or file digests, so revisions and SHA-256 values below are still as recorded upstream. Fields that still need a full download say "needs download". No web-search-only facts are used.

Update, 2026-10-05: Hub downloads now work. Every file below was downloaded at a pinned commit SHA and hashed (`docs/m0/hub-pins.md`); facts confirmed that way are tagged "[download, 2026-10-05]". The spike ran on CPU the same day; results are in "Spike results" at the end.

| Field | Value |
| --- | --- |
| Candidate | Kev-0.8B, `jaredpalmer/kev-0.8b` (code: https://github.com/jaredpalmer/kev). Fallback: Kev-0.6B, `jaredpalmer/kev-0.6b`. Other 1.0 sizes listed below, not in spike scope. |
| Code revision | `6b719c3c3f367295f6ef336f4f751cf5ff970abc` (git tag `kev-1.0`, lightweight; the release record names it as the packaging merge). Inspected HEAD of `main`: `fe64b1274ea7f80d4095866df90666abb03e9cf6` (2026-10-03); since `kev-1.0` only `kev/checkpoint.py` changed in `kev/` (release-date handling). |
| Weight revision | Kev-0.8B weights `9a45d25eb2ab761841196625383fa1dff0e56c1e`; Hub tag `v1.0` points to card-only commit `bf75a6a8848ea6960ff2ed108d9ed44c2941174f` with the same weight bytes. Base `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Source: `runs/release/kev-1.0.json`. Repository exists with `adapter_model.safetensors` (43,338,624 bytes), `head.pt` (2,103,999 bytes), `tokenizer.json`, `tokenizer_config.json`, `adapter_config.json`, `provenance.json` [hub, 2026-10-05]. Commit SHAs and file hashes confirmed [download, 2026-10-05]: `9a45d25e` and `bf75a6a8` (current `main`, also the `v1.0` tag) hold byte-identical weight, head and tokenizer files; base `main` is `dc7cdfe2`. |
| Requested by / demand evidence | Plan section 7 (P1 spike) and backlog P-027. Upstream activity: Kev 1.0 released 2026-10-01; an upstream ExecuTorch example for Kev exists (`pytorch/executorch` `examples/systemone/kev`, seen at `5e21c13cc9e34fa2922d5708ba5962e3e365afdb`). No named .NET consumer yet (P-028). Repository stars and downloads not measured (GitHub API was not reachable for this repository). |
| Proposed owner | Unassigned |
| Proposed maturity at first release | Experimental |
| Decision | Admitted at maturity Experimental. M2 slot stays with GLiNER2; Kev takes it only if the GLiNER2 P-014 spike lands high and an owned Qwen BPE profile covers Kev's `tokenizer.json`; otherwise reconsidered after M3 maintenance cost is measured. Binding tolerances: `model-pipeline/protected/kev-0.8b/onnx-fp32/tolerances.yaml`. |
| Decided by and date | Mark, 2026-10-08 (PR #4) |

### Upstream inventory

Git tags (`git ls-remote --tags`):

| Tag | Object | Commit |
| --- | --- | --- |
| `kev-1.0` | lightweight | `6b719c3c3f367295f6ef336f4f751cf5ff970abc` |
| `kev-family` | `d038913b26aa9c674fdb5961367e94137258b598` (annotated) | `3aa96e53f8ed00a87434d69cb62b24bb16246816` |
| `research-archive-2026-09-24` | `58b766b5d2db3d9d0f0b320db5834729b0614ae2` (annotated) | `41c7b5a384ce93dd88ae0a28c379e947fe716a8d` |
| `v0.1.0` | lightweight | `ac67bf4e52d7bdc8420d8024c396df5585915d8c` |

Kev 1.0 checkpoints (from `docs/releases/kev-1.0.md` and `runs/release/kev-1.0.json`):

| Model | Hub repo | Weights revision | `v1.0` tag commit | Form | Base and revision | Temperature |
| --- | --- | --- | --- | --- | --- | --- |
| Kev-0.8B | `jaredpalmer/kev-0.8b` | `9a45d25eb2ab761841196625383fa1dff0e56c1e` | `bf75a6a8848ea6960ff2ed108d9ed44c2941174f` | LoRA r16 + head | `Qwen/Qwen3.5-0.8B-Base` `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 2.3510958125672174 |
| Kev-4B | `jaredpalmer/kev-4b` | `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` | `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` | LoRA r16 + head | `Qwen/Qwen3.5-4B-Base` `1001bb4d826a52d1f399e183466143f4da7b741b` | 2.406050072164233 |
| Kev-9B (v2) | `jaredpalmer/kev-9b` | `b5d8c18e44c60888d138b65cb6507ff0a5a448a0` | `db029f08b290afd9fee4aa4bbcd9ae48602d1eb0` | LoRA r16 + head | `Qwen/Qwen3.5-9B-Base` `68c46c4b3498877f3ef123c856ecfde50c39f404` | 2.193649959389252 |
| Kev-27B (v2) | `jaredpalmer/kev-27b` | `28be62e9c5ae0471bda2b7c636a55224b4f4b887` | `af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf` | full bf16 weights (51 GB, 11 shards) + head | `Qwen/Qwen3.8-27B` `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 1.319507910772894 |
| Kev-0.6B (previous generation) | `jaredpalmer/kev-0.6b` | not recorded upstream; Hub `main` is `dece6dba8d43f0f7ded45e9f5b9df12474d90843` (2026-09-20) [download, 2026-10-05; API only, files not downloaded] | none | LoRA r16 (alpha 32, attention and MLP projections only) + head | `Qwen/Qwen3-0.6B-Base` `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` (suite manifests; the Hub `provenance.json` and `adapter_config.json` name the base without a revision) [hub, 2026-10-05] | in `head.pt`; needs download |

Recorded SHA-256 for Kev-0.8B at `9a45d25e`: `adapter_model.safetensors` `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a`, `head.pt` `f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6`, `tokenizer.json` `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` (the same `tokenizer.json` hash is recorded for all four 1.0 checkpoints). GitHub release `kev-1.0` asset `kev-0.8b.tar.gz` SHA-256 `0ae144c7675f0c3f333be0bb878a0f202ab9e6fa84169fb7cb16efe6176c1ef1` (not downloaded). All three file hashes match the downloaded files [download, 2026-10-05]; the `tokenizer.json` hash is the Kev repository's copy (19,989,325 bytes), not the base file (`fe000e3e...`, 12,807,196 bytes).

Hub `provenance.json` against the upstream record [hub, 2026-10-05]:

| Item | Hub `provenance.json` | Upstream record above | Result |
| --- | --- | --- | --- |
| Kev-0.8B base | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | same | match |
| Kev-0.8B adapter SHA-256 | `9b908623...885a` | `9b908623...885a` | match |
| Kev-0.8B head SHA-256 | `bc77488acd55d782a34ec875885c8647aa4afbb446fb1c4c3bb35fbc94ce865f` (`measured_checkpoint`, `inference_temperature` 1.0) | `head.pt` `f400bd12...20b6`, temperature 2.3510958125672174 | mismatch, resolved [download, 2026-10-05]: the Hub `head.pt` at `9a45d25e` is `f400bd12...`, the release head. Loaded with `torch.load(weights_only=True)` it carries temperature 2.3510958125672174 and a `temperature_fit` entry. `bc77488a...` is not on the Hub at either commit; it is consistent with the head as training wrote it before the fit was stored (inferred, not verified by bytes). |
| Kev-0.8B training code | `git_commit` `45923b7a3460b6d36358e2e143455902c1eb856b` | tag `kev-1.0` is `6b719c3c...` | different commits. Provenance records the training commit; `kev-1.0` is the later packaging commit. Not an error, but the inference code pin (`6b719c3c`) is not the training code. |
| Kev-0.8B environment | torch 2.8.0+cu128, CUDA, NVIDIA H200, LoRA r16, bf16 training | torch 2.8.0 in `uv.lock` | consistent |
| Kev-4B | `git_commit` `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`, adapter `90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2`, head `979aaa60135274973a6dc57fe24dfc4f1ce88e65df46c73d86b2047c94d897df` | not compared | recorded only |
| Kev-0.6B | `git_commit` `37cb2f6155afec1054de2cc78dfd7c7dd24a956d`, torch 2.8.0+cu128, H100, LoRA r16, bf16, base without revision | weights revision not recorded | the Hub file gives the training commit, not a weights revision |

Kev-0.8B `adapter_config.json` [hub, 2026-10-05]: peft 0.21.0, r 16, alpha 32, dropout 0.05, `task_type` `FEATURE_EXTRACTION`, targets `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj`, `revision` null. This matches the target list in Execution route.

Upstream reference environment at `6b719c3c` (`uv.lock`): Python 3.12 or 3.13, torch 2.8.0, transformers 5.17.0, peft 0.21.0. CUDA serving adds flash-linear-attention 0.5.2.

## Rights (hard blockers)

| Item | License or terms | Redistribution |
| --- | --- | --- |
| Code | Apache-2.0 (`LICENSE` at both inspected commits; `pyproject.toml` `license = "Apache-2.0"`). No `NOTICE` file. | Permitted with license and attribution. We do not plan to adopt upstream source; porting the encoder and answer mapping to C# would be a reimplementation, and any copied code must be recorded in `PROVENANCE.md` and `THIRD-PARTY-NOTICES.md`. |
| Weights | Kev adapter and head: Apache-2.0 per the model card front matter (`license: apache-2.0`, `library_name: peft`, `base_model: Qwen/Qwen3.5-0.8B-Base`) [hub, 2026-10-05] and the release record. The Kev repository has no `LICENSE` file; the license is stated only in card metadata. Kev-0.6B card front matter also says `apache-2.0` [hub, 2026-10-05]. Base Qwen3.5-0.8B-Base: `LICENSE` file is Apache-2.0 [hub, 2026-10-05]. | Apache-2.0 permits redistribution. Not decided whether we redistribute merged weights or fetch from the Hub at a pinned revision (P-009). License text must be pinned with the revision SHA at download. |
| Tokenizer assets | Taken from the base model (`kev/checkpoint.py`: "The tokenizer always comes from the base"). Base repository license is Apache-2.0 [hub, 2026-10-05]. | Same as base weights. |
| Evaluation data | Mixed per source. In git (decision-v7, transfer-v4, transfer-v9, hard-v1, devtools-v1, documents-v1 dev/test): derived from public datasets with their own terms; documents-v1 is CFPB narratives (US government work); devtools-v1 records per-repository licences including GPL and NOASSERTION (`scripts/devtools_v1_licences.json`). breadth-v1, longdoc-v1 and documents-v2 live only in the private dataset `jaredpalmer/kev-private-evals`, because several sources forbid public redistribution (SATA-Bench CC-BY-NC-4.0, SGD CC-BY-SA-4.0, Humicroedit with no licence). | Do not copy suite records into this repository. Reference upstream files by commit and SHA-256 for internal fidelity testing. Private suites are not accessible to us and cannot be part of our evidence. |

No hard blocker found. Licenses are confirmed as Apache-2.0 from Hub metadata and the base `LICENSE` file; admission still needs the license text pinned at the downloaded revision.

## Score

Provisional. Rescored after the spike: backend feasibility 3 to 4 (real-weight parity on both development partitions, C# execution on Linux x64; long states above 8k and other OSes unmeasured). Weighted total 3.45 to 3.65.

| Criterion | Weight | Score (0 to 5) | Notes |
| --- | --- | --- | --- |
| User demand | 25% | 3 | Plan names visible demand and the System One request shape (S17, S18). Upstream ExecuTorch maintainers accepted a native Kev example. No named .NET consumer. |
| Backend feasibility | 20% | 4 (was 3) | A hidden-state ONNX graph for the Qwen3.5 hybrid backbone builds and runs on ORT 1.30 CPU in a random-weight probe (see Execution route). Real-weight parity is unproven. Upstream reports an fp32 ExecuTorch lowering of Kev-0.8B at max abs probability difference 1.06e-5 and 0 flips on 1,264 questions, which shows the computation lowers cleanly with explicit state. Spike 2026-10-05: with real weights, ORT matches PyTorch to 8.4e-5 in probability with 0 flips on both development partitions, and the graph runs from C# on Linux x64. |
| Reusable architecture coverage | 20% | 4 | Same Qwen3.5 backbone class as the pilot LLM default candidate; same Qwen BPE tokenizer family; first user of the classification contract in section 5.1. |
| Access and licensing clarity | 15% | 4 | Apache-2.0 code and weights, confirmed in Hub metadata and the base `LICENSE` [hub, 2026-10-05]; full revision SHAs recorded upstream, not yet checked by download. Evaluation data mixed and partly private. |
| Testability | 10% | 4 | Frozen suites with per-partition SHA-256, dataset and base revisions, and code hashes; release records with fp32 metrics per checkpoint (`runs/release/kev-08b-r15.json`). Raw upstream predictions are not committed, so item-level references must be regenerated (P-003). |
| Maintenance burden (5 = low) | 10% | 3 | Upstream changes daily, but 1.0 pins checkpoints and code. The ORT contrib ops used for DeltaNet are recent. Input encoding and answer mapping must be ported exactly. |
| **Weighted total** | | **3.65** (was 3.45) | 0.75 + 0.80 + 0.80 + 0.60 + 0.40 + 0.30 |

## Execution route

### What the model computes (from `kev/model.py`, `kev/api.py`, `kev/checkpoint.py` at `6b719c3c`)

- Backbone: `AutoModelForCausalLM.from_pretrained(base).model`, the text decoder without the LM head, run prefill-only. Kev-0.8B: 24 layers, 18 Gated DeltaNet and 6 full attention (model card; confirmed by base `config.json` `layer_types`, every fourth layer `full_attention`, `full_attention_interval` 4 [hub, 2026-10-05]). Base text config [hub, 2026-10-05]: hidden 1024, intermediate 3584, 8 attention heads, 2 KV heads, `head_dim` 256, `attn_output_gate` true, linear attention 16 key and 16 value heads of dim 128, `linear_conv_kernel_dim` 4, `rms_norm_eps` 1e-6, vocab 248,320, `tie_word_embeddings` true, `mtp_num_hidden_layers` 1 (a multi-token-prediction layer that Kev does not use). LoRA rank 16, alpha 32, on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` plus the DeltaNet projections `in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj`. The loader merges the adapter into the base in fp32 (`merge_and_unload`) by default.
- Hidden states used: `last_hidden_state` of the backbone (after the final RMSNorm), cast to fp32. Only two kinds of positions are read: each question's `<decide>` token and each option's closing `</opt>` token.
- Marker tokens: existing Qwen special tokens reused as delimiters, no new embeddings. state = `<|fim_prefix|>`, question = `<|fim_middle|>`, option open = `<|box_start|>`, option close = `<|box_end|>`, decide = `<|fim_suffix|>`. Token IDs [hub, 2026-10-05], same in the base and Kev-0.8B `tokenizer.json`: `<|fim_prefix|>` 248060, `<|fim_middle|>` 248061, `<|fim_suffix|>` 248062, `<|box_start|>` 248049, `<|box_end|>` 248050. Pad is `<|endoftext|>` 248044 (Kev `tokenizer_config.json`). Kev-0.6B (Qwen3 vocabulary, `added_tokens.json`): 151659, 151660, 151661, 151648, 151649.
- Sequence: `[<state>] + state_tokens`, then per question `[<q>] + instruction_tokens + ([<opt>] + option_tokens + [</opt>])* + [<decide>]`. Each text segment is tokenized separately with `add_special_tokens=False`; markers are inserted by ID.
- Pointer head: `q, k = Linear(d, 256)` with bias; logits = `(k(h_opts) @ q(h_decide)) / sqrt(256) / T`; softmax over the question's options. `T` is stored in `head.pt` (a torch pickle with the head state dict and metadata). The .NET bundle needs `head.pt` converted to safetensors plus JSON at bundle-build time.
- Position IDs and isolation: on attention-only bases (Qwen3, Kev-0.6B) all questions share one packed sequence with a block-causal 4D mask, and each question's positions restart at `len(state)`. On hybrid bases (every Kev 1.0 model) DeltaNet layers cannot honour that mask, so each question runs as its own causal row `state + branch`. In that row the positions are simply `0 .. len-1`. The "restart" is then nothing more than starting every row from the same cached state. The attention-only path also switches to rows above 16,384 tokens, and upstream tests that rows and packed form agree to fp32 rounding.
- Serving path: run the state once, keep the attention KV plus per-DeltaNet-layer conv state and recurrent state, then run the question rows (right-padded, up to 16,384 tokens per pass) continuing from a copy of that state. `forward()`, which produced every published number, runs the state once per question instead; upstream reports the two agree to fp32 rounding.

Consequence for the plan: Kev-0.8B needs a standard causal decoder forward that outputs final hidden states and can carry state between calls. No custom mask and no position remapping. Plan section 7's sentence about restarting position IDs per question describes the packed form, which no Kev 1.0 model uses.

### Route A (primary): ONNX graph from the ONNX Runtime GenAI model builder, run with direct ONNX Runtime C#

- `onnxruntime-genai` 0.17.1 (PyPI) has a builder for `Qwen3_5ForConditionalGeneration` with `exclude_lm_head=true`, which replaces logits with a `hidden_states` output. DeltaNet layers become `com.microsoft` contrib ops `CausalConvWithState`, `LinearAttentionGate`, `LinearAttention` and `GatedRMSNorm`; attention layers use `GroupQueryAttention` with `MRotaryEmbedding`. The non-paged CPU export uses `LinearAttention`; the `GatedDeltaNet` op is CUDA-only.
- The graph is a plain ONNX model. It can run in `Microsoft.ML.OnnxRuntime` without the GenAI generation loop, so it falls under the "Direct ONNX Runtime C#" row of plan section 5.2, and the owned tokenizer feeds token IDs directly.
- Probe run here (random weights, tiny config: 3 DeltaNet layers and 1 attention layer, fp32, ORT 1.30.0 CPU, transformers 5.17.0 eager as reference). Graph I/O: `input_ids [B,S]`, `attention_mask [B,total]`, `position_ids [3,B,S]`, `past_key_values.N.key/value`, `past.N.conv`, `past.N.recurrent`; outputs `hidden_states [B,S,d]` plus the matching `present.*`. Max abs hidden-state difference against PyTorch: up to 2.9e-4 at hidden magnitude about 5 for single rows of 5 to 1,000 tokens; 1.6e-5 to 3.0e-5 for a 150-token state followed by a 7 to 70-token branch from the carried state; up to 1.9e-4 for a right-padded batch of 3 rows. This shows the I/O contract matches Kev's row-plus-prefix computation. It says nothing about fidelity with real weights, where differences of this size could move probabilities; the spike measures that.
- Native support: the `Microsoft.ML.OnnxRuntime` 1.30.0 NuGet package's `linux-x64` `libonnxruntime.so` contains float kernels for `LinearAttention`, `CausalConvWithState`, `LinearAttentionGate`, `GatedRMSNorm`, `MRotaryEmbedding` and `GroupQueryAttention` (symbol-level check only; not yet executed from C#). Windows and macOS binaries not checked.
- Builder fit, resolved [hub, 2026-10-05]: the base `config.json` has `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type` `qwen3_5`, with a `text_config` (`qwen3_5_text`) and a `vision_config` (12-layer ViT). So the builder's only accepted architecture is the base's own, and a merged Kev backbone can be saved by copying the base checkpoint and replacing the language-model tensors (spike step 3); the vision tensors stay in the checkpoint and are unused. Rotary settings (`text_config.rope_parameters`): `rope_type` `default`, `rope_theta` 10,000,000, `partial_rotary_factor` 0.25 (rotary dim 64 of `head_dim` 256), `mrope_interleaved` true, `mrope_section` [11, 11, 10] (sums to 32, half the rotary dim). This is the 0.25 case that ran in the probe; the spike must still confirm the graph's rotary dim is 64. `max_position_embeddings` is 262,144, above the 73,728 positions Kev needs (65,536 + 8,192).
- Open point: the builder's handling of the MTP layer and vision tensors in a merged checkpoint is unverified until the spike export.
- Cannot provide: nothing Kev needs is missing in principle. Tokenizer and template are not involved (Kev has no chat template). Generation-loop features of GenAI are unused.

### Route A2: `torch.onnx.export` of the backbone

- Legacy TorchScript exporter (torch 2.8.0, opset 18): fails on the depthwise causal conv (`expected padding to be a single integer value ... got padding=[]`).
- Dynamo exporter (`dynamo=True`, torch 2.8.0): succeeds, but the sequence length is specialized to the example length even with `Dim` declared. transformers 5.17.0 switches the chunked delta rule to a forward-substitution form under export and keeps a Python loop over chunks, which is unrolled. The tiny probe graph was 9.3 MB, took 234 s to export, ran only at the traced length, and matched PyTorch to 1.7e-4. A usable graph would need one export per length bucket, or a rewritten backbone with explicit state and a single delta-rule op, as the ExecuTorch example does (`examples/systemone/kev/model.py` calls `torch.ops.llama.gated_delta_rule`). For ONNX that op would have to be mapped to `com.microsoft::LinearAttention` through a custom translation. Second choice after route A.
- optimum-onnx 0.1.0 has no Qwen3.5 support (no matches in the wheel).

### Route B (fallback): TorchSharp port

- TorchSharp 0.107.0 on NuGet requires libtorch 2.10.0 (`libtorch-cpu` or `libtorch-cuda-12.8*`).
- Work: port the Qwen3.5 text decoder (offset RMSNorm `1 + w`, gated attention with q/k norm and output gate, partial RoPE with interleaved MRoPE sections that collapse to plain RoPE for text, GQA, depthwise causal conv k=4 with SiLU, L2-normalized q/k, gated delta rule, gated RMSNorm, SwiGLU), a safetensors loader, and LoRA merge or a pre-merged bundle. Estimated several hundred lines of C#. The delta rule can start as the per-token recurrence (simple, slow on long states); the chunked form needs a triangular solve, and TorchSharp support for `linalg.solve_triangular` is unverified.
- Costs: large native dependency (libtorch), no Native AOT, version pairing between TorchSharp and libtorch, and a maintained architecture port. Fidelity should be close to PyTorch because the kernels are the same library.

### Route C (fallback): Kev-0.6B on Qwen3

- Attention-only backbone. The builder supports `Qwen3ForCausalLM` with `exclude_lm_head`, and the row form avoids the packed 4D mask, which `GroupQueryAttention` does not accept.
- Drawbacks: weights revision not recorded upstream (Hub commit SHA needs download; Hub `provenance.json` gives only training commit `37cb2f61...`); no 1.0 tag; upstream no longer develops it; weaker out of domain (transfer-v4 test 0.642 against 0.697 for Kev-0.8B, per README); trained on states of at most 384 tokens; uses the Qwen3 tokenizer, not the Qwen3.5 one, so it does not share the pilot LLM's tokenizer profile.

### Other upstream backends (not routes for us)

CUDA serving uses flash-linear-attention Triton kernels and CUDA graphs; Apple Silicon uses MLX (mlx-lm 0.31.3). Upstream reports bf16 serving differs from the fp32 evaluation path by up to about 0.03 (GPU) and 0.05 (Mac) in probability. An unmerged upstream branch `origin/executorch/export-contract` (`e023a1557c167252b62b6553fc9b0ad8b4f1ba1e`) records ExecuTorch parity: Kev-0.8B at `9a45d25e`, XNNPACK fp32, decision-v7 development, 1,264 questions: max abs probability difference 1.06e-5, 0 argmax flips (`runs/et-parity-0.8b-r15-xnnpack-fp32/report.json` on that branch). The ExecuTorch example README pins a different Kev-0.8B revision (`54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8`), not the 1.0 weights.

## Tokenizer profile

- Algorithm: byte-level BPE (Qwen family). transformers 5.17.0 `Qwen3_5Tokenizer` defines an NFC normalizer, a regex split `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`, a ByteLevel pre-tokenizer without prefix space, a BPE model and a ByteLevel decoder. This regex differs from the Qwen2/Qwen3 one (it adds `\p{M}`).
- Base `tokenizer.json` [hub, 2026-10-05; confirmed at `dc7cdfe2` by download, SHA-256 `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`]: 12,807,196 bytes, version 1.0. Normalizer `NFC`. Pre-tokenizer `Sequence[Split(regex above, with \p{M}, behavior Isolated, invert false), ByteLevel(add_prefix_space false, trim_offsets false, use_regex false)]`. Post-processor `ByteLevel` (add_prefix_space false, trim_offsets false, use_regex false). Decoder `ByteLevel` (add_prefix_space false, trim_offsets false, use_regex false). Model `BPE`, `unk_token` null, `byte_fallback` false, empty prefix and suffix. So the base file matches the class default.
- Kev-0.8B repository `tokenizer.json` is a different file [hub, 2026-10-05]: 19,989,325 bytes, and its Split regex is the Qwen2/Qwen3 one without `\p{M}` (`[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*`), its pre-tokenizer ByteLevel has `trim_offsets` true, its decoder ByteLevel has `add_prefix_space` true and `use_regex` true, and its BPE model adds `ignore_merges` false. Its `tokenizer_config.json` (`tokenizer_class` `Qwen2Tokenizer`, `backend` `tokenizers`) carries a `pretokenize_regex` that does include `\p{M}`. The two files tokenize text with combining marks differently. Resolved at runtime [download, 2026-10-05]: Kev loads `AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base", revision="dc7cdfe2...")`, which under transformers 5.17.0 returns `Qwen2Tokenizer` (named by the base `tokenizer_config.json`). transformers 5 builds the pipeline from the class, and the live `backend_tokenizer` equals the Kev repository's `tokenizer.json` in every field (normalizer, the regex without `\p{M}`, ByteLevel options, 33 added tokens, vocabulary, merges). The base file's `\p{M}` regex is never applied. On the M0 tokenizer corpus the base file and the runtime tokenizer give different ids for 2 of 46 cases (Devanagari and Thai). The profile to implement is therefore the Kev repository's `tokenizer.json` (`06b95093...`), which is also the hash upstream records.
- Asset: `tokenizer.json` from `jaredpalmer/kev-0.8b` at `9a45d25e` (SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`, 19,989,325 bytes), the effective runtime tokenizer (previous bullet). Upstream records that hash for all four 1.0 checkpoints. The base file (`fe000e3e...`) is not what runs.
- Components Kev uses: encode only, `add_special_tokens=False`, on each segment separately. Added-token matching applies to user text. Kev's only escaping is `re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text)`: any `<|name|>` in user text becomes `<¦name¦>` (U+00A6) before tokenization, so user text cannot produce the five delimiter tokens or other `<|...|>` specials. Added tokens of other shapes still match in user text [hub, 2026-10-05]: the base file has 22 added tokens (IDs 248044 to 248065), of which `<tool_call>` 248058 and `</tool_call>` 248059 are not of the `<|name|>` shape (both `special` false). The Kev-0.8B copy has 33 (to 248076) and adds `<tool_response>` 248066, `</tool_response>` 248067, `<think>` 248068, `</think>` 248069, `<tts_pad>` 248072, `<tts_text_bos>` 248073, `<tts_text_eod>` 248074, `<tts_text_bos_single>` 248075, plus `<|audio_start|>`, `<|audio_end|>`, `<|audio_pad|>`. User text containing `<tool_call>` encodes to ID 248058, and with the runtime tokenizer `<think>` encodes to 248068 (checked [download, 2026-10-05]). The fim markers are `special` false and the box markers `special` true. Decode is not used.
- Text rendering before tokenization (`kev/api.py` `render`): `None` becomes empty; scalars use Python `str()` (so `True`, `False`, and Python float formatting); lists become `- item` lines; objects become `key: value` lines with two-space indentation per level. Choice options are `name` or `name: description`; Noul options are `no` and `yes` with optional descriptions; Score options are the rendered levels. The C# port must reproduce Python `str()` for numbers and booleans exactly.
- Existing owned profile: none yet (`tokenizers-dotnet/profiles/` holds only a README). The planned Qwen BPE profile for the pilot LLM covers Kev-0.8B only if the pilot pins a Qwen3.5 model with the same `tokenizer.json`. Kev-0.6B would need a separate Qwen3 profile.
- Upstream notes the ExecuTorch C++ tokenizer once dropped accents during NFC normalization (fixed upstream). Our conformance corpus should include combining-mark and NFC cases.

## Feature contract

| Upstream feature | In scope | Evidence source |
| --- | --- | --- |
| Request with one `state` (string, object or array) and many named questions | Yes, as the in-process classification contract of plan 5.1 | `kev/api.py` `SystemOneRequest`, `render`, `to_record` |
| Noul question: probability of yes | Yes | `kev/api.py` `to_answers`; README API section |
| Choice question: 1 to 255 options, argmax, probabilities, confidence `(p_max - 1/K)/(1 - 1/K)` | Yes | `kev/api.py` `choice_confidence` |
| Score question: 1 to 255 levels, expected level, legend, probabilities, confidence | Yes | `kev/api.py` `score_confidence` |
| Probabilities rounded to 4 decimals | Yes | `kev/api.py` `round_prob` |
| Calibrated temperature from `head.pt` (raw with T = 1) | Yes | `kev/checkpoint.py` `load`; `kev/model.py` `PointerHead` |
| Delimiter escaping of user text | Yes | `kev/model.py` `user_tokens` |
| Question isolation (each question its own row on the shared state) | Yes | `kev/model.py` `rows_of`, `forward_rows_batch` |
| State prefix reuse across requests | Experimental | `kev/model.py` `prefix`, `_branch_rows_from_prefix`; `kev/serve.py` `PrefixCache` |
| States up to 65,536 tokens with 422-style refusal above | Untested above 8,192 (the validated context length); refuse, never truncate | `kev/model.py` `admit`; Kev-0.8B card, Long documents |
| fp32 evaluation path | Yes, the reference profile | README Serving Performance; `kev/checkpoint.py` `LoadOptions` |
| bf16 serving | Unsupported at first release | README Serving Performance |
| `usage.input_tokens` and `usage.output_tokens` | Experimental (`output_tokens` needs exact Python `json.dumps` formatting) | `kev/serve.py`; `kev/api.py` `output_tokens` |
| HTTP server, `/v1/models`, bearer auth, `x-typesafe-request-id` | Unsupported (library first; adapters later) | `kev/serve.py` |
| `/v1/systemone/permute` and `/v1/systemone/separate` demo endpoints | Unsupported | `kev/serve.py` |
| `KEV_DATE_FACTS` date preprocessing | Untested | `kev/api.py` `with_date_facts` |
| `KEV_TRUNCATE_STATES` truncation | Unsupported | `kev/serve.py` |
| Option isolation mode | Unsupported (not available on hybrid bases upstream) | `kev/model.py` `DecisionModel.__init__` |
| Fine-tuning, training, calibration fitting | Unsupported | `kev/train.py`, `scripts/calibrate_checkpoint.py` |
| Kev-4B, Kev-9B, Kev-27B | Untested (same architecture class; 27B is full bf16 weights) | Release notes |

## Hardware and compute estimate

- Kev-0.8B weights: base `model.safetensors-00001-of-00001.safetensors` is 1,746,942,600 bytes in bf16 [hub, 2026-10-05], about 0.87B parameters including the vision tower and MTP layer, so about 3.5 GB in fp32 for the full checkpoint (less for the text backbone alone). Adapter 43 MB, head 2 MB. Upstream measured 3.8 GB resident on an L4 in bf16 and reports a 12 GB peak for the ExecuTorch fp32 export at 0.8B.
- Spike machine: x86-64 CPU, 32 GB RAM, about 20 GB disk for base, merged fp32 copy and ONNX files. No GPU required. A CUDA GPU is optional for comparing against the upstream CUDA kernels.
- Reference runs: decision-v7 development (1,264 questions) and transfer-v4 development (656 questions) on CPU fp32 PyTorch and on ORT. Expect hours, not days, on one CPU machine. A few 8,192-token states from documents-v1 development add memory and latency evidence.
- Kev-27B would need an 80 GB GPU or about 51 GB of host memory plus working memory; out of scope.

## Open risks

- Hub access: resolved 2026-10-05 (`docs/m0/hub-pins.md`).
- Head hash mismatch: resolved; `head.pt` is `f400bd12...`, the release head (Rights and provenance table).
- Tokenizer file mismatch: resolved in the other direction from what this record assumed. The runtime tokenizer is the Kev repository's file, not the base file (Tokenizer profile). A consumer that loads the base `tokenizer.json` with the `tokenizers` library, instead of through transformers 5.17.0, tokenizes combining marks differently from Kev.
- `head.pt` is a torch pickle. The spike must load it with `torch.load(weights_only=True)` or convert it in an isolated step; never unpickle it with full permissions.
- Real-weight fidelity of the ORT contrib ops (`LinearAttention` and friends) is unmeasured. The random-weight probe shows differences up to 2.9e-4 in hidden states; upstream ExecuTorch fp32 reached 1e-5 in probabilities, which sets a realistic bar.
- The CPU PyTorch reference uses transformers' torch fallback for DeltaNet. Published numbers came from CUDA runs with flash-linear-attention, and upstream found kernel changes moved Kev-27B v1 probabilities by 0.03 to 0.06. Our reference must be regenerated on a recorded environment, and published aggregates used only as a cross-check.
- The ORT GenAI builder and the contrib ops are new and change between releases. Pin the builder version and ORT version together and record the graph hash.
- Long states: CPU cost of 8k to 64k states through `LinearAttention` and GQA is unknown. Only 8,192 tokens is validated upstream for Kev-0.8B.
- Exact porting of Python behaviour: `render` uses Python `str()`; `output_tokens` uses `json.dumps`. Mismatches change token IDs or usage counts.
- Upstream moves daily. Pin by SHA; do not track `main`.
- Kev-0.8B weaknesses upstream lists (below-chance When2Call tool routing, date arithmetic, knowledge questions) carry over and must appear in our capability descriptor.
- Plan section 7 says "restarts position IDs for each question" and that generation-loop backends cannot provide hidden states. Both are accurate in general but the route above shows the hybrid row form needs neither a custom mask nor a generation loop. Section 7 may need a wording update after the spike (not changed here).

## Spike plan

Run once Hub downloads work. Work in a scratch directory; do not commit weights or ONNX files. Record versions and SHA-256 of every downloaded or produced file.

### 1. Acquire and verify

```sh
hf download jaredpalmer/kev-0.8b --revision 9a45d25eb2ab761841196625383fa1dff0e56c1e --local-dir kev-0.8b
hf download Qwen/Qwen3.5-0.8B-Base --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 --local-dir qwen3.5-0.8b-base
sha256sum kev-0.8b/adapter_model.safetensors kev-0.8b/head.pt qwen3.5-0.8b-base/tokenizer.json
# expect 9b908623...885a, f400bd12...20b6 (or bc77488a...865f, see Open risks), 06b95093...e523
sha256sum kev-0.8b/tokenizer.json   # record; differs in size from the base file
```

Record: the Hub commit SHAs actually downloaded; both LICENSE files; confirm the `config.json` values and token IDs recorded above [hub, 2026-10-05] against the downloaded files; load `head.pt` with `torch.load(..., weights_only=True)`. For the fallback, also download `jaredpalmer/kev-0.6b` and `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd` and record the Kev-0.6B commit SHA.

### 2. Reference (P-003 runner)

```sh
git clone https://github.com/jaredpalmer/kev && cd kev && git checkout 6b719c3c3f367295f6ef336f4f751cf5ff970abc
uv sync --frozen --extra serve
uv run python -m kev.benchmark --run ../kev-0.8b --suite evals/v7/decision-v7 --device cpu --out runs/ref-decision-dev
uv run python -m kev.benchmark --run ../kev-0.8b --suite evals/v4/transfer-v4 --device cpu --out runs/ref-transfer-dev
```

(`--run` with a local directory avoids a second Hub fetch; the loader still reads the base from the Hub cache at the recorded revision.) Default `LoadOptions` is the fp32 exact path. Keep `predictions.jsonl` (per-question probabilities). Cross-check aggregates against `runs/release/kev-08b-r15.json`: decision-v7 development accuracy 0.8267405063291139 (n = 1,264), transfer-v4 development accuracy 0.6478658536585366 (n = 656). Explain any difference by environment (CPU torch fallback against CUDA kernels).

Also dump a fixture subset: for 64 records, token IDs, positions, and fp32 hidden states at every `<decide>` and `</opt>` index, from both `forward()` (state repeated per row) and `probs()` (state cached). Add four documents-v1 development records truncated to about 2,048 and 8,192 tokens.

### 3. Merge

Load with `Checkpoint(...).load("cpu", LoadOptions(dtype=torch.float32))`, which merges LoRA in fp32. Save the merged backbone in the layout the builder accepts: if the base is `Qwen3_5ForConditionalGeneration`, copy the base checkpoint and replace the language-model tensors with the merged ones. Save the pointer head (`q.weight`, `q.bias`, `k.weight`, `k.bias`) and temperature as safetensors plus JSON. Check that the saved merged model reproduces the step 2 fixture probabilities to max abs difference 1e-6.

### 4. Export (route A)

```sh
python -m onnxruntime_genai.models.builder -i merged -o kev-0.8b-onnx-fp32 -p fp32 -e cpu -c cache \
  --extra_options exclude_lm_head=true
```

Pin `onnxruntime-genai==0.17.1` and `onnxruntime==1.30.0` (or the versions current at spike time, recorded). Record the graph inputs and outputs, op counts by domain, and SHA-256 of `model.onnx` and `model.onnx.data`. Check `genai_config.json` context length against 73,728 positions.

### 5. Compare in Python ORT (CPU execution provider)

For each comparison report max and mean abs difference, the error distribution, and argmax flips:

1. Single rows without cache (the `forward()` layout) on the 64 fixtures: hidden states at readout indices and probabilities.
2. State prefix then question rows from the carried state, including N rows on a replicated state (the `probs()` layout).
3. Right-padded batches of rows of different lengths, against each row alone (batch independence).
4. Full decision-v7 and transfer-v4 development partitions: probabilities and accuracy against step 2.
5. The 2,048 and 8,192-token states: probabilities, peak RSS and latency. One 32,768-token state for memory and time only.
6. NaN and Inf scan on all outputs.

### 6. C# smoke

In a throwaway console project outside `ai-dotnet/` (no backlog item creates a Kev project yet), load the graph with `Microsoft.ML.OnnxRuntime` 1.30.0, feed the fixture token IDs from step 2, apply the pointer head in C#, and compare with the Python ORT outputs. This verifies the NuGet native library executes the contrib ops, which so far is a symbol-level check only.

### 7. Fallbacks, only if route A fails parity

- Route A2: export an explicit-state backbone wrapper (pattern of ExecuTorch `examples/systemone/kev/model.py`) with `torch.onnx.export(dynamo=True)`, mapping the delta rule to `com.microsoft::LinearAttention`; repeat step 5.
- Route B: TorchSharp 0.107.0 with libtorch-cpu 2.10.0; port one DeltaNet layer and one attention layer and compare layer outputs with PyTorch on the fixtures before porting the rest.
- Route C: run steps 1 to 6 for Kev-0.6B with `Qwen3ForCausalLM`, using the row form.

Stop after two unsuccessful repair attempts on the same failure and record the evidence with a minimal reproducer.

### 8. Pass criteria

These are proposals. A maintainer sets the binding values in a protected tolerance file before step 5 runs (plan 10.1); implementation work does not change them.

- fp32 ORT against fp32 PyTorch on the same CPU, decision-v7 and transfer-v4 development: max abs probability difference at most 1e-4, and no argmax flip except on questions whose reference top-two margin is below 1e-4 (each listed). Upstream ExecuTorch fp32 reached 1.06e-5 and 0 flips, so a result far above that needs an explanation even if it passes.
- Cached-state rows against uncached rows in ORT: max abs probability difference at most 1e-5.
- Batch independence: each row's probabilities in a padded batch within 1e-5 of the row alone.
- Accuracy on both development partitions identical to the step 2 reference, apart from listed near-tie flips.
- 8,192-token state completes on the spike machine without NaN or Inf; peak memory and latency recorded.
- C# outputs equal Python ORT outputs within 1e-6.

### 9. Decision

Go for the M2 slot if route A (or A2) meets every criterion and the owned Qwen BPE profile can cover the pinned `tokenizer.json`. Otherwise record which route failed, at which step, with the numbers, and either fall back (B or C) with its own result or record no-go. Update this file: fill the Hub-dependent fields, rescore, set Decision, and add the decider and date.

## Spike results

Run on 2026-10-05 on a 4 vCPU Intel Xeon @ 2.10GHz, 15 GB RAM, no GPU. Scripts: `model-pipeline/reference/kev/` (`kev_fixtures.py`, `kev_merge.py`, `kev_ort.py`). Downloaded weights, merged checkpoint, ONNX files and per-record outputs stay under `.models/` and are not committed. Pass thresholds are the provisional ones from step 8 above, labeled provisional: no maintainer has set binding values in a protected tolerance file, so these results are reported against them and do not change them.

### Environment

- Kev code at `6b719c3c3f367295f6ef336f4f751cf5ff970abc`. Packages from Kev's `uv.lock` (`uv export --frozen --no-dev`), with one deviation: torch 2.8.0 is the CPU build from `download.pytorch.org/whl/cpu` instead of the lock's CUDA wheel and NVIDIA libraries (not needed on CPU; the CPU build of the same release). Python 3.12.3, transformers 5.17.0, peft 0.21.0. DeltaNet runs transformers' torch fallback (`torch_chunk_gated_delta_rule`, `causal_conv1d_fn`); attention is eager. Added to the same environment, with no other version changed: onnxruntime 1.30.0, onnxruntime-genai 0.17.1, onnx 1.23.1, onnx-ir 1.0.0.
- `head.pt` loaded only with `weights_only=True`: every Kev process ran with `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`, and the scripts refuse to start without it. The base was served from a local Hugging Face cache at `dc7cdfe2` with `HF_HUB_OFFLINE=1`.

### Step 1: acquire and verify

Done (`docs/m0/hub-pins.md`). All recorded hashes match: adapter `9b908623...`, `head.pt` `f400bd12...` (the release head, temperature 2.3510958125672174), `tokenizer.json` `06b95093...` (the Kev repository copy, which is also the tokenizer Kev builds at runtime; see Tokenizer profile). Base `config.json` values and marker token IDs match the [hub] values above.

### Step 2: reference

| Partition | Records | Questions | Accuracy (clean, knowable) | Release record (`runs/release/kev-08b-r15.json`) | Wall time |
| --- | ---: | ---: | --- | --- | --- |
| decision-v7 development | 1,204 | 1,468 | 0.8267405063291139 (n = 1,264) | 0.8267405063291139 (n = 1,264) | 55 min |
| transfer-v4 development | 764 | 764 | 0.6478658536585366 (n = 656) | 0.6478658536585366 (n = 656) | 31 min |

The CPU fp32 reference reproduces both published accuracies exactly, although the published numbers came from CUDA with flash-linear-attention kernels. No record was rejected or truncated.

Fixture subset: 64 decision-v7 development records spaced evenly over the partition (all 12 sources, 79 questions, states up to 271 tokens, branches up to 853), with token ids, positions and fp32 hidden states at every `<decide>` and `</opt>`, in both the `forward()` layout (state repeated per row) and the `probs()` layout (state cached). The two layouts differ by at most 1.79e-6 in probability in PyTorch itself. Long states: two documents-v1 development records cut to 2,048 state tokens, and two synthetic 7,552-token states joined from development documents (no development state is longer than 5,001 tokens; 7,552 is upstream's 8k training state size, and the 8k serving context caps state plus branch at 8,192). PyTorch on CPU: about 42 s per 2k record and 110 s per 7.5k record in the `forward()` layout, 9.6 GB peak RSS at 7.5k.

### Step 3: merge

`kev_merge.py` takes Kev's own fp32 merge (`Checkpoint.load`, `merge_and_unload`) and writes a full `Qwen3_5ForConditionalGeneration` checkpoint in fp32 (3,493,815,280 bytes; vision tower and MTP layer upcast and kept), with `config.json` set to float32 so the builder does not round the merged weights back to bf16. The head goes to `head.safetensors` plus `head.json`.

- All 320 language-model tensors are bit-identical to Kev's in-memory merge.
- Recomputing the 79 fixture questions from the saved checkpoint: max abs probability difference 1.79e-7 with the rows batched and padded the way Kev's `_rows_hidden` runs them (provisional bar 1e-6: pass). With each row run alone the difference is 1.13e-6: PyTorch fp32 itself moves by about 1e-6 between batch layouts, which bounds how tight any cross-runtime threshold can usefully be.

### Step 4: export (route A)

`python -m onnxruntime_genai.models.builder -i kev-merged-fp32 -o kev-onnx-fp32 -p fp32 -e cpu --extra_options exclude_lm_head=true exclude_mtp=true`, onnxruntime-genai 0.17.1.

- First attempt failed: the builder enables MTP export when `mtp_num_hidden_layers` > 0 and refuses to combine it with `exclude_lm_head` (`ValueError: Qwen3.5 MTP export cannot be combined with exclude_lm_head`). Repair 1: `exclude_mtp=true` (Kev does not use the MTP layer). The export then succeeded in about one minute. This was the only repair on route A.
- Graph: opset 22 plus `com.microsoft` 1. Inputs `input_ids [B,S]`, `attention_mask [B,total]`, `position_ids [3,B,S]`, KV cache `past_key_values.{3,7,11,15,19,23}.key/value [B,2,past,kv_cache_dim]` (the last dim is symbolic; 256 for an empty cache), DeltaNet state `past.N.conv` and `past.N.recurrent [B,16,128,128]` for the 18 other layers. Outputs `hidden_states [B,S,1024]` and the matching `present.*`.
- Ops: 18 each of `LinearAttention` (`update_rule` gated_delta), `CausalConvWithState`, `LinearAttentionGate`, `GatedRMSNorm`; 6 `GroupQueryAttention` (8 heads, 2 KV heads, scale 1/16); 12 `MRotaryEmbedding`; rotary cache `[262144, 32]`, so the rotary dim is 64 as expected. `genai_config.json` context length 262,144, above the 73,728 positions Kev needs.
- SHA-256: `model.onnx` `55dc88d21208af6c3e86550bc10e0072e43e11ada0f386275bd1f959794691f2` (307,884 bytes), `model.onnx.data` `b52e656289cdb18508cb232ed0f7f7512de56d84c3733a2b05efa45cdee13bc0` (3,077,177,344 bytes). The builder also wrote the 19,989,325-byte Kev `tokenizer.json`, the runtime tokenizer again.

### Step 5: Python ONNX Runtime (CPU EP) against PyTorch

| Check | Result | Provisional bar | Outcome |
| --- | --- | --- | --- |
| 1. Single rows, no cache, 79 fixture questions | probability max 4.35e-5 (mean 4.7e-6, p99 3.2e-5); hidden states at readouts max 2.16e-4; 0 flips | 1e-4, no flips | pass |
| 2. State prefix, rows from the carried state, against ORT single rows | max 6.6e-7 | 1e-5 | pass |
| 2b. Same, against PyTorch `probs()` layout | max 4.28e-5 | (as 1) | pass |
| 3. Right-padded batches of a record's rows against each row alone | 0.0 on 25 questions | 1e-5 | pass |
| 4a. decision-v7 development, all 1,468 questions, `forward()` layout | probability max 8.43e-5 (mean 4.3e-6, p99 3.2e-5); 0 flips; accuracy 0.8267405063291139 in both | 1e-4, no flips, identical accuracy | pass |
| 4b. transfer-v4 development, all 764 questions | probability max 5.83e-5 (mean 6.6e-6, p99 2.4e-5); 0 flips; accuracy 0.6478658536585366 in both | 1e-4, no flips, identical accuracy | pass |
| 5. 2k states (2 records, 4 questions) | max 9.1e-6, 0 flips; state pass 12.5 s, row 13 s; peak RSS 3.1 GB | completes, finite | pass |
| 5. 7.5k synthetic states (2 records) | max 3.1e-5, 0 flips; state pass 51 to 53 s; peak RSS 6.3 GB | completes, finite | pass |
| 5. 32,768-token synthetic state (memory and time only) | process killed without output, consistent with running out of memory | memory and time only | not measured |
| 6. NaN and Inf scan | none in any output | none | pass |

The decision-v7 ORT pass took 25 min (2 threads, sharing the CPU with the transfer-v4 reference) against 55 min for the PyTorch reference; transfer-v4 took 4.7 min alone against 31 min. The 32k attempt fits a CPU `GroupQueryAttention` that materializes the full score matrix (8 heads × 32,768² × 4 bytes is about 34 GB, against 15 GB here); that explanation is an inference, not profiled. Long states above 8k need either more memory or a chunked prefill (the state can run in pieces through the carried cache), which was not tried.

### Step 6: C# smoke

A throwaway console project in the session scratchpad (not in `ai-dotnet/`; no backlog item creates a Kev project), `net10.0`, `Microsoft.ML.OnnxRuntime` 1.30.0 from NuGet, a minimal safetensors reader for the head, and the pointer head written in C#. On 24 single-row fixture questions it loaded the graph (51 inputs, 49 outputs), ran the contrib ops on the NuGet `linux-x64` native library, and returned hidden states at the readout positions bit-identical to Python ORT (max abs 0) and probabilities within 2.07e-7 of Python ORT (the C# head sums in a different order), 0 flips. Provisional bar 1e-6: pass. This replaces the symbol-level check recorded under Route A with an executed one, on Linux x64 only.

### Step 7: fallbacks

Not needed: route A met every provisional criterion it was measured against. Routes A2, B and C were not run. The stop rule was not triggered (one repair on route A).

### Decision input

Route A (ONNX Runtime GenAI builder graph, run with plain ONNX Runtime from Python and C#) meets every provisional criterion measured, on CPU at fp32 for Kev-0.8B, on both development partitions. What remains outside this spike: binding tolerances (maintainer), states above 8k tokens, Windows and macOS native libraries, bf16 or quantized variants, and the owned Qwen BPE profile for the Kev repository's `tokenizer.json`. The M2 slot decision belongs to the maintainer; this record does not set Decision.

## GPU results (2026-10-05)

Run 2026-10-05 to 2026-10-08 on a workstation: Intel Core i7-5820K (6 cores, 12 threads), 94 GB RAM, Linux x86-64, NVIDIA driver 610.57.04 (CUDA 13.3 capable), NVIDIA GeForce RTX 4060 Ti 16 GB (Ada, sm_89; the machine's other GPU was not used). On 2026-10-05 the GPU and CPU were shared with two other agents' jobs (CPU load average 25 to 40); timings from that period are labelled "contended". Uncontended timings were rerun on 2026-10-08. Downloaded weights, merged checkpoint, ONNX files, environments and per-record outputs stay on that machine under `/home/mark/Projects/AI-models/kev/` and are not committed. Thresholds are the provisional ones from step 8, labelled provisional, and are applied to the fp32 paths only. fp16 and bf16 numbers are drift measurements with no bar.

Scripts: `model-pipeline/reference/kev/`. `kev_fixtures.py` and `kev_ort.py` gained `--device cuda` / `--provider cuda` (plus `placement`, `compare`, `--state-chunk`, `--random-states`, `--cuda-tf32`); `kev_layers.py` is new (per-layer diagnosis); `requirements-cuda.txt` records the three environments.

### Environment

- Same pins as the CPU spike: Kev code `6b719c3c3f367295f6ef336f4f751cf5ff970abc`, Kev-0.8B weights `9a45d25eb2ab761841196625383fa1dff0e56c1e`, base `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Every file was downloaded again at those SHAs. Adapter `9b908623...`, `head.pt` `f400bd12...`, `tokenizer.json` `06b95093...`, base `model.safetensors-00001-of-00001.safetensors` `c2b1e5a1...` and the other base files all match `docs/m0/hub-pins.md`.
- Python 3.12.14 (uv-managed). Three environments. Each starts from Kev's `uv.lock` (`uv export --frozen --no-dev --extra serve`), installed unchanged, and adds:
  - `env-cpu`: torch 2.8.0+cpu in place of the CUDA wheel (as in the CPU spike), onnxruntime 1.30.0, onnxruntime-genai 0.17.1, onnx 1.23.1, onnx-ir 1.0.0.
  - `env-cuda` (PyTorch CUDA reference): the lock's own torch 2.8.0+cu128, plus upstream's CUDA serving kernels installed the way its Modal image installs them. That means flash-linear-attention 0.5.2 (with fla-core 0.5.2 and einops 0.8.1), triton 3.7.1 (the lock has 3.4.0; upstream requires at least 3.7.1), and the causal-conv1d 1.7.0 prebuilt wheel (`cu12torch2.8cxx11abiTRUE-cp312`, installed with `--no-deps`). Kev's `kernel_environment` in `report.json` records the DeltaNet kernels as `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule` and `causal_conv1d.causal_conv1d_interface.causal_conv1d_fn`, and attention as `sdpa`.
  - `env-ortgpu`: onnxruntime-gpu 1.30.0 with its `[cuda,cudnn]` extras. It is a CUDA 13.0 build; the extras resolved to nvidia-cuda-runtime 13.4.92, nvidia-cublas 13.8.1.7 and nvidia-cudnn-cu13 9.27.0.42. Also onnxruntime-genai 0.17.1 (installed with `--no-deps`, so the CPU onnxruntime package is not pulled in), onnx 1.23.1 and onnx-ir 1.0.0.
- `head.pt` was loaded only with `weights_only=True` (`TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` in every Kev process). GPU processes ran with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.
- PyTorch CUDA uses the settings of `kev.predictors.LocalPredictor`: TF32 off in torch, and SDPA's flash and memory-efficient kernels off (rows over 16,384 tokens use `long_row_kernels`). These switches do not reach the fla Triton kernels, which upstream documents as using TF32 dot products, so one read was also run with `TRITON_F32_DEFAULT=ieee`.
- ONNX Runtime CUDA EP ran with `use_tf32=0` unless stated otherwise. ORT's default is `use_tf32=1`; that setting is measured separately below.

### Artefacts built again

| Artefact | SHA-256 | Note |
| --- | --- | --- |
| CPU export `-e cpu -p fp32`, `model.onnx` | `046aa7b4367dee6c29bd494f330e60c5d2f665540558e26ea02148ecfd7f6278` | 307,884 bytes. Same size as the spike's `55dc88d2...`, different bytes. |
| CPU export, `model.onnx.data` | `dd78b21c101ef945dc4b7ca31367647da70e4d7fbd3f3d2e05a3245c573dc51a` | 3,077,177,344 bytes. Same size as the spike's `b52e6562...`, different bytes. Cause not investigated; the spike's files were not kept. |
| CUDA export `-e cuda -p fp32`, `model.onnx` as built | `b87bde0525c53d5ac3e6d9914bce9b5317cd89eefadfd6c3251579ec8660a763` | Does not load (failure A below). Its weights file is byte-identical to the CPU export's (`dd78b21c...`). |
| CUDA fp32 graph after repairs | `model-patched2.onnx` `3601b7e4b29186afb581d1e16c2a00bc57543a2d85802e6e55963c8245089a16`, then `value_info` removed (`model-patched3.onnx`) | Used for every "CUDA fp32 graph" number below. |
| CUDA export `-e cuda -p fp16` | `model.onnx` `b5a8c359cb407bd3e27f30ad9a084ad7f67f591dff06c7e800781bf399b2b548`, data `29e7dc0fd2658c88ff5eff8f6f3d2cb01e4d421e224bba1bed57bfc55e0c4aa5` | Loads and runs unmodified. |
| CUDA export `-e cuda -p bf16` | `model.onnx` `be14494f73361a35a570c16f1b352d8a7b347d5dc37e83bc76a2f11aa9e79395`, data `79fd6f7b4259d0fa744fd8468ff781c9dbc0f70fb2e54d63991a91308c119635` | Loads and runs unmodified. |

Every builder run used `--extra_options exclude_lm_head=true exclude_mtp=true` (the CPU spike's repair). Graph structure:

- `-e cuda -p fp32`: the DeltaNet layers use the same `LinearAttention` op (`update_rule` gated_delta) as the CPU graph, not the CUDA-only `GatedDeltaNet` op. Attention uses `MultiHeadAttention` (6 nodes) with a mask subgraph generated by the builder and an explicit `repeat_kv`, in place of `GroupQueryAttention`.
- `-e cuda -p fp16` and `bf16`: 18 nodes each of `LinearAttention`, `CausalConvWithState`, `LinearAttentionGate` and `GatedRMSNorm`, 6 `GroupQueryAttention`, 12 `MRotaryEmbedding`. KV cache and hidden states are fp16 or bf16.
- None of the graphs from builder 0.17.1 used `GatedDeltaNet`.

### Builder failures on the CUDA fp32 export

Failure A (model load). ONNX Runtime refuses the `-e cuda -p fp32` graph: `Node input 'past_key_values.0.key' is not a graph input`. The mask subgraph (`/model/attn_mask_reformat/past_key_subgraph/Shape`) reads the past length from layer 0's key cache. In Qwen3.5, layer 0 is a DeltaNet layer; the first attention layer is layer 3. Repair 1 (the only attempt): point that `Shape` node at `past_key_values.3.key`, which has the same layout. The graph then loads and runs single rows.

Failure B (continuing from a cached state). With a non-empty KV cache and more than one new token, `/model/attn_mask_reformat/input_ids_subgraph/Expand` fails with `left operand cannot broadcast on dim 3 LeftShape: {1,1,693,693}, RightShape: {1,1,693,703}`. The causal triangle is built as [S, S] instead of [S, past + S]. The mask therefore supports prefill and one-token decode, but not Kev's path of a state followed by a branch.
- Repair 1: rebuild the triangle as [S, past + S], with row i allowing the columns below past + i + 1 (4 added nodes). The run then failed in `repeat_kv/Transpose_2` with buffer-reuse shape mismatches (`{1,703,8,256} != {1,693,8,256}`). These come from the intermediate shapes the builder declares.
- Repair 2: drop the graph's 1,480 intermediate `value_info` entries. After this the cached path runs, and its numbers are below.

That is two repairs on failure B; the stop rule was not triggered. The repaired graph is a hand-patched builder output and should not be adopted as is.

Minimal reproducer for both failures:
1. `python -m onnxruntime_genai.models.builder -i <merged Qwen3.5 checkpoint> -o out -p fp32 -e cuda --extra_options exclude_lm_head=true exclude_mtp=true`
2. `onnxruntime.InferenceSession("out/model.onnx", providers=["CUDAExecutionProvider"])` fails with failure A.
3. After patching A, run a 10-token state, then a 693-token input that continues from its cache. This fails with failure B.

The fp16 and bf16 CUDA exports, and the CPU export on either EP, do not use this mask subgraph and have neither failure.

### Goal 1: PyTorch CUDA reference (upstream `kev.benchmark`)

The CPU fp32 reference was generated again on the workstation (`env-cpu`, 4 threads per partition, contended). decision-v7 development: 0.8267405063291139 (n = 1,264). transfer-v4 development: 0.6478658536585366 (n = 656). Both equal the release record and the CPU spike.

Every row below compares per-question probabilities with that CPU fp32 reference: 1,468 questions on decision-v7 and 764 on transfer-v4. "Flips" are argmax changes; the reference's top-two margin is listed for each.

| Read (CUDA, RTX 4060 Ti) | decision-v7 max / mean / p99 abs | flips | accuracy (n = 1,264) | transfer-v4 max / mean / p99 abs | flips | accuracy (n = 656) |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 with fla 0.5.2 and causal-conv1d (upstream's CUDA evaluation path) | 2.68e-3 / 2.1e-4 / 1.5e-3 | 0 | 0.8267405063291139 | 4.36e-3 / 3.5e-4 / 1.5e-3 | 0 | 0.6478658536585366 |
| Same, with `TRITON_F32_DEFAULT=ieee` | 3.25e-3 / 1.3e-4 / 1.1e-3 | 0 | 0.8267405063291139 | 1.98e-3 / 2.2e-4 / 1.2e-3 | 0 | 0.6478658536585366 |
| bf16 serving path (`KEV_DTYPE=bf16 KEV_FUSED=1`: bf16 backbone, LoRA folded, fused fla kernels) | 5.15e-2 / 2.0e-3 / 1.3e-2 | 2 (margins 0.025, 0.027) | 0.8283227848101266 | 2.22e-2 / 3.1e-3 / 1.2e-2 | 3 (margins 0.0011, 0.0025, 0.0029) | 0.6478658536585366 |

- Both fp32 CUDA reads reproduce the release accuracies exactly, but they differ from the CPU fp32 read by up to 4.4e-3. That is far above the provisional 1e-4 bar. The bar was written for ORT against PyTorch on the same CPU, so this is context rather than a fail.
- The gap matches upstream's own note that fp32 Qwen3.5 reads on CUDA carry fla's TF32 dot products (about 0.003 in probability).
- `TRITON_F32_DEFAULT=ieee` did not close the gap here; upstream reports that it closed it to 4e-6 on Kev-4B. The per-layer run below shows the fla delta-rule output still about 1e-3 from a float64 recurrence with ieee set. Not investigated further.
- bf16 against fp32 on CUDA (same kernel set): decision-v7 max 0.049 with 2 flips, transfer-v4 max 0.018 with 3 flips. One decision-v7 question exceeds upstream's "up to about 0.03 on GPU" (0.049 against CUDA fp32, 0.052 against CPU).
- `kev.serve`'s CUDA graphs were not part of these reads. `kev.benchmark` scores the eager `forward` path; `cuda_graphs` applies only to `probs_batch`.

### Goal 2: ONNX Runtime CUDA EP

Node placement, from session profiling of one 703-token row (second run):

| Graph | Nodes on CUDA EP | Nodes on CPU EP |
| --- | ---: | --- |
| CUDA fp32 (repaired) | 756 | 188, all shape arithmetic for the mask subgraph and `repeat_kv`: Gather 53, Unsqueeze 53, Concat 27, Equal 14, Where 14, Reshape 12, Mul 12, and one each of Add, Slice and Squeeze. Every compute op runs on CUDA. |
| CPU export on the CUDA EP | 649 | 8: six `GroupQueryAttention` nodes (no CUDA kernel for this fp32 configuration, so they run on CPU with copies), one Gather, one Cast |
| CUDA fp16 | 654 | 2: Gather and Cast (the token-id path) |
| CUDA bf16 | 654 | 2: Gather and Cast |

At session creation ORT also logs "Some nodes were not assigned to the preferred execution providers" and "1 Memcpy nodes are added"; the profile above lists those nodes. Session profiling therefore gives the fallback visibility plan 11 requires. ORT's own verbose node-assignment log needs a non-minimal build and was not used.

Fixture comparisons (spike step 5): 64 decision-v7 development records, 79 questions, against the CPU PyTorch fixtures.

| Check | ORT CPU EP, 12 threads | ORT CUDA EP, CUDA fp32 graph | ORT CUDA EP, CPU graph | Provisional bar | Outcome (fp32) |
| --- | --- | --- | --- | --- | --- |
| 1. Single rows: probability max (mean, p99) | 4.07e-5 (4.7e-6, 3.4e-5) | 4.03e-5 (4.7e-6, 3.3e-5) | 4.05e-5 (4.7e-6, 3.3e-5) | 1e-4, no flips | pass, 0 flips |
| 1. Hidden states at readouts, max | 2.21e-4 | 2.16e-4 | 2.12e-4 | | |
| 2. Cached-state rows against ORT single rows | 1.79e-7 | 1.73e-6 | 1.49e-6 | 1e-5 | pass |
| 2b. Cached-state rows against PyTorch `probs()` layout | 4.14e-5 | 4.07e-5 | 4.11e-5 | (as 1) | pass |
| 3. Padded batches against each row alone (25 questions) | 0 | 5.96e-7 | 1.25e-6 | 1e-5 | pass |
| 6. NaN / Inf | none | none | none | none | pass |

With ORT's default `use_tf32=1`, the same CUDA runs give:

| Check (TF32 on) | CUDA fp32 graph | CPU graph on CUDA EP |
| --- | --- | --- |
| Single rows, probability max | 8.2e-4 | 7.1e-4 |
| Cached rows against single rows | 7.4e-4 | 9.6e-4 |
| Padded batches | 3.7e-4 | 4.7e-4 |
| Hidden states at readouts | 8.6e-3 | 9.4e-3 |

That fails every provisional fp32 bar. fp32 parity on Ampere or later GPUs needs TF32 off (`use_tf32=0`). The fp16 and bf16 graphs gave identical numbers with TF32 on and off.

Full development partitions, forward layout, against the CPU fp32 PyTorch reference:

| Path | decision-v7 max / mean / p99 | flips | accuracy | transfer-v4 max / mean / p99 | flips | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| ORT CPU EP, 12 threads | 8.33e-5 / 4.3e-6 / 3.3e-5 | 0 | 0.8267405063291139 | 5.74e-5 / 6.6e-6 / 2.4e-5 | 0 | 0.6478658536585366 |
| ORT CUDA EP, CUDA fp32 graph | 8.05e-5 / 4.2e-6 / 3.1e-5 | 0 | 0.8267405063291139 | 5.67e-5 / 6.5e-6 / 2.3e-5 | 0 | 0.6478658536585366 |
| ORT CUDA EP, CPU graph | 8.08e-5 / 4.3e-6 / 3.1e-5 | 0 | 0.8267405063291139 | 5.69e-5 / 6.6e-6 / 2.3e-5 | 0 | 0.6478658536585366 |
| ORT CUDA EP, fp16 graph | 6.39e-3 / 3.2e-4 / 2.2e-3 | 0 | 0.8267405063291139 | 3.16e-3 / 4.8e-4 / 2.1e-3 | 0 | 0.6478658536585366 |
| ORT CUDA EP, bf16 graph | 9.86e-2 / 3.0e-3 / 2.1e-2 | 5 (margins 0.024, 0.0010, 0.0048, 0.0009, 0.012) | 0.8267405063291139 | 3.82e-2 / 4.4e-3 / 1.9e-2 | 4 (margins 0.0011, 0.026, 0.0025, 0.011) | 0.6448170731707317 |

- fp32 ORT on CUDA meets the provisional bars against the CPU PyTorch reference: max 8.1e-5, 0 flips, identical accuracy.
- ORT CUDA against ORT CPU: max 9.8e-6 on decision-v7 and 2.5e-6 on transfer-v4. The two graphs on the CUDA EP agree with each other to 3.3e-6.
- Against the PyTorch CUDA fp32 read, the same ORT CUDA results differ by up to 2.7e-3 (decision-v7) and 4.4e-3 (transfer-v4). The ORT paths agree with PyTorch on CPU, not with PyTorch on CUDA with fla. A binding fp32 reference should therefore name its device and kernel set; CPU PyTorch is the one ORT matches.
- fp16 drift against fp32: max 6.4e-3, 0 flips.
- bf16 drift against fp32: max 0.099, 9 flips over both partitions, transfer-v4 accuracy down 0.003. That is about twice PyTorch's own bf16 serving drift (0.052).

### Goal 3: long states

| Run | State tokens | Result | Peak memory | Time |
| --- | ---: | --- | --- | --- |
| PyTorch CPU fp32, state run once (`probs` layout) | 32,768 | completed, finite | 76.5 GB RSS. Eager attention materializes 8 × 32,768² × 4 bytes = 34 GB per score tensor. The run failed under a 60 GB address-space cap and completed under an 85 GB cap. | state 746 s (12 threads, uncontended) |
| PyTorch CUDA fp32, both layouts | 32,768 | completed, finite. The two layouts differ by 2.6e-5; against PyTorch CPU, 1.1e-4. | 8.06 GB allocated, 8.92 GB reserved | state 66 s, forward layout 83 s |
| ORT CPU EP, state in one pass | 32,768 | completed, finite. Against PyTorch CUDA: up to 1.3e-4. | 14.0 GB RSS | state 1,058 s, 875 s per question row (contended, 6 threads) |
| ORT CPU EP, state in 2,048-token pieces | 32,768 | completed, finite. Against PyTorch CPU: 2.5e-6. | 5.7 GB RSS | state 298 s (12 threads, uncontended), branch 1.5 to 2.1 s |
| ORT CUDA EP, CUDA fp32 graph, one pass | 8,192 | completed, finite | 8.3 GB (process total from nvidia-smi) | 3.3 s |
| same | 12,288 | failed: the output of `/model/attn_mask_reformat/Tile` exceeds 4 GiB (the fp32 mask subgraph materializes [1, 8, S, S] in fp32) | | |
| ORT CUDA EP, fp16 graph, one pass | 24,576 | completed, finite | 5.3 GB | 2.8 s |
| same | 28,672 | failed: `CausalConvWithState` CUDA kernel returns `invalid argument` (`causal_conv_with_state_impl.cu` line 556) | | |
| ORT CUDA EP, CUDA fp32 graph, state in 2,048-token pieces | 32,768 | completed, finite. Against PyTorch CPU: 1.0e-6. | 8.8 GB | state 13.2 s, branch 0.57 s |
| same, fp16 and bf16 graphs | 32,768 | completed, finite. Against PyTorch CUDA: 1.7e-4 (fp16) and 1.5e-3 (bf16). | 3.2 GB | state 7.7 s (fp16), 8.3 s (bf16) |

- 2k and 7.5k states (the spike's long fixtures, generated again): ORT CPU max 3.3e-5 and ORT CUDA fp32 max 1.7e-5, both with 0 flips and finite outputs.
  - ORT CUDA: 1.8 s for the 7.5k state, 0.45 s for a branch from it.
  - ORT CPU (contended): 140 s and 3.4 s.
  - PyTorch CPU fixture generation peaked at 9.5 GB RSS at 7.5k.
- 32k states work on both devices only when the state is fed in pieces through the carried cache. Kev's own long-record path already runs the state once and branches from it.
- Single-pass ceilings on CUDA:
  - Between 8,192 and 12,288 tokens for the fp32 graph, set by the mask subgraph.
  - Between 24,576 and 28,672 tokens for every graph, set by the `CausalConvWithState` launch limit.

  The lengths in between were not bisected. Neither failure was out-of-memory on the 16 GB card.
- The previous spike's 32k run was killed on a 15 GB machine. That fits the PyTorch CPU peak (76.5 GB) but not the ORT CPU peak (14.0 GB for one pass). Why the earlier ORT run was killed is not explained.

### Goal 4: C# smoke on CUDA

Setup: a throwaway console project outside the repository, at `/home/mark/Projects/AI-models/kev/csharp/KevSmoke` (`net10.0`, .NET SDK 10.0.100).
- Packages: `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 (with `Microsoft.ML.OnnxRuntime.Managed` 1.30.0). The CUDA 13 and cuDNN libraries come from the `env-ortgpu` wheels via `LD_LIBRARY_PATH`.
- Run: CUDA EP with `use_tf32=0`, the repaired CUDA fp32 graph, and the pointer head plus a minimal safetensors reader written in C#.

On the same 24 single-row fixture questions as the CPU spike:
- Hidden states at the readouts are bit-identical to Python ORT CUDA (max abs 0).
- Probabilities are within 2.98e-7, from the summation order in the C# head. 0 flips.
- Provisional bar 1e-6: pass.

The provider setup is the only CUDA-specific code:

```csharp
using var cuda = new OrtCUDAProviderOptions();
cuda.UpdateOptions(new Dictionary<string, string> { ["device_id"] = "0", ["use_tf32"] = "0" });
opts.AppendExecutionProvider_CUDA(cuda);
```

### Goal 5: timing (exploratory, not a benchmark claim)

Setup: warm, fp32, batch 1, one question per pass (forward layout: state and branch as one row). Input: the first 300 decision-v7 development records (460 questions, mean 207 tokens per row). Uncontended, 2026-10-08. Wall time includes host-to-device copies and, for PyTorch, Kev's encoding and head.

| Path | Questions timed | Mean per question | p50 | p90 | Questions / s |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch CUDA fp32 (`kev.benchmark`, fla) | 459 | 82 ms | | | 12.2 |
| ORT CUDA EP, CUDA fp32 graph | 459 | 48 ms | 32 ms | 115 ms | 20.7 |
| ORT CPU EP, 12 threads | 459 | 1.43 s | 0.80 s | 4.31 s | 0.70 |
| PyTorch CPU fp32 (`kev.benchmark`, first 100 records only, 99 questions) | 99 | 4.43 s | | | 0.23 |

Contended full decision-v7 passes, for scale only:

| Path | Time |
| --- | ---: |
| ORT CUDA, CUDA fp32 graph | 91 s |
| ORT CUDA, CPU graph on CUDA EP (`GroupQueryAttention` on CPU) | 171 s |
| ORT CUDA, fp16 graph | 72 s |
| ORT CUDA, bf16 graph | 73 s |
| ORT CPU, 12 threads | 3,279 s |
| PyTorch CUDA fp32 (scoring time) | 300 s |
| PyTorch CPU | 9,299 s |

### ORT CPU EP rerun and the error gap (maintainer request)

ORT CPU EP, fp32, against the CPU PyTorch reference generated on this workstation:

| Run | Questions | max abs | mean | p99 | flips | accuracy |
| --- | ---: | --- | --- | --- | ---: | --- |
| decision-v7 development, 12 threads | 1,468 | 8.33e-5 | 4.3e-6 | 3.3e-5 | 0 | equal (0.8267405063291139) |
| transfer-v4 development, 12 threads | 764 | 5.74e-5 | 6.6e-6 | 2.4e-5 | 0 | equal (0.6478658536585366) |
| transfer-v4 development, first 160 questions, 1 thread | 160 | 5.74e-5 | 8.2e-6 | 2.6e-5 | 0 | equal |
| 1 thread against 12 threads, same 160 questions | 160 | 1.49e-7 | 4.2e-8 | 1.3e-7 | 0 | |

The cloud spike's 8.43e-5 and 5.83e-5 reproduce within 1e-6 on different hardware and thread counts. The thread count moves results by at most 1.5e-7. So the gap to upstream's ExecuTorch fp32 result (1.06e-5) is not a threading or hardware effect.

Localization method (`kev_layers.py`): two rows were analysed, the worst decision-v7 question (`mnli/validation_matched/7311`, question `relation`, 8.3e-5) and fixture `r0000` q0.
- ORT CPU's intermediate tensors were exported from the graph.
- Each PyTorch fp32 CPU layer was then run on ORT's own input to that layer. Each number below is therefore the error that one ORT layer adds by itself.
- The DeltaNet delta-rule output was also compared with a float64 token-by-token recurrence, computed from PyTorch's delta-rule inputs.

| Component (worst question) | Local max abs against PyTorch fp32 | Reference magnitude |
| --- | --- | --- |
| Full-attention mixers (GQA, MRoPE, output gate), 6 layers | 2.7e-7 to 1.9e-6 | 0.6 to 3.3 |
| MLPs, layers 0 to 22 | 1.6e-7 to 1.9e-6 | 0.3 to 3.4 |
| MLP, layer 23 | 7.6e-5 | 57 (1.3e-6 relative) |
| DeltaNet mixers, 18 layers | 7.7e-7 to 4.1e-5 | 0.3 to 3.2 |
| DeltaNet `LinearAttention` output, ORT against float64 | up to 5.5e-5 (layer 2); above 1e-6 in 11 of 18 layers (0, 1, 2, 4, 6, 8, 10, 12, 13, 14, 22) | 0.01 to 0.44 |
| DeltaNet delta rule, PyTorch fp32 (chunked) against float64 | at most 2.3e-7 | |
| Final norm | 1.6e-5 | 27 |

- In layers 0 to 14 and 22, the error of the ORT CPU `LinearAttention` (`gated_delta`) kernel against exact arithmetic is 8 to 1,000 times that of PyTorch's chunked fp32 rule. In layers 16 to 21 the two are similar. The same layers stand out on both analysed rows.
- Every other op family adds about fp32 rounding.
- The accumulated hidden-state difference at the readouts grows from 6e-6 after layer 1 to 2.5e-4 after layer 24, which gives the 8.3e-5 in probability.
- So the delta-rule kernel is the main source of the 8x gap to ExecuTorch, whose lowering calls a fused gated delta rule op. That the kernel's accumulation order is the cause is an inference; its source was not read.
- On the CUDA EP with ORT's default `use_tf32=1` (worst question): every MatMul adds about 1e-4 locally (1.1e-2 in MLP layer 23), and the readout difference reaches 3.8e-3. This explains the TF32-on fixture numbers above. With `use_tf32=0`, ORT CUDA tracks ORT CPU to 1e-5.
- PyTorch CUDA with fla and `TRITON_F32_DEFAULT=ieee`, on the same ORT inputs (fixture `r0000`): its delta-rule output is up to 2.6e-3 from float64 (layer 18), and one layer's CPU-to-CUDA spread in PyTorch reaches 3.3e-3. Of the three fp32 implementations measured, this CUDA PyTorch read is the least exact.
- No fix was attempted (diagnosis only).

### What was not done

- The CUDA-only `GatedDeltaNet` op: no export from builder 0.17.1 used it, so it was not exercised.
- ORT CUDA graphs (`enable_cuda_graph`), IO binding, batched timing, `kev.serve` with CUDA graphs, and latency under concurrency.
- Bisecting the single-pass CUDA ceilings (8,192 to 12,288 tokens for the fp32 graph, 24,576 to 28,672 for the convolution), and states above 32,768 tokens.
- A fix or upstream report for builder failures A and B, the `CausalConvWithState` launch limit, or the accuracy of the CPU `LinearAttention` kernel.
- Explaining why `TRITON_F32_DEFAULT=ieee` did not bring fla to fp32 accuracy, and why the CPU export built again has different bytes from the spike's.
- Windows and macOS, and the C# smoke with the CPU graph on the CUDA EP.

### Decision input (GPU)

- fp32 parity on CUDA is reachable with route A, provided `use_tf32=0`. ORT CUDA EP, with either the CPU graph or the repaired CUDA fp32 graph, matches CPU PyTorch within the provisional bars on both partitions. Cached-state rows, padded batches and the C# smoke are all within their bars too.
- The `-e cuda -p fp32` graph as built does not load (failure A) and cannot continue from a cached state (failure B). The CPU export on the CUDA EP has neither problem, but runs `GroupQueryAttention` on CPU and was about 2x slower in this test.
- The fp32 reference must name its kernel set. PyTorch CUDA with upstream's fla kernels differs from PyTorch CPU and from ORT by up to 4.4e-3, with identical accuracy.
- bf16 is not a parity target: drift up to 0.05 on the PyTorch serving path and 0.10 with the ORT bf16 graph, with flips. ORT fp16 stays within 6.4e-3 with no flips on these partitions.
- States up to 32,768 tokens work on the 16 GB GPU and on CPU when the state is fed in pieces. The single-pass limits come from kernels and the graph, not from memory.
- Open for the maintainer:
  - Whether binding fp32 tolerances are set against CPU PyTorch as measured (max 8.3e-5), or whether the `LinearAttention` kernel must first improve toward 1e-5.
  - Whether the CUDA fp32 route uses the CPU export on the CUDA EP or waits for a builder fix.
  - Whether chunked state prefill becomes part of the contract above some length.
