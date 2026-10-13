# M0: Generation-backend check

Decides the kickoff item "Generation backend: ONNX Runtime GenAI or an existing llama.cpp binding" (`docs/kickoff.md`) for backlog P-016, with the inputs P-017 and P-029 need. Runs on 2026-10-05 and 2026-10-08. The kickoff decision itself belongs to the maintainer; section 9 is a recommendation.

## 1. Setup

### Environment

- Host: Intel Core i7-5820K (6 cores, 12 threads, AVX2, no AVX-512), 94 GB RAM, Linux x86-64 (kernel 7.2.5). GPU: NVIDIA RTX 4060 Ti 16 GB (PCI index 1, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`), driver 610.57.04, CUDA 13.3 toolkit at `/opt/cuda`, no system cuDNN. The GPU and CPU were shared with other jobs. On 2026-10-05 the load average was about 40 (12 hardware threads); the 2026-10-08 benchmarks ran at load 1 to 15, recorded per run.
- .NET SDK 10.0.100 (repository `global.json`), runtime 10.0.12.
- Python 3.12.14 with `model-pipeline/reference/generation/requirements.txt` (compiled with hashes from `requirements.in`): torch 2.14.1+cu130, transformers 5.18.0, tokenizers 0.23.2, onnxruntime-genai 0.17.1 (builder), onnxruntime 1.30.0, onnx 1.23.1, onnx-ir 1.0.0, numpy 2.5.3.
- GGUF conversion environment: llama.cpp `requirements/requirements-convert_hf_to_gguf.txt` at the conversion commit, resolved: torch 2.6.0+cpu, transformers 5.5.1, numpy 1.26.4, `gguf` from the same llama.cpp checkout.

### Model

`Qwen/Qwen3.5-0.8B` (the chat model) at `2fc06364715b967f1860aea9cf38778875588b17`, Apache-2.0. Every file is hashed in `docs/m0/hub-pins.md` (section "Pilot LLM"); weights `model.safetensors-00001-of-00001.safetensors` `04b1c301...4696`. The checkpoint is bf16; every fp32 run upcasts it. Architecture as recorded for the base in `model-pipeline/admissions/kev.md`: 24 layers, 18 Gated DeltaNet and 6 full attention, vocab 248,320, tied embeddings, one MTP layer that no run here uses.

### Prompt set and reference

- `model-pipeline/reference/generation/prompts-v1.jsonl` (SHA-256 `7b236f0d013ce1057f075f7d2d941dde836618ba0f848d6a123e0f86c4b928d2`), 22 prompts written for this check: English (3), system prompt, multi-turn, multilingual (Chinese, Japanese, Arabic, Hindi, Thai, Russian/German with diacritics, emoji sequences), code (Python, C#, SQL), four special-token look-alikes in user text (`<|im_end|>`, `<|endoftext|>`, `<think>`/`<tool_call>`/`<|fim_prefix|>`, broken forms), `enable_thinking=true`, a tool definition, and a long prompt (Apache-2.0 sections 1 to 6 from the snapshot's `LICENSE`, 1,666 tokens rendered). `max_new_tokens` 128.
- Reference: `run_reference.py`, transformers 5.18.0 `Qwen3_5ForCausalLM` in fp32, TF32 off, greedy (`do_sample=False`), stop ids 248046 `<|im_end|>` and 248044 `<|endoftext|>`, prompt rendered with the model's chat template (`apply_chat_template`, `add_generation_prompt=True`) and tokenized by the tokenizer transformers builds (`Qwen3_5Tokenizer`, see hub-pins). DeltaNet runs transformers' PyTorch fallback (no flash-linear-attention, no causal-conv1d).
- Reference self-agreement: CPU (8 threads) against CUDA: 22/22 identical token sequences; step-0 log-softmax max abs difference 1.6e-4 over the full vocabulary. Comparisons below use the CPU reference.

### Method

Each C# runner feeds the reference prompt ids directly (pre-tokenized input), decodes greedily in its own loop up to the stop ids or 128 tokens, and writes the shared result format (`compare.py` docstring): output ids, finish reason and the step-0 log-softmax. It also renders each prompt with the backend's own template engine and tokenizes it with the backend's tokenizer, and separately tokenizes the reference rendering with the backend tokenizer. `compare.py` reports exact sequence match, the first divergent step and the reference's top-1 minus top-2 logit margin at that step, and step-0 log-probability differences.

Runners: `model-pipeline/reference/generation/dotnet/OrtGenAiRunner` (Microsoft.ML.OnnxRuntimeGenAI 0.17.1, or `.Cuda` 0.17.1 with `-p:UseCuda=true`) and `dotnet/LlamaSharpRunner` (LLamaSharp 0.27.0 with `LLamaSharp.Backend.Cpu` and `.Cuda12` 0.27.0). Both are standalone `net10.0` console projects with central versions and lock files (`packages.lock.json`; `packages.cuda.lock.json` for the CUDA variant of the ORT runner).

## 2. Artifacts

### A. ONNX Runtime GenAI

Builder: `python -m onnxruntime_genai.models.builder -i <snapshot> -o <dir> -p <precision> -e <cpu|cuda> --extra_options exclude_mtp=true`, onnxruntime-genai 0.17.1 (the `rel-0.17.1` branch head `83de55a1f3886cddbb32a5517e26adb3f9d65159`; the repository has no `v0.17.1` tag). `exclude_mtp=true` drops the MTP head, which the generation loop here does not use.

Recipe finding: with the snapshot's own `config.json` (`text_config.dtype` bfloat16) the builder computes derived tensors in bf16 before writing them as fp32: every RMSNorm weight (`1 + w` offset: input, post-attention, q/k norm, DeltaNet gated norm, final norm; max change 1.6e-2) and the 18 `linear_attn.neg_exp_A` tensors (`-exp(A_log)`, `builders/qwen.py` line 635; max change 2.2e-2). Declaring float32 in a copy of `config.json` (weights hard-linked, nothing else changed) fixes this. Builds marked "x" below use that copy; without it fp32 matches only 16/22 sequences (section 3).

| Build | `model.onnx` SHA-256 | `model.onnx.data` SHA-256 (bytes) |
| --- | --- | --- |
| fp32x, `-e cpu` | `4a052c4a1444c9437e00ffd48baecb0838d3d9ca1703c3153dbd46e945e84703` | `129c58a320b6de7192713cc9c7e4ba744dcf7fabb0f57c6535021b45b3dc5143` (3,077,177,344) |
| fp16x, `-e cuda` | `26895023e4be1f528aef04ce3a07a14e70a2421cfb400a49fec639ea9a14f46d` | `47f6ee15d95080acc66f7c4acdf4012c83b20063cf0cfeb59a6b7ae09950a3b2` (1,572,143,104) |
| int4x, `-e cuda` (RTN, block 32) | `d6a369a5b9df03ee985ab32798a643f711a2a3c7f8338de6d89694cd3000d279` | `de8d146ea05c64b3765b9468998ab5ade8963f9a99e50840087fc9ff740e5c1c` (492,175,360) |
| int4x, `-e cpu` (RTN, block 32, accuracy level 4) | `766e49f3c58727e8cfd0865cfc0053a84998136f36395668e363c532193e7952` | `709e08797497f609547f7aef1387cfb35bd0deb3f4039acf001f0680b23f9745` (541,523,968) |
| fp32 (bf16 config), `-e cpu` | same graph as fp32x | `85b2d9585ba7b957dae65398c474573be00c231f601db102fd12152302535518` |

Also built: fp16, int4 (bf16 config, CPU and CUDA), int4 `algo_config=k_quant_mixed` (CUDA), fp32 `-e cuda` (fails, section 6).

The builder writes `tokenizer.json` (`a5cd9732...`, the transformers 5.18.0 runtime pipeline), `chat_template.jinja` (equal to the snapshot's) and `tokenizer_config.json` with `tokenizer_class` `Qwen3_5Tokenizer` (`004893c9...`). ONNX Runtime GenAI 0.17.1 rejects that class (section 6); every run here used a copy with the class renamed to `Qwen2Tokenizer` (`12d88f4c...`). `genai_config.json` sets `eos_token_id` [248046, 248044], `pad_token_id` 248044, greedy search.

### B. llama.cpp through LLamaSharp

- LLamaSharp 0.27.0 (NuGet, 2026-04-26) is the newest published version. Git tags v0.28.0 and v0.29.0 exist (v0.29.0 on 2026-08-24, llama.cpp `815a2a59`) but are not on NuGet. The LLamaSharp README maps 0.27.0 to llama.cpp `3f7c29d318e317b63f54c558bc69803963d7d88c` and lists "Qwen3.5 & Gemma4"; the v0.27.0 tag's submodule points at `73c9eb8c`. The native binaries carry no commit string; they contain `llm_build_qwen35` and `ggml_gated_delta_net`.
- All LLamaSharp runs used the unmodified NuGet natives (`libllama.so` `b9b54139...`, `libggml-cuda.so` `ae3fb875...`, hashes equal to the package contents). No native swap was needed.
- GGUF: `convert_hf_to_gguf.py` at `3f7c29d3` from the pinned safetensors (f32 and f16), then `llama-quantize` from a CPU build of the same commit:

| GGUF | SHA-256 | Bytes |
| --- | --- | ---: |
| f32 | `7bb360c54366f79d495f5e6d036b109b50abaaf76c82506488c2189f8614b436` | 3,020,533,696 |
| f16 | `4887b52521064ff2197eb64b0166929bbd8471ffb5085d83e807d76bd89bd186` | 1,516,744,640 |
| Q8_0 | `f3ebae5c29052d767286c9fc9aff4f7c292d323cb914630e956ca545426163a5` | 811,843,520 |
| Q4_K_M | `771f67fba8cad7b571e5d81840bf16b1021f625bfe6f340597d4bc51bb67720a` | 529,297,344 |

- GGUF metadata: `tokenizer.ggml.pre` `qwen35`, `add_bos` false, chat template embedded. llama.cpp reports BOS 11 (not added), EOS = EOT = 248046, pad 248044, end-of-generation set {248044, 248046, 248063, 248064, 248065} (`<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>` included).
- Published GGUF `ggml-org/Qwen3.5-0.8B-GGUF` at `8fea620810c4afa23dd6443f999a48574c1611a3` (`Qwen3.5-0.8B-Q8_0.gguf`, `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`, made from the same `2fc06364` weights per its `.src_sha`) does not load under 0.27.0's llama.cpp: `missing tensor 'blk.24.ssm_conv1d.weight'`. It carries the MTP layer as block 24, which a newer converter writes and `3f7c29d3` cannot read.
- `LLamaSharp.Backend.Cuda12.Linux` does not ship the CUDA runtime: `libggml-cuda.so` needs `libcudart.so.12` and `libcublas.so.12`. This host has CUDA 13 only; the runs added `nvidia-cuda-runtime-cu12` 12.9.79 and `nvidia-cublas-cu12` 12.9.1.4 (pip wheels) to `LD_LIBRARY_PATH`. They run on driver 610.

## 3. Fidelity against the transformers fp32 reference (22 prompts, greedy)

"Exact" counts prompts whose whole generated sequence (up to 128 tokens or stop) equals the reference. "Token match" is the matched prefix length summed over prompts divided by the reference length summed. Step-0 columns are the max abs log-probability difference over the full vocabulary and over the reference's top 5.

| Backend, artifact, device | Exact | Token match | Step-0 argmax | Step-0 max abs dlogp (all / top 5) | Divergences: step (reference margin) |
| --- | ---: | ---: | ---: | --- | --- |
| ORT GenAI fp32x, CPU | 22/22 | 1.000 | 22/22 | 2.46 / 0.19 (0.0014 / 0.0006 without `special-endoftext`) | none |
| ORT GenAI fp32x graph, CUDA EP, `use_tf32=0` | 22/22 | 1.000 | 22/22 | 2.46 / 0.19 (as above) | none |
| ORT GenAI fp32 (bf16 config), CUDA EP, `use_tf32=0` | 16/22 | 0.853 | 22/22 | 2.47 / 0.21; 0.05 to 0.18 on every prompt | 6, all at margins below 0.03 |
| ORT GenAI fp32 (bf16 config), CUDA EP, TF32 on (default) | 14/22 | 0.795 | 22/22 | 2.47 / 0.21 | 8, one at margin 1.02 (`ru-de-mix`) |
| ORT GenAI fp16x, CUDA | 18/22 | 0.887 | 22/22 | 0.076 / 0.034 | `ru-de-mix` 38 (0.007), `special-think-tool` 15 (0.003), `special-broken` 5 (0.014), `thinking-on` 76 (0.0004) |
| ORT GenAI fp16 (bf16 config), CUDA | 15/22 | 0.824 | 22/22 | 0.153 / 0.096 | 7, all at margins below 0.03 |
| ORT GenAI int4x RTN, CUDA | 1/22 | 0.056 | 13/22 | 8.99 / 6.35 | 21, from step 0 |
| ORT GenAI int4 k_quant_mixed, CUDA | 2/22 | 0.071 | 15/22 | 6.95 / 3.61 | 20 |
| ORT GenAI int4x RTN, CPU | 1/22 | 0.052 | 14/22 | 8.99 / 6.97 | 21 |
| LLamaSharp f32, CUDA | 22/22 | 1.000 | 22/22 | 0.023 / 0.009 | none |
| LLamaSharp f32, CPU | 21/22 | 0.993 | 22/22 | 0.011 / 0.004 | `code-sql` 112 (0.002) |
| LLamaSharp f16, CUDA | 19/22 | 0.932 | 22/22 | 0.272 / 0.156 | `ru-de-mix` 38 (0.007), `code-sql` 72 (0.017), `special-think-tool` 95 (0.003) |
| LLamaSharp f16, CPU | 20/22 | 0.968 | 22/22 | 0.027 / 0.017 | `code-sql` 112 (0.002), `thinking-on` 76 (0.0004) |
| LLamaSharp Q8_0, CUDA | 8/22 | 0.505 | 22/22 | 1.66 / 1.49 | 14, margins 0.0004 to 0.26 |
| LLamaSharp Q8_0, CPU | 9/22 | 0.469 | 22/22 | 1.57 / 1.46 | 13, margins 0.04 to 0.34 |
| LLamaSharp Q4_K_M, CUDA | 2/22 | 0.086 | 17/22 | 7.80 / 3.11 | 20 |
| LLamaSharp Q4_K_M, CPU | 2/22 | 0.075 | 17/22 | 7.15 / 2.80 | 20 |

Notes:

- The ORT GenAI step-0 maximum of 2.46 comes from one prompt, `special-endoftext`, whose prompt contains token 248044 (`<|endoftext|>`, typed by the user and kept as a special token by the reference tokenizer). 248044 is also `pad_token_id` in `genai_config.json`, and GenAI masks pad positions in the input: with `pad_token_id` overlaid to 248063 the difference on that prompt drops to 0.0013 and the sequence still matches. In that overlay run one prompt (`ar`) repeated a token at step 99 where the reference margin is 2.35; this was seen once and not investigated or rerun (untested whether reproducible).
- Without `special-endoftext`, ORT GenAI fp32x is within 1.4e-3 of the reference at step 0 on every prompt (total variation at most 1.1e-4), close to the reference's own CPU/CUDA spread (1.6e-4). LLamaSharp f32 is within 2.3e-2 (CUDA) and 1.1e-2 (CPU).
- Every fp16 and fp32 divergence happens at a reference margin below 0.03 logits (near ties), except `ru-de-mix` with TF32 on. The ORT CUDA EP enables TF32 by default (`use_tf32`); fp32 parity needs `use_tf32=0`.
- Both 4-bit routes change the step-0 argmax on 5 to 9 of 22 prompts. Neither backend's default 4-bit recipe is close to the reference on this model; Q8_0 keeps the argmax at step 0 but diverges within tens of tokens. Task-level quality of quantized variants was not measured.

## 4. Tokenizer and template boundary (plan 6.4, P-029)

| Question | ORT GenAI 0.17.1 | LLamaSharp 0.27.0 |
| --- | --- | --- |
| Accepts pre-tokenized ids from our tokenizer | Yes: `Generator.AppendTokens(ReadOnlySpan<int>)`, ids of any length, appended incrementally. `Model` loads without the tokenizer files; the GenAI `Tokenizer` is only needed for its own encode/decode/template. | Yes: `LLamaBatch` plus `llama_decode` (LLamaSharp `LLamaContext`/`BatchedExecutor`); the high-level executors take text. |
| Returns ids for our own decoding | Yes: `GetNextTokens()`, `GetSequence()` | Yes (sampled id per step; the runner samples argmax from raw logits) |
| Backend tokenizer on the reference rendering | 22/22 identical ids (ort-extensions, builder's `tokenizer.json`), after the class rename | 22/22 identical ids (GGUF vocabulary, special tokens parsed) |
| Backend template rendering equals reference | 21/22. Jinja rendering of the shipped template, including tools and `reasoning_content`. `ApplyChatTemplate(template, messages, tools, add_generation_prompt)` has no way to pass `enable_thinking`, so `thinking-on` renders the non-thinking suffix (a custom template string might work around it; untested). | 0/22. `LLamaTemplate` calls `llama_chat_apply_template`, which matches the template to a built-in ChatML formatter instead of running Jinja: it drops the `<think>\n\n</think>\n\n` suffix the template appends after `<|im_start|>assistant\n`, ignores tools (renders no system tools block) and `enable_thinking`. llama.cpp's Jinja engine lives in `libcommon`, which LLamaSharp does not ship or bind. |
| Stop ids | `eos_token_id` [248046, 248044] from `genai_config.json`; `IsDone()` after either. | EOG set {248044, 248046, 248063, 248064, 248065}; EOS/EOT 248046. |
| BOS | none added (`bos_token_id` 1 in config is unused by the loop) | BOS 11 reported, `add_bos` false, not added |
| Pad handling | Positions equal to `pad_token_id` (248044) are masked in prompts (section 3). | not applicable |

Both backends meet the 6.4 requirement that the direct path use the owned tokenizer: neither needs its own tokenizer when ids are fed. Neither supplies a template engine that can stand in for P-029 as a declared dependency without caveats: ORT GenAI's renders the pinned template exactly except for template kwargs; LLamaSharp's does not render this model's template.

Under the reference runtime (transformers 5.18.0) the effective pipeline has the `\p{M}` pre-tokenizer regex; the Kev lane's pinned runtime (transformers 5.17.0) loads the base without it (hub-pins). The prompt set has Hindi and Thai with combining marks, and both backends agreed with the 5.18.0 ids there.

## 5. Semantics

| Capability | ORT GenAI 0.17.1 (C#) | LLamaSharp 0.27.0 (C#) |
| --- | --- | --- |
| Streaming | `TokenizerStream.Decode` per token reconstructs `Tokenizer.Decode` of the full output: 22/22. | `StreamingTokenDecoder` equals detokenizing the full output (6 multilingual/emoji/code prompts); `StatelessExecutor` streaming (template off, greedy) reproduced the reference text for 64 tokens. |
| Cancellation | The loop is caller-driven (`GenerateNextToken` per step), so cancellation is checked between steps: 5.8 ms (CPU fp32x) and 153 ms (CUDA, loaded host, 2026-10-05) from `Cancel()` to exit after 10 tokens. No cancellation inside a step (a long prefill cannot be interrupted). The bundled `OnnxRuntimeGenAIChatClient` throws `OperationCanceledException` about 1 ms after `Cancel()`. | `InferAsync` with a token: `TaskCanceledException` 1.4 ms (CPU) and 21 ms (CUDA, loaded host) after cancel during decode. Cancel requested 20 ms into the 1,666-token prefill was not honoured: the call ran the whole prefill and completed with no pieces (348 ms CUDA, 18.3 s CPU). |
| Request state, concurrent requests | One `Generator` per request. Two generators interleaved token by token, and four on two threads sharing one `Model`, each matched the reference prefix (no state leakage observed). | Two sequences in one batch each matched the reference; `llama_memory_seq_cp` prefix reuse matched a fresh run. |
| KV/state reuse for a conversation | Appending the next turn's ids to the same generator equals a fresh generator on the full prompt (16 tokens). `RewindTo(n)` with 0 < n fails: "RecurrentState::RewindTo(26) is not supported without a snapshot"; `RewindTo(0)` works. The native library exports `OgaGenerator_SnapshotState`, but the C# wrapper does not expose it (untested). | `llama_memory_seq_rm` partial removal returns false on the hybrid memory (whole-sequence removal works; `can_shift` false). Sequence state save/load (20.9 MB at 40 tokens) and continuing from it equals a fresh run. |
| Microsoft.Extensions.AI | `OnnxRuntimeGenAIChatClient` (`IChatClient`) ships in the managed package (Microsoft.Extensions.AI.Abstractions 9.8.0). Defect: when generation ends at EOS the last streamed update repeats the previous piece ("Paris.." for "Paris."); output token count includes the stop token. Inferred cause: it decodes the last element of `GetSequence(0)` after the stop, which is still the previous token. | LLamaSharp 0.27.0 depends on Microsoft.Extensions.AI.Abstractions 10.4.1 (adapter not exercised here). |

## 6. Failures and repairs

1. ORT GenAI `Tokenizer(model)`: `OnnxRuntimeGenAIException: Unsupported tokenizer class: Qwen3_5Tokenizer`. Reproducer: build any precision with the 0.17.1 builder under transformers 5.18.0, then `new Tokenizer(new Model(dir))`. Repair 1 (rename the class to `Qwen2Tokenizer` in `tokenizer_config.json`) worked; ids then match the reference on 22/22.
2. ORT GenAI builder, `-p fp32 -e cuda`: `Load model ... failed: Invalid model. Node input 'past_key_values.0.key' is not a graph input` (the attention-mask reformat subgraph reads layer 0, a DeltaNet layer). Repair 1: rewire that `Shape` node to `past_key_values.3.key`; the model then loads (with cuDNN 9.24 from the `nvidia-cudnn-cu13` wheel on `LD_LIBRARY_PATH`, needed by `MultiHeadAttention`) but the first decode step fails in `/model/layers.7/attn/k_proj/repeat_kv/Transpose_2`: "Shape mismatch attempting to re-use buffer. {1,1,8,256} != {1,25,8,256}". Repair 2: `past_present_share_buffer=false`; same error. Stopped (two-repair rule). fp32 on CUDA was measured instead with the CPU-targeted fp32x graph on the CUDA EP (section 3), which is slow (section 8; node placement not inspected, so CPU fallback of the DeltaNet ops is possible).
3. Published GGUF `ggml-org/Qwen3.5-0.8B-GGUF@8fea6208` Q8_0 does not load in LLamaSharp 0.27.0 (`missing tensor 'blk.24.ssm_conv1d.weight'`; reproducer: `llama-bench -m Qwen3.5-0.8B-Q8_0.gguf` built at `3f7c29d3`). Not repaired; GGUFs were converted at the binding's own llama.cpp commit instead.
4. `python -m onnxruntime_genai.models.builder --help` crashes in argparse (`ValueError: unsupported format character ' '`, a `%` in the help text). Cosmetic; options were read from `builder.py`.
5. Builder bf16 derived tensors (section 2) and GenAI pad masking (section 3): not failures of a run, but both silently change outputs; each has a configuration workaround.

## 7. Packages, platforms, license, cadence

| | ONNX Runtime GenAI | LLamaSharp |
| --- | --- | --- |
| Packages, version | `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 (CPU; depends on `Microsoft.ML.OnnxRuntime` 1.30.0), `.Cuda` 0.17.1 (depends on `Microsoft.ML.OnnxRuntime.Gpu` 1.30.0), `.Managed` 0.17.1 (net8.0, netstandard2.0, net9.0 mobile) | `LLamaSharp` 0.27.0 (managed) plus `LLamaSharp.Backend.Cpu`, `.Cuda12` (meta package for `.Cuda12.Linux` and `.Cuda12.Windows`), `.Vulkan` 0.27.0 |
| CPU natives | linux-x64, linux-arm64, win-x64, win-arm64, osx-arm64, android, ios | linux-x64 and linux-musl-x64 (noavx, avx, avx2, avx512), linux-arm64, win-x64 (cpu variants), win-arm64, osx-arm64 (Metal), osx-x64 |
| CUDA natives | linux-x64, win-x64 (and win-arm64 GenAI files), CUDA 13 build (`libcublas.so.13`, `libcudart.so.13`); CUDA runtime, cuBLAS and cuDNN not bundled | linux-x64, win-x64, CUDA 12 build; CUDA runtime and cuBLAS not bundled |
| Package size | GenAI 108 MB (all RIDs) + ORT 150 MB; GenAI.Cuda 30 MB + ORT.Gpu.Linux 226 MB | Backend.Cpu 36 MB; Cuda12.Linux 226 MB (`libggml-cuda.so` 530 MB unpacked) |
| Exercised here | linux-x64 CPU and CUDA | linux-x64 CPU (avx2 variant) and CUDA |
| License | MIT (both) | MIT (binding); llama.cpp MIT |
| Release cadence (NuGet) | 14 versions from 0.12.0 (2026-02-13) to 0.17.1 (2026-09-29): a minor about monthly, patches within days | 0.24.0 2025-05-14, 0.25.0 2025-08-16, 0.26.0 2026-02-15, 0.27.0 2026-04-26; 0.28 and 0.29 tagged in Git but not published as of 2026-10-08 |
| Native/model coupling | GenAI version must match the builder version that wrote the graph and `genai_config.json`; graph ops are ORT contrib ops (`LinearAttention`, `CausalConvWithState`, `GroupQueryAttention`) | GGUF must be converted by a llama.cpp compatible with the binding's pinned commit; newer converters can write files the binding cannot read (failure 3) |

## 8. Exploratory warm throughput (C#)

Exploratory: shared machine, load average recorded per run, 1 warm-up then n = 5 samples (LLamaSharp: n = 5 on 2026-10-08), medians with [min to max]. Prefill on the 1,666-token `long-license` prompt; decode is 128 tokens forced (EOS suppressed by `min_length` in ORT GenAI; LLamaSharp decodes past EOS in its own loop). TTFT is prefill plus the first generated token. CPU runs use 6 threads. Not comparable to the plan's matched-native targets except where a native row is given.

| Backend, artifact, device | Load at start | Prefill tok/s, 1,666 tokens | TTFT ms, 1,666 tokens | Decode tok/s | Short prompt: TTFT ms, decode tok/s |
| --- | ---: | --- | --- | --- | --- |
| ORT GenAI fp16x, CUDA | 3.1 | 12,573 [12,493 to 12,615] | 132.6 [132.2 to 133.5] | 114.7 [91.4 to 121.9] | 22 tokens: 10.6 [8.7 to 12.9], 95.7 [87.6 to 127.9] |
| ORT GenAI int4x, CUDA | 4.6 | 12,094 [12,079 to 12,099] | 137.9 [137.8 to 138.1] | 153.7 [143.6 to 158.6] | 22 tokens: 11.5 [9.9 to 11.9], 202.1 [143.9 to 262.6] |
| ORT GenAI fp32x graph, CUDA EP, `use_tf32=0` | 6.4 | 1,802 [1,256 to 1,980] | 924.6 [841.6 to 1,326.7] | 22.1 [12.9 to 23.8] | 22 tokens: 19.7 [18.9 to 25.5], 55.9 [40.4 to 57.4] |
| LLamaSharp f32, CUDA | 7.2 | 10,912 [10,838 to 10,920] | 152.8 [152.7 to 153.9] | 77.0 [76.8 to 77.1] | 24 tokens: 16.2 [16.2 to 16.3], 76.8 [74.4 to 77.8] |
| LLamaSharp f16, CUDA | 8.9 | 15,388 [14,917 to 15,463] | 108.4 [107.9 to 111.9] | 129.2 [128.1 to 130.0] | 24 tokens: 13.6 [10.7 to 16.9], 129.6 [128.0 to 134.1] |
| LLamaSharp Q8_0, CUDA | 8.2 | 15,629 [15,465 to 15,638] | 106.8 [106.7 to 107.9] | 192.5 [191.5 to 193.9] | 24 tokens: 9.1 [8.9 to 10.3], 198.3 [196.4 to 201.3] |
| LLamaSharp Q4_K_M, CUDA | 7.7 | 14,571 [14,391 to 15,284] | 114.7 [109.2 to 116.1] | 198.7 [197.7 to 237.2] | 24 tokens: 9.9 [7.8 to 10.0], 237.5 [220.3 to 253.1] |
| ORT GenAI fp32x, CPU | 11.5 | 106 [100 to 123] | 15,782 [13,494 to 16,583] | 5.9 [4.4 to 6.6] | 22 tokens: 307 [221 to 833], 7.2 [5.3 to 7.7] |
| ORT GenAI int4x, CPU | 14.0 | 110 [86 to 122] | 15,101 [13,620 to 19,390] | 28.2 [21.1 to 30.1] | 22 tokens: 220 [210 to 251], 30.1 [27.9 to 33.0] |
| LLamaSharp f32, CPU | 5.4 | 158 [158 to 158] | 10,547 [10,521 to 10,570] | 10.7 [10.7 to 10.7] | 24 tokens: 184 [183 to 185], 10.9 [10.9 to 10.9] |
| LLamaSharp f16, CPU | 5.8 | 175 [174 to 175] | 9,537 [9,517 to 9,553] | 19.6 [19.6 to 19.9] | 24 tokens: 154 [154 to 160], 20.7 [20.7 to 20.7] |
| LLamaSharp Q8_0, CPU | 5.8 | 192 [82 to 196] | 8,694 [8,497 to 20,197] | 30.1 [2.0 to 30.3] | 24 tokens: 125 [125 to 128], 32.2 [32.2 to 32.2] |
| LLamaSharp Q4_K_M, CPU | 13.7 (disturbed) | 56 [54 to 154] | 29,730 [10,806 to 30,715] | 1.0 [0.6 to 26.3] | 24 tokens: 1,699 [472 to 1,907], 4.1 [3.5 to 4.8] |

Native baseline: `llama-bench` built at `3f7c29d3` (CPU, `GGML_NATIVE=ON`, so not the NuGet binary), 6 threads, 5 repetitions, 2026-10-08: pp1666 / tg128 tok/s f32 131.4 / 7.9, f16 121.0 / 14.4, Q8_0 195.2 / 32.0, Q4_K_M 217.0 / 41.0. The C# CPU numbers are not below these, but build flags and machine load differ, so this is not a measured binding overhead. No native CUDA baseline was run, and no Python ONNX Runtime GenAI baseline; the plan's 95% decode target is untested for both backends.

Observations: on CUDA, LLamaSharp f16 decodes about 13% faster than ORT GenAI fp16x (129 against 115 tok/s) and prefills about 22% faster; at 4 bits LLamaSharp Q4_K_M and ORT int4 are 199 and 154 tok/s. On CPU, ORT GenAI fp32x (5.9 tok/s decode, 106 tok/s prefill) is slower than LLamaSharp f32 (10.7 and 158), and ORT ran at higher machine load (11.5 against 5.4), so part of the gap may be load. The fp32x graph on the CUDA EP is slow (22 tok/s); it was not built for CUDA and node placement was not inspected. The Q4_K_M CPU row was disturbed by other jobs and should be rerun before use.

## 9. Recommendation

Recommendation for P-016: **ONNX Runtime GenAI**, `Microsoft.ML.OnnxRuntimeGenAI` / `.Cuda` 0.17.1 with the 0.17.1 builder, behind the adapter the plan requires (5.2), using the owned tokenizer and our own greedy/sampling loop over `Generator`. Pilot LLM: **`Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`**.

Reasons:

1. Fidelity: the fp32 artifact matches the transformers reference on 22/22 sequences on both CPU and CUDA, with step-0 log-probabilities within 1.4e-3 (apart from the pad-masked prompt), about ten times closer than LLamaSharp f32 (2.3e-2). fp16 fidelity is similar for both (18/22 against 19/22 on CUDA, all divergences at near ties).
2. Template: GenAI's template engine renders the pinned Jinja template exactly on 21/22 prompts, including tools, so the preview can declare it as the interim template dependency that plan 6.4 allows while P-029 is built. LLamaSharp's `llama_chat_apply_template` renders none of the 22 prompts correctly for this model (it drops the non-thinking suffix the model was trained to see), so a LLamaSharp route would need P-029 before any preview, or a binding to llama.cpp's Jinja code.
3. One native stack: the P0 encoder runs on direct ONNX Runtime C#, and the Kev route A graph comes from the same builder and runs on ONNX Runtime 1.30.0. GenAI 0.17.1 depends on that same ORT version, so the pilot LLM adds no second native inference library to the process (plan 5.2's incompatible-native concern).
4. Currency and packaging: GenAI publishes about monthly (0.17.1 on 2026-09-29) and builds CUDA 13; the newest LLamaSharp on NuGet is from April 2026, its later tags are unpublished, its CUDA build is CUDA 12, and its llama.cpp commit cannot read the published Qwen3.5 GGUF.

Conditions that come with the recommendation (each was needed to get the results above):

- Build from a copy of the checkpoint whose `config.json` declares float32 (section 2), with `exclude_mtp=true`. Record the recipe in the bundle.
- Do not use GenAI's `Tokenizer` for model semantics. If it is used for the interim template path, rename `tokenizer_class` to `Qwen2Tokenizer` (failure 1) and compare its rendering against the reference corpus.
- Overlay `pad_token_id` to an id that cannot appear in prompts (or have the adapter refuse prompts containing 248044), because GenAI masks pad positions.
- CUDA fp32 needs `use_tf32=0`; use fp16 for CUDA and fp32 for CPU. fp32 built for the CUDA EP is unsupported (failure 2).
- Implement request state as one `Generator` per request or conversation; partial rewind is unsupported on this hybrid model, so edits to earlier turns mean a new generator and a full prefill. Cancellation is between steps only.
- Do not use the bundled `OnnxRuntimeGenAIChatClient` for P-017 as is (duplicate final piece, stop token counted as output); write our own `IChatClient` over the adapter.

Costs of this choice, measured: CPU throughput is the weakest point (fp32 decode 5.9 tok/s against 10.7 for LLamaSharp f32 on this AVX2 host, under different loads), and CUDA decode is about 13% below LLamaSharp at fp16. 4-bit is not usable from either backend's default recipe (fidelity, section 3). If CPU decode speed outweighs template fidelity and the single native stack for the maintainer, LLamaSharp 0.27.0 with GGUF converted at `3f7c29d3` is the measured alternative; it would require P-029 first and a pinned CUDA 12 runtime.

Decisions for the maintainer: the kickoff rows "Generation backend" and "Pilot LLM ID and revision"; which runtime tokenizer pipeline the owned Qwen BPE profile targets (transformers 5.18.0 with `\p{M}`, used here, or Kev's 5.17.0 pipeline without it; hub-pins); whether a quantized variant is in scope for the preview (neither 4-bit recipe tried here keeps the step-0 argmax on all prompts); and binding tolerances for G2 (these results are observations, not tolerances).

## 10. Status of capabilities checked

| Capability (Qwen3.5-0.8B, linux-x64) | ORT GenAI 0.17.1 | LLamaSharp 0.27.0 |
| --- | --- | --- |
| Architecture loads and generates | supported (CPU fp32/int4, CUDA fp16/int4); fp32 built for CUDA unsupported | supported (CPU and CUDA, f32/f16/Q8_0/Q4_K_M) |
| Greedy parity at fp32 | experimental: 22/22 CPU and CUDA (`use_tf32=0`) | experimental: 22/22 CUDA, 21/22 CPU |
| Pre-tokenized ids in, ids out | supported | supported |
| Backend template rendering | experimental: 21/22, no template kwargs | unsupported for this template (0/22) |
| Backend tokenizer | experimental: 22/22 ids after class-name repair | experimental: 22/22 ids |
| Stop/EOS | supported (248046, 248044) | supported (EOG set of 5 ids) |
| Streaming text reconstruction | supported (`TokenizerStream`) | supported (`StreamingTokenDecoder`) |
| Cancellation | between steps only; not inside prefill | during decode; not inside prefill |
| Prefix reuse by appending | supported | supported (sequence copy, state save/load) |
| Partial rewind | unsupported (recurrent state; snapshot API not in C#) | unsupported (`seq_rm` partial returns false) |
| Concurrent requests, state isolation | experimental (2 interleaved, 4 parallel generators) | experimental (2 sequences in one batch) |
| Microsoft.Extensions.AI adapter | bundled `IChatClient` has a streaming defect | adapter present, untested |
| 4-bit fidelity | unsupported at default recipes (RTN, k_quant_mixed) | unsupported at Q4_K_M; Q8_0 diverges early |
| win-x64, osx-arm64, linux-arm64 | untested (natives present) | untested (natives present) |
| Native AOT, trimming | untested | untested |
| Long context beyond 1,666 tokens, batching beyond 2 | untested | untested |

## 11. Files

- `model-pipeline/reference/generation/`: `fetch_snapshot.py` (download by commit SHA and verify), `run_reference.py` (transformers reference), `compare.py` (scoring), `prompts-v1.jsonl`, `requirements.in` and `requirements.txt`, `dotnet/` (the two C# runners, shared `Common/Shared.cs`, standalone `Directory.Build.props` and `Directory.Packages.props`).
- Large files, not in Git, under `/home/mark/Projects/AI-models/gen/`: `hf/` (snapshot), `onnx/` (builder outputs), `gguf/`, `out/` (per-run `results.jsonl`, `step0/`, `environment.json`, `bench.json`, `semantics.json`, comparison JSON in `out/compare-vs-ref-cpu/`), `logs/`, `venvs/`, `build/` (llama.cpp `3f7c29d3` CPU and CUDA builds), `nupkg/`.
