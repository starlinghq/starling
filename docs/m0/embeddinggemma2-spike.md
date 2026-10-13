# EmbeddingGemma 2: admission spike (text embeddings)

Status: spike result, 2026-10-10. Brief: the spike plan in `model-pipeline/admissions/embeddinggemma-2.md` (plan section 7, priority P1). Template: the e5 path (`docs/m0/reference-runs.md` section 1, `docs/m0/e5-gpu.md`) and the GLiNER2 spike report (`docs/m0/gliner2-p014.md`). The upstream model card and repository files were read as data.

Pins: checkpoint `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b` (every file hashed in `docs/m0/hub-pins.md`, `model.safetensors` 1,488,915,288 bytes, SHA-256 `197a3296...`, `tokenizer.json` `4d777ef5...`); code transformers 5.19.0 (first release that defines `embedding_gemma2`), sentence-transformers 6.1.0, tokenizers 0.23.3, torch 2.14.1, ONNX Runtime 1.30.0, onnx 1.23.1, onnxscript 0.7.2. Environments: `model-pipeline/reference/embeddinggemma2/requirements.txt` (CPU) and `requirements-cuda.txt` (GPU), both hashed.

Contract under test: text embeddings only (mean pooling, L2 normalization, 768 dimensions, explicit query and document prompts). Image, audio and video towers are out of the contract.

## 1. Summary

| Question | Answer |
| --- | --- |
| Does the text path export to ONNX? | Yes, with no repair. The dynamo exporter (`torch.onnx.export(..., dynamo=True)`, opset 20) exports the text path (embedding, per-layer embeddings, 24 layers with sliding-window and full attention and per-layer KV-head counts, final norm, 512-to-768 projection) with dynamic batch and sequence. The TorchScript exporter (opset 18) also exports it with no repair. |
| ORT CPU parity (138 rows: 46 texts x 3 prompt settings, up to 1,750 tokens) | Final embedding max abs 4.2e-7, min cosine 0.99999999999473. Token embeddings (768, values up to 3,932 in magnitude) max abs 2.2e-2, at most 6.2e-6 relative to `max(1, |ref|)`. Identical at 1 and 12 threads. Padded batches vs single rows 1.8e-7. No NaN/Inf. |
| Batch independence across the sliding window | Synthetic lengths 511 to 2,049 tokens (512, 513, 1,024, 1,025 and neighbours) padded next to a 12-token row: ORT CPU batch equals single bit for bit; ORT vs PyTorch at most 2.1e-7; PyTorch padded vs single at most 2.1e-7. |
| ORT CUDA parity (GPU 1, `use_tf32=0`) | Embedding max abs 4.8e-7, min cosine 0.99999999999536; token embeddings 1.1e-5 relative; padded batches 3.0e-7. 2,423 nodes on CUDA, 130 shape nodes on CPU (57 `Concat`, 72 `Slice`, 1 `Squeeze`). With ORT's default `use_tf32=1`: embedding 1.9e-4, min cosine 0.9999986, batch vs single 3.1e-4. Pin `use_tf32=0`, as for e5 and GLiNER2. |
| C# (Microsoft.ML.OnnxRuntime 1.30.0) on fixture ids | CPU 1 and 12 threads: embedding max abs 3.7e-7 vs the reference, min cosine 0.99999999999473, 1 and 12 threads byte-identical. CUDA `use_tf32=0`: 3.6e-7. C# vs Python ORT 4.2e-7 (same provider), consistent with pooling arithmetic (C# double accumulation vs NumPy float32). |
| Tokenizer | `tokenizer.json` is SentencePiece-style BPE with byte fallback. transformers' live tokenizer equals the file field by field; sentence-transformers' chat-template preprocessing gives the same ids as `tokenizer.json` on `prompt + text` for every corpus text and prompt. The owned library (0.1.0-preview.3, Qwen BPE merged) still lacks byte fallback and an unknown token in the BPE model, a string-pattern `Split` with `MergedWithPrevious`, and the `Sequence`/`Replace`/`ByteFallback`/`Fuse` decoders (section 5). |
| Recommended route | One ONNX graph from ids and mask to 768-wide token embeddings, dynamo exporter; mean pooling, normalization, prompts and Matryoshka truncation in C#. Same shape as e5. |
| Estimate | 14 to 21 working days for a text-embedding preview with Extensions.AI and ML.NET adapters; the Gemma BPE profile is the long pole (section 7). |
| Recommendation | Admit at Experimental, text embeddings only (section 9). Decision left to the maintainer. |

## 2. Card and configuration facts

| Item | Value | Source |
| --- | --- | --- |
| License | `license: apache-2.0` in the card front matter; the card's license link (`ai.google.dev/gemma/docs/gemma_4_license`) presents the Apache License 2.0 text. No LICENSE file in the repository. Not gated. | card, linked page |
| Usage policy | The card's Ethics section says "Deployments must adhere to the Gemma Prohibited Use Policy" (`ai.google.dev/gemma/prohibited_use_policy`). The Gemma documentation lists that policy and an intended-use statement as separate documents beside the Apache-2.0 text. Whether the card sentence binds redistributors beyond Apache-2.0 is a question for the maintainer (decision 2); it does not block redistribution under Apache-2.0. | card |
| Prompts | `config_sentence_transformers.json` names 20 prompts; `default_prompt_name` is null (no prompt unless requested). `query` and `SearchQuery`/`Retrieval-query` are `task: search result \| query: `; `document`, `Document` and `Retrieval-document` are `title: none \| text: `. Others: `task: question answering \| query: `, `task: fact checking \| query: `, `task: code retrieval \| query: ` (CodeRetrieval and InstructionRetrieval), `task: classification \| query: ` (Classification, MultilabelClassification), `task: clustering \| query: `, `task: sentence similarity \| query: ` (STS, SentenceSimilarity, PairClassification, Summarization), `task: search result \| query: ` (BitextMining, Reranking). The card says titled documents use `title: {title} \| text: {content}` written by the caller. | `config_sentence_transformers.json`, card |
| Prompt pooling | `1_Pooling/config.json` `include_prompt: true`; prompt tokens are pooled. sentence-transformers renders text through the chat template (prompt as a system message), which gives the same ids as tokenizing `prompt + text` (checked on the corpus). | modules, reference run |
| Matryoshka | 768 (native), 512, 256, 128; truncate the leading dimensions and L2-normalize again. sentence-transformers `truncate_dim` with `normalize_embeddings` equals slicing the reference embedding and renormalizing (max abs 6e-8 on 20 texts at each size). | card, probe |
| Context length | Card: 8,192 tokens shared across modalities. `max_position_embeddings` 262,144. `sentence_bert_config.json` sets no `max_seq_length` and `tokenizer_config.json` `model_max_length` is 1e30, so sentence-transformers does not truncate: longer text runs silently. The corpus reaches 1,750 tokens; nothing above 2,049 tokens was run. | card, configs |
| 512-to-768 projection | Inside the text backbone: `EmbeddingGemma2TextModel.embedding_projection` (`Linear(512, 768, bias=False)`) is applied to every token after the final RMSNorm, so `last_hidden_state` (the sentence-transformers `token_embeddings`) is already 768 wide. Pooling then averages projected tokens; the code notes that projecting per token equals projecting after mean pooling (linear, no bias). It belongs to the exported graph. | transformers 5.19.0 `modeling_embedding_gemma2.py` |
| Text architecture | 24 layers, hidden 512, intermediate 2,048 (gated GELU-tanh), 4 query heads; head dim 256 and 2 KV heads on sliding layers, head dim 512 and 1 KV head on full layers; Q/K RMSNorm with scale, V RMSNorm without; attention scaling 1.0; RoPE theta 10,000 (sliding) and 1,000,000 (full); embedding scale sqrt(512); projection-only per-layer embeddings (PLE: `Linear(512, 24 x 512)`, RMSNorm, a gated bottleneck block after each layer's MLP); one stored `layer_scalar` per layer. Attention is bidirectional on every layer. | `config.json`, modeling code |
| `layer_types` | sliding x5, full, sliding x5, full, sliding x5, full, sliding x5, full: full attention on layers 5, 11, 17, 23 (0-based), sliding on the other 20. | `config.json` |
| Sliding window | `sliding_window` 512; transformers' bidirectional rule is `abs(q - k) <= 512`, so a token sees up to 1,025 positions (the card's "1024 tokens"). Text over 513 tokens uses it. | `masking_utils.py` |
| Precision | Weights bfloat16; transformers 5 loads bfloat16 by default, so the fp32 reference must pass `dtype=float32`. The card says to use bfloat16 or float32 and never float16 (activations exceed its range; NaN or silently degraded output). | `config.json`, card |
| Parameters | 744,371,992 in `model.safetensors`; text path 271,002,624 (embedding table 262,144 x 512 = 134M), 1.08 GB in fp32. | safetensors header |
| Loading | sentence-transformers needs Pillow and torchvision even for text (it loads `EmbeddingGemma2Processor`, which imports the Gemma 4 image processor). Text-only loading (`vision_config`/`audio_config` = None) gives the same embeddings (max abs 9e-8). | reference run |

## 3. Reference runner and fixtures

`model-pipeline/reference/embeddinggemma2/run_embeddinggemma2.py` (CPU, fp32):

- `tokenizer.jsonl`: 49 corpus cases (46 single, 3 pairs) through the `tokenizers` library on the pinned `tokenizer.json`: ids, tokens, type ids, attention and special-token masks, offsets in Unicode scalars, UTF-16 code units and UTF-8 bytes, decode with and without special tokens. No truncation (no case reaches 8,192 tokens). Byte fallback is exercised by 9 cases (`ws-tab-newline` for `\r`, `ws-unicode-spaces`, `control-chars`, `combining-stack`, `german-turkish-case`, `nfkc-compat`, `math-alnum-astral`, `cjk-ext-b`, `unknown-char`).
- `tokenizer-batch.json`: a right-padded batch (pad id 0) of six cases including `long-mixed`, equal row by row to single encodings.
- `tokenizer-implementations.json`: transformers 5.19.0 `GemmaTokenizer` live pipeline vs `tokenizer.json`: no field differs. Ids differ only on `pair-empty-second` (transformers drops an empty second sequence; the `tokenizers` library encodes `<bos><eos>` for it). sentence-transformers preprocessing vs `tokenizer.json` ids of `prompt + text`: no difference on 46 texts x 3 prompt settings.
- `embeddings.jsonl`: 138 rows (46 single texts with no prompt, `query` and `document`), each with the ids, the fp32 embedding and its norm, computed by the sentence-transformers modules on the `tokenizer.json` ids. `encode()` end to end gives the same vectors (max abs 0).
- `reference.json`: the reference is stable: padded batches of 16 vs single 4.7e-7, eager vs sdpa attention 2.7e-7, 1 vs 6 PyTorch threads 3.1e-7, text-only vs full load 8.9e-8, no NaN/Inf.

Proposed protected copies of these files, plus CPU and CUDA parity evidence, are in a separate stacked pull request (`model-pipeline/protected/embeddinggemma-2/`, `PROPOSED.md` in each directory).

## 4. Export and parity

Scripts: `eg2_text.py` (exportable restatement), `export_embeddinggemma2_onnx.py`, `parity_embeddinggemma2_ort.py`, `compare_embeddinggemma2_csharp.py`. Graphs stay under `/home/mark/Projects/AI-models/embeddinggemma2/` on the workstation; none is committed.

`eg2_text.TextPath` restates `EmbeddingGemma2TextModel.forward` with the two masks built in the graph from the 2-D attention mask, as additive float32 masks (0 or the float32 minimum): a boolean mask would give NaN for a padded query whose window holds no valid key, and NaN survives a zero pooling weight. Outputs: `last_hidden_state` (512, after the final norm, diagnostic) and `token_embeddings` (768). It matches transformers' text model bit for bit (max abs 0 on random ids at 7, 600 and 1,300 tokens, with padding).

| Graph | Exporter | Result | Nodes | Files |
| --- | --- | --- | --- | --- |
| `eg2-text-dynamo-sdpa.onnx` | dynamo, opset 20, dynamic batch and sequence (sequence max 8,192 declared to the exporter) | exported, no repair, `onnx.checker` ok, 159 s | 2,601 | 4.8 MB graph (SHA-256 `734aeaa1...`) + 1,084,030,976 bytes external data |
| `eg2-text-legacy-sdpa.onnx` | TorchScript, opset 18 | exported, no repair (a tracer warning on the sdpa `is_causal` Python bool, constant false because a mask is always passed), 17 s | 7,773 | one 1,085,174,886-byte file |

Neither failure category of the stop rule occurred; no repair was attempted. Parity below is for the dynamo graph (the e5 and GLiNER2 experience: fewer CPU-placed shape nodes on CUDA). The TorchScript graph was checked only on the export smoke (random ids, 9 and 600 tokens: token embeddings 2.1e-2 abs on values in the thousands, like the dynamo graph).

### Parity (138 rows, PyTorch fp32 CPU reference)

Candidates pool `token_embeddings` in NumPy float32. Cosine in float64. Token-embedding magnitudes reach 1,588 to 3,932 per row, so the absolute token-level error is read relative to `max(1, |ref|)`.

| Candidate | Hidden (512) max / p99 | Token embeddings max / p99 / relative max | Embedding max / mean / p99 | Min cosine | Batch of 8 vs single | NaN/Inf |
| --- | --- | --- | --- | --- | --- | --- |
| Python ORT CPU, 12 threads | 1.0e-3 / 2.9e-5 | 2.2e-2 / 7.1e-4 / 6.2e-6 | 4.2e-7 / 2.1e-8 / 8.9e-8 | 0.99999999999473 | 1.8e-7 | none |
| Python ORT CPU, 1 thread | identical to 12 threads | | | | | none |
| Python ORT CUDA, `use_tf32=0` | 1.8e-3 / 3.1e-5 | 3.9e-2 / 7.5e-4 / 1.1e-5 | 4.8e-7 / 2.1e-8 / 8.9e-8 | 0.99999999999536 | 3.0e-7 | none |
| Python ORT CUDA, `use_tf32=1` (ORT default) | 1.08 / 1.3e-2 | 16.9 / 0.31 / 4.7e-3 | 1.9e-4 / 9.6e-6 / 4.1e-5 | 0.9999986 | 3.1e-4 | none |
| C# ORT CPU, 1 and 12 threads | | | 3.7e-7 / 2.1e-8 / | 0.99999999999473 | 1.8e-7 | none |
| C# ORT CUDA, `use_tf32=0` | | | 3.6e-7 / 2.1e-8 / | 0.99999999999536 | 3.0e-7 | none |

- Worst rows are the two long texts (`long-english`, `long-mixed`, 1,572 to 1,750 tokens), which cross the sliding window; rows under 100 tokens reach 7.6e-3 absolute (`ascii-sentence` with the document prompt), at most 2.4e-6 relative.
- Sliding-window lengths (511, 512, 513, 514, 515, 600, 1,024, 1,025, 1,026, 1,027, 1,100, 1,537, 2,049 tokens, prefixes of the longest row ending in `<eos>`, each padded in a batch with a 12-token row): CPU batch vs single 0 for both rows; ORT vs PyTorch at most 2.1e-7 (cosine >= 0.9999999999991); CUDA `use_tf32=0` batch vs single at most 1.6e-7. Padded positions are finite in every output.
- C# vs Python ORT on the same provider: max abs 4.2e-7 (CPU) and 4.2e-7 (CUDA, single rows; 4.8e-7 batched), min cosine 0.99999999999967. The C# smoke feeds `embeddings.jsonl` ids and pools in double; graph outputs were not compared byte for byte.
- Placement on CUDA (ORT profile of 138 single-row runs): 2,423 nodes on the CUDA EP, 130 on CPU, all shape arithmetic (57 `Concat`, 72 `Slice`, 1 `Squeeze` per run).
- Timing is indicative only (GPU 1 at 100% utilization from another job with 10.7 GB in use; CPU shared): 138 single rows took 75 s (Python ORT, 12 threads), 121 s (1 thread), 15 s (Python CUDA), 4.4 s (C# CUDA), 85 to 114 s (C# CPU). Session load 1.1 to 1.8 s.

## 5. Tokenizer gap (owned library at `bdf3a09`, 0.1.0-preview.3)

The Qwen byte-level BPE work merged into main as PR #26 while this spike ran. The Gemma profile is `tokenizer.json` `4d777ef5...`: BPE, 262,144 vocabulary entries (256 `<0xNN>` byte tokens), 514,906 merges in pair form, `byte_fallback` true, `unk_token` `<unk>` (id 3), `fuse_unk` true, `ignore_merges` false, no dropout, no affixes; normalizer `Replace(String " " -> "▁")`; pre-tokenizer `Split(String " ", MergedWithPrevious, invert false)`; post-processor `TemplateProcessing` (`<bos> $A <eos>`, pair `<bos> $A <eos> <bos> $B <eos>` with type id 1); decoder `Sequence[Replace(String "▁" -> " "), ByteFallback, Fuse]`; 24 added tokens, all special, `normalized` false, no strip or single-word flags, ids 0 to 4, 46 to 52, 98, 100, 101, 105, 106, 255999, 256000, 258880 to 258884.

What the merged library covers and what the Gemma profile still needs (read from `TokenizerJsonLoader.cs`, `Normalizers.cs` and the tests at `bdf3a09`; nothing was changed in `tokenizers-dotnet/`):

| Component | State at `bdf3a09` | Gemma needs |
| --- | --- | --- |
| BPE model (port of tokenizers v0.23.2) | Present, used by the Qwen profiles. The loader rejects `byte_fallback: true` and any non-null `unk_token` (`UnsupportedTokenizerComponentException`). | Byte fallback: a character missing from the vocabulary becomes one `<0xNN>` token per UTF-8 byte, each carrying the character's offsets; unknown-token handling with `fuse_unk` (reachable only if a byte token were missing; all 256 exist, but the loader must accept and implement or prove the path). Words are not byte-mapped: symbols are Unicode characters, so the char-not-in-vocabulary path is new. Load cost of a 32 MB file with 515k merges needs measuring; the Qwen fast-path reader may assume byte-level vocabularies. |
| Normalizer `Replace` | Present. `String` patterns are accepted (`SimplePattern.Literal`); `Regex` patterns only from the list `" {2,}"`. No test exercises a `String` pattern or a multi-byte replacement. | Accepted as is; needs conformance evidence for the 1-to-3-byte replacement (" " to "▁") and its offset alignment. |
| Pre-tokenizer `Split` | Present for `Regex` patterns that the hand-written Qwen scanner recognizes, `behavior` `Isolated` only; a `String` pattern is rejected (`BpeLoaderTests` asserts it). | `String` pattern with `MergedWithPrevious`. After the normalizer there is no U+0020 left, so this split never fires on Gemma input (each added-token-free segment is one BPE word, up to the whole text); a minimal implementation still has to be generic or prove the no-op. |
| Post-processor `TemplateProcessing` | Present (e5 and Qwen). | Covered: `<bos>`/`<eos>`, pair template with type id 1. Note `pair-empty-second`: the `tokenizers` library emits `<bos><eos>` for an empty second sequence. |
| Decoders | `ByteLevel` and `Metaspace` only; no `Sequence` decoder. | `Sequence`, `Replace` (string), `ByteFallback` (consecutive `<0xNN>` tokens to bytes, invalid UTF-8 to U+FFFD per tokens-library rules), `Fuse`. |
| Added tokens | Present (AddedVocabulary); the e5 and Qwen profiles use special, non-normalized tokens. | 24 special, non-normalized tokens with non-contiguous ids; likely covered, needs the corpus check (`special-literal-*` cases, e.g. `<mask>`, `<pad>`, `<unk>` literals split out with a lone `▁` before them). |
| NFC, ByteLevel, Qwen scanner | Present | Not used by Gemma. |

Conformance input for the profile: the proposed `tokenizer.jsonl` (49 cases) and `tokenizer-batch.json`. A differential corpus like the Qwen one (`differential.jsonl.gz`) would add coverage for byte fallback on rare scripts.

## 6. Feature contract (proposed)

| Feature | Status proposed | Evidence |
| --- | --- | --- |
| Text embedding, mean pooling over the mask (prompt included), L2 normalization, 768 dimensions | experimental | sections 3 and 4 |
| Prompts | experimental, explicit: the caller chooses a named prompt (`query`, `document` and the card's task prompts) or none; never inserted silently; titled documents formatted by the caller | `config_sentence_transformers.json`; fixtures cover none, `query`, `document` |
| Matryoshka truncation to 512, 256, 128 | experimental (slice and renormalize; equal to sentence-transformers `truncate_dim`) | probe in section 2; fixtures to add |
| Input length | up to 8,192 tokens including `<bos>`, `<eos>` and the prompt; longer input rejected with an error (sentence-transformers would run it silently). Inputs between 2,049 and 8,192 tokens untested until a fixture covers them. | card |
| Batch | experimental (padded batches equal single rows within 3e-7) | section 4 |
| CUDA | experimental with `use_tf32: 0` pinned; TF32 unsupported in the profile | section 4 |
| Pair inputs | unsupported (the tokenizer supports pairs; embeddings do not use them) | |
| bfloat16, float16, quantized variants | unsupported in the reference profile (float16 is unsafe per the card) | card |
| Image, audio, video, interleaved multimodal input; `<\|image\|>`, `<\|video\|>`, `<\|audio\|>` placeholders in text | unsupported. A text containing those literals tokenizes to the placeholder ids; the text-only profile must reject them rather than embed a placeholder without media. | `config.json`, chat template |
| Context beyond 8,192 tokens | unsupported | card |

## 7. Implementation estimate

Package `Starlings.Models.EmbeddingGemma2` on the e5 pattern (`Starlings.Runtime.Onnx`, `Starlings.Acquisition`, the Extensions.AI and ML.NET adapter shapes), plus a Gemma BPE profile in `Starlings.Tokenizers`.

| Work item | Days |
| --- | --- |
| Gemma BPE profile in the owned tokenizer: byte fallback and unknown token in the BPE model, string `Split` `MergedWithPrevious`, `Sequence`/`Replace`/`ByteFallback`/`Fuse` decoders, profile YAML, conformance on the 49 fixture cases, a differential corpus, load-time and memory check on the 32 MB file, independent release | 5 to 8 |
| Reproducible export step (pinned environment, dynamo graph, stripped metadata, digest) and a hosting decision for a 1.08 GB graph (no ONNX file on the Hub) | 1 to 2 |
| Model package: embedder on Runtime.Onnx, explicit prompts, 8,192-token limit and placeholder rejection, Matryoshka option, padded batching, CUDA profile with `use_tf32=0` | 3 to 4 |
| C# fidelity tool and stage results for `model-pipeline evaluate` (the embedding task already exists) | 1 to 2 |
| Extensions.AI and ML.NET adapters, copied from the e5 packages, with their tests | 2 to 3 |
| Acquisition lock, manifest, capability statuses, README, sample, long-input and Matryoshka fixtures | 2 |
| **Total** | **14 to 21** |

What keeps it low: the graph exports with no repair on both exporters, one graph, no custom operator, ONNX Runtime CPU and CUDA already match the reference to 5e-7, and the C# side is the e5 shape. What remains costly is the tokenizer profile, which is new code (byte fallback, decoders) in a separately released library, and the 1.08 GB artifact (hosting, acquisition time, memory).

## 8. Proposed tolerances (values only; no tolerances file)

From the observed worst values above, with headroom for other CPUs and GPUs. For the maintainer to set (plan 10.1).

| Key | Proposed bound | Observed worst (CPU / CUDA TF32 off / C#) |
| --- | --- | --- |
| Input ids, attention mask, special-token mask, offsets (3 units), decode | exact | exact (reference) |
| Embedding max abs (vs reference) | 5e-6 | 4.2e-7 / 4.8e-7 / 3.7e-7 |
| Embedding min cosine | >= 0.9999999 | 0.99999999999473 / 0.99999999999536 / same |
| Token embeddings (768), `\|d\| <= r * max(1, \|ref\|)` | r = 1e-4 | 6.2e-6 / 1.1e-5 / not compared |
| Last hidden state (512) | diagnostic only, not gating (not part of the graph contract) | 1.0e-3 / 1.8e-3 abs |
| Padded batch vs single, embedding max abs | 5e-6 | 1.8e-7 / 3.0e-7 / 3.0e-7 |
| Matryoshka truncated embedding vs truncated reference | same as embedding | 6e-8 (ST vs slice) |
| NaN/Inf | none, including padded positions | none |
| Provider options | CUDA `use_tf32: 0` pinned | TF32 on: 1.9e-4, fails the 5e-6 bound |

## 9. Rescored admission score and recommendation

| Criterion | Weight | Before | After | Reason |
| --- | --- | --- | --- | --- |
| User demand | 25% | 5 | 5 | unchanged |
| Backend feasibility | 20% | 3 | 5 | both exporters, no repair; CPU and CUDA parity at 5e-7; C# parity |
| Reusable architecture coverage | 20% | 4 | 4 | e5 route and adapters reused; new backbone handled by the export; tokenizer profile new |
| Access and licensing clarity | 15% | 4 | 4 | Apache-2.0 confirmed in the card and linked text; the card's prohibited-use sentence needs a maintainer reading |
| Testability | 10% | 4 | 4 | sentence-transformers reference stable to 5e-7; MTEB for task evaluation not run |
| Maintenance burden (5 = low) | 10% | 3 | 3 | third tokenizer family; 1.08 GB artifact; multimodal requests to decline |
| **Weighted total** | | **3.95** | **4.35** | |

Recommendation: admit at Experimental with the text-embedding contract in section 6, route as in section 1. Implementation can start with the package and export step while the Gemma BPE profile is built; the profile gates any release that tokenizes in C#.

## 10. Decisions for the maintainer

1. Admission: admit at Experimental (text only), defer, or block.
2. Rights: whether the card's "Deployments must adhere to the Gemma Prohibited Use Policy" sentence needs to be passed on in package notices or the catalog entry, given the Apache-2.0 license.
3. ONNX artifact source: Starling exports and hosts the 1.08 GB graph (Apache-2.0 permits it with notice), or users export it themselves (needs Python at acquisition time).
4. Long input: reject above 8,192 tokens (proposed), or follow sentence-transformers and run any length.
5. Tolerances: set or change the values in section 8; accept the proposed fixtures (stacked pull request, needs the `protected-change` label).
6. Owner (plan 7 requires a named owner or an explicit Experimental mark) and slot relative to the other work in progress.

## 11. Not done

- No product code: no `ai-dotnet` project, no tokenizer change, no evaluator change.
- No MTEB or other task-quality run; no PyTorch CUDA reference; no bfloat16 profile.
- Inputs between 2,049 and 8,192 tokens were not run; no Windows; one CPU (AVX2, no AVX-512) and one shared GPU (RTX 4060 Ti).
- The TorchScript graph was not taken through the full parity run.
- Timing is not a benchmark (P-008 protocol not run).
- No draft registry manifest (optional in the brief); the fields in sections 2 and 6 are ready for one.
