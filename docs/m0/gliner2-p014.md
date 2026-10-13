# P-014: GLiNER2 feature inventory and execution spike

Status: spike result, 2026-10-10. Backlog P-014 (plan section 18). Follow-up (P-015, same day): decisions 1 to 6 of section 8 were taken (route A hosted by Starling, near-tie rule, reject above 512 tokens, spans only, classification out, fixtures accepted); the package is `ai-dotnet/src/Starlings.Models.Gliner2`, evidence in `ai-dotnet/README.md` and `docs/evidence/p015-gliner2-dotnet/`. Inputs: `docs/kickoff.md` row "GLiNER2 revision and feature contract" (settled 2026-10-08), `docs/m0/encoder-selection.md` section 1, `docs/m0/reference-runs.md` section 2, the accepted tokenizer fixtures in `model-pipeline/protected/gliner2-base-v1/tokenizer/`.

Pins: checkpoint `fastino/gliner2-base-v1` at `f9634218e53580c56edf0de97ca1a7d3f1c2354e`, GLiNER2 source at `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf` (version 2.0.0). Every file was downloaded again at that revision and matched `docs/m0/hub-pins.md` (`model.safetensors` 833,938,108 bytes, SHA-256 `845fc4bd93c525b86124c58ab4f56c9eacf8587953086b14c501fab25957c007`). Upstream repository text and the model card were read as data.

Contract under test (kickoff): entity extraction with labels and threshold, English only. Label descriptions join once fixtures cover them. Classification is a named candidate only if this spike lands low. Relations and structured (JSON) extraction are out.

## 1. Summary

| Question | Answer |
| --- | --- |
| Does a full-model ONNX export work? | Yes. The dynamo exporter (`torch.onnx.export(..., dynamo=True)`, torch 2.14.1, opset 18) exports backbone and heads as one graph with no repair. The TorchScript exporter needs two repairs, both in the small count head, none in DeBERTa. |
| Does DeBERTa-v3's disentangled attention export? | Yes, with both exporters, with dynamic batch and sequence. The relative-position bucketing (`Sign`, `Abs`, `Log`, `Ceil`) is computed in the graph, and inputs longer than 512 tokens run (checked to 1,329 tokens). |
| ORT CPU parity (route A, 58 cases) | Hidden state max abs 2.3e-5. Span logits max abs 7.8e-3 on logits up to 1,182 in magnitude (max relative 2.4e-5; max abs 6.2e-5 where \|logit\| < 10). Scores max abs 3.7e-6. Public entities equal on 58/58 cases, 0 threshold flips, no NaN/Inf. Identical at 1 and 12 threads; padded batches bit-identical to single runs. |
| ORT CUDA parity (GPU 1, `use_tf32=0`) | Hidden 2.5e-5, span logits 4.0e-3, scores 2.5e-6, 58/58, 0 flips. With ORT's default `use_tf32=1`: span logits 2.4, a threshold flip on a near-threshold case and a reorder on the 1,329-token case (56/58). The e5 TF32 lesson holds: pin `use_tf32=0`. |
| C# with the owned tokenizer | Input ids, word positions and schema positions exact on 58/58 cases; C# ORT logits bit-identical to Python ORT on CPU and CUDA; public entities equal 58/58 (confidence max abs 1.8e-6 against PyTorch). |
| Recommended route | A: one ONNX graph from ids and routing positions to count logits and span logits, dynamo exporter. Everything around it (word splitting, schema prefix, decoding) is C#. |
| P-015 estimate | 16 to 23 working days for the declared contract (section 7). |

## 2. Rights and provenance (G0 inputs)

| Item | License or terms | Evidence | Redistribution |
| --- | --- | --- | --- |
| GLiNER2 code | Apache-2.0 | `LICENSE` at the pinned commit (SHA-256 `c71d239d...`) | yes, with notice |
| `fastino/gliner2-base-v1` weights | Apache-2.0 | model card front matter only (`license: apache-2.0`); no LICENSE file in the Hub repository | Apache-2.0 permits redistributing a converted ONNX file with license and attribution |
| Tokenizer assets (`tokenizer.json`, `spm.model`) | Apache-2.0 (same card); vocabulary from `microsoft/deberta-v3-base` (MIT) | card, `docs/m0/hub-pins.md` | yes |
| Backbone initialisation | `microsoft/deberta-v3-base`, MIT | hub-pins | |
| Training data | not listed: the card says "multi-domain datasets for NER, classification, and structured extraction" | card | no dataset terms to record; a G0 gap to note, not a blocker |
| Language | English (card `language: en`; DeBERTa-v3 English vocabulary) | card, `encoder_config/config.json` | |

No ONNX file is published for this checkpoint, unlike e5. A C# package therefore needs a converted graph from somewhere; see decision 1 in section 8.

## 3. Feature inventory

Pinned entity-extraction path, `GLiNER2.extract_entities(text, labels, threshold)` → `extract` → `batch_extract` → `ExtractorCollator` (inference) → `SpanExtractorModel` + `ExtractorRuntimeMixin` (span architecture). Source references are `gliner2/<file>:<line>` at `55656fbf`. "Evidence" names the field or file of the proposed fixtures (`model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/extraction.jsonl` unless stated).

| # | Step (upstream) | Behaviour at the pin | Reference evidence | C# work for P-015 | Route |
| --- | --- | --- | --- | --- | --- |
| 1 | Text normalization, `processor.py:537` | Appends `.` unless the text ends in `.`, `!` or `?`; empty text becomes `.`. Offsets index the normalized text, so a span can include the appended `.` and end one past the original text (seen at threshold 0.0: `visited Cupertino.`, end 27 of a 26-character input). | `normalized_text`; cases `empty-text`, `no-final-punct`, `ends-question`, `threshold-0.0` | 5 lines; document the offset consequence | C# |
| 2 | Word splitting, `processing/word_splitter.py:20` | Regex alternation, leftmost, in order: URL (`https?://\S+`, `www.\S+`), email, `@handle`, `\w+(?:[-_]\w+)*`, single non-space. Python `re` semantics: `\w` is L*, Nd, Nl, No and `_` (not combining marks), `\s` includes U+001C to U+001F, `[a-z]` with IGNORECASE also matches U+0130, U+0131, U+017F, U+212A. Code-point offsets. | `words`, `start_map`, `end_map`; tokenizer fixtures (46 cases) | Hand-written scanner over runes (a .NET `Regex` cannot match astral letters with `\p{L}` and differs on `\s` and `\w`). The smoke's scanner matches all 4,810 recorded words of 104 cases. | C# |
| 3 | Lower-casing, `processor.py:635` | Python `str.lower()` per word: full mapping (`İ` → `i̇`, two code points), Final_Sigma; offsets keep indexing the original characters. | `turkish-dotted-i`; tokenizer fixture `turkish` | Custom lower-casing (`Rune.ToLowerInvariant` plus the two special cases); Unicode version differences (Python 3.11 is Unicode 14) noted as a known limit | C# |
| 4 | Word cap `max_len`, `processor.py:640` | Optional; drops words after the first `max_len` before encoding. Default `None`: no truncation of any kind. | `truncation-max-len-40`, `truncation-max-len-200` | Out of the initial contract (listed unsupported) unless chosen as the long-input policy (decision 3) | C# |
| 5 | Schema prefix, `processor.py:1024`, `:1204` | One entity schema: `( [P] entities ( [E] l1 [E] l2 ... ) )`, then `[SEP_TEXT]`, then words. Labels keep their case, are deduplicated (dict keys, first occurrence) and keep the caller's order. With descriptions the prompt token is `entities [DESCRIPTION] label: text ...` (one token string with the special token inside). | `schema_tokens`, `labels`; `label-duplicate`, `label-case`, `descriptions` | Build the token list; already done in the smoke (exact on 58/58) | C# |
| 6 | Subword encoding, `processor.py:1374` | Each schema token and each word tokenized alone (`tokenizer.tokenize`, fast tokenizer, no template, no `[CLS]`/`[SEP]`), pieces mapped with `convert_tokens_to_ids` (unknown → 3). Positions recorded: first subword of each word (a word with no pieces keeps a placeholder at the next position), `[P]`, each `[E]`. | `input_ids`, `word_index`, `schema_index`; tokenizer fixtures | Owned tokenizer `Encode(word, addSpecialTokens: false)` + `TokenToId`; profile `gliner2-base-v1-unigram` | C# (owned tokenizer) |
| 7 | Padding, `processor.py:675` | Right padding with id 0, mask 0/1; word and schema positions padded with 0. | parity `batch` | Batch builder; group by label count (row 12) | C# |
| 8 | Backbone | DeBERTa-v3-base (12 layers, 768, relative attention `p2c|c2p`, 256 position buckets, no absolute positions), `last_hidden_state`. No length limit in code; 512 is `max_position_embeddings`. | tensors `last_hidden_state` (not committed; digests in `tensor_sha256`) | none | ONNX |
| 9 | Token pooling | `first`: the first subword vector of each word (`_extract_embeddings_fast`, `processor.py:1503`). Schema vectors at `[P]` and `[E]`. | `word_index`, `schema_index` | none (gather in graph) | ONNX |
| 10 | Span representation, `layers.py:361` | `SpanMarkerV0`: start MLP and end MLP (768→3072→768), concatenation, ReLU, out MLP (1536→3072→768), for every start word and width 0..7 (`max_width` 8). | `span_logits_valid` | none | ONNX |
| 11 | Count gate, `runtime.py:641` | `count_pred([P] vector)` (MLP 768→1536→20); argmax 0 gives an empty result (`{"entities": {}}`). | `count_logits`, `pred_count`, `gate_open` | argmax over 20 logits | ONNX + C# |
| 12 | Label projection, `layers.py:188` | `CountLSTMv2`: one GRU step from a learned position embedding with the `[E]` vector as state, then a 2-layer transformer (d=128, 4 heads, post-norm) over the labels, so label vectors depend on the whole label set and its order. Only count row 0 is read for entities, and count rows are independent, so the graph fixes the count at 1. Labels are not masked: one padded batch must have one label count. | `label-order-a/b/c` (same entities, different logits); wrapper check `wrapper_span_logits_max_abs_valid` ≤ 1.2e-4 | none; batching rule in C# | ONNX |
| 13 | Scoring, `runtime.py:654` | `einsum(span_rep, label_proj)` → logits [labels, words, 8]; `sigmoid` in fp32. Most entity scores saturate to exactly 1.0. | `span_logits_valid`, output `confidence` | `1/(1+exp(-x))` in float | ONNX + C# |
| 14 | Threshold, `runtime.py:1147` | `score >= threshold`, the threshold rounded to fp32 by torch's comparison. Thresholds 0.0 and 1.0 are degenerate: 0.0 keeps every span (scores underflow to 0.0), 1.0 keeps only saturated scores. | `threshold-*`, `threshold-near-*`, `threshold_margins` | compare in float | C# |
| 15 | Span to characters, `runtime.py:1147` | `char_start = start_map[s]`, `char_end = end_map[e-1]`, surface `text[cs:ce].strip()`; empty surfaces dropped. | output `start`, `end`, `text` | code-point to UTF-16 conversion for slicing; Python `strip` whitespace set | C# |
| 16 | Overlap handling, `candidate_decoder.py:172` | Default for span checkpoints (`overlap_policy=None`): per label, stable sort by confidence (descending), greedy drop of character-overlapping spans. Spans of different labels may overlap (`city: Boston` and `sports team: Boston Celtics`). Explicit policies (`allow`, `nested`, `flat`, `longest`) exist. | `nested-*`, `same-span-two-labels`, `label-multiword` | stable `OrderByDescending` + greedy check; explicit policies unsupported in the contract | C# |
| 17 | Output format, `runtime.py:135`, `:720` | `{"entities": {label: [...]}}` in label order; with `include_spans` and `include_confidence` each item is `{text, confidence, start, end}`. Plain format returns strings and deduplicates on lower-cased text, so repeated entities collapse (`Paris` ×3 → `["Paris"]`; `NASA`, `Nasa`, `nasa` → `["NASA"]`). Gate closed → `{}`; nothing above threshold → `{label: []}`. | `output`, `output_plain`; `repeated-*` | typed results (label, text, start, end, confidence, offset unit); decide whether the plain dedup is exposed (decision 4) | C# |
| 18 | Result order | Descending confidence within a label, ties in torch.where order (start, then width). Saturation at 1.0 makes ties common, so order is sensitive to backend noise (CUDA with TF32 reorders a 1,329-token case). | parity `cases_with_order_only_differences` | match order exactly with TF32 off; tolerance rule for near-ties (section 6) | C# |
| 19 | Batch behaviour, `runtime.py:250` | `batch_size` 8 by default; padded batches; per-text schemas allowed. Reference batch vs single: same entities on 46/46 cases (threshold 0.5, no word cap), confidence max abs 3.6e-7. | `batch.json` | batch API grouping texts by label count | C# |
| 20 | Long inputs | No truncation by default; 566 and 1,329-token inputs run and extract entities across the whole text. `extract_long` (384-word chunks, 64 overlap, merge) is a separate API. | `long-over-512-tokens`, `long-over-1024-tokens` | choose a policy (decision 3); `extract_long` out of contract | C# |
| 21 | Malformed input | Inference collation uses `error_policy="fallback"` (`processor.py:401`): a record that fails to transform becomes a dummy record and yields an empty result with a log warning. `strict_extraction` raises per-sample decode errors. | not exercised | fail explicitly in C# (no silent dummy) | C# |
| 22 | Precision and device options | fp32 reference. `quantize()` (fp16), `compile()`, `flashdeberta` are optional upstream paths. | | separate variants; out of contract | |

Out of contract (kickoff), confirmed present upstream and listed `unsupported` in the draft manifest: relation extraction (`_extract_relations`), structured extraction (`_extract_structures`, `extract_json`), text classification (`_extract_classification_result`, classifier MLP over `[L]` vectors), per-label thresholds, `dtype="str"`, regex validators, entity attributes, explicit overlap policies, `extract_long`, the `char` splitter, LoRA adapters. Label descriptions: one development case passes end to end (`descriptions`); the contract keeps them out until more fixtures exist. English only: the corpus has non-ASCII Latin names (accents, Nordic letters, Turkish dotted I, fullwidth Latin, math bold, NBSP, zero-width characters); no other language was tested and none is claimed.

What the fixtures do not cover: the closed count gate (the count head predicted 1 on all 58 cases, including empty text and text without entities), more than 12 labels, descriptions beyond one case, and adversarial splitter input (very long URLs, `@` runs).

## 4. Reference runner and corpus

- `model-pipeline/reference/gliner2/run_gliner2_extraction.py` runs the pinned public API on CPU fp32 (torch 2.14.1+cpu, transformers 4.57.6, tokenizers 0.22.2; `requirements-export.txt`) and records, per case, the public output with and without confidence/spans, the model inputs, count logits, span logits over valid spans, threshold margins and SHA-256 digests of the stage tensors. With `--tensors-out` it writes the stage tensors (input ids, mask, positions, last hidden state, word and schema vectors, count logits, span representations, label projections, span logits) as `.npz` (76 MB for the corpus; not committed).
- `gliner2_entity.py` restates the entity path as exportable modules and as a pure-Python decoder. The decoder reproduces the public output exactly on 58/58 cases from the reference logits; the restated modules match GLiNER2's logits within 1.2e-4 (different matmul shapes, and the fused attention fast path in the reference) with the same entities on every case.
- Corpus: `model-pipeline/reference/corpus/gliner2-extraction-corpus-v1.jsonl`, 58 cases. Agent-written development data, not the P-030 release set. Categories (plan 10.1): basic (2), empty results (5), overlapping and nested candidates (4), repeated entities (3), label order, subset, duplicates, case, multi-word and 12 labels (8), threshold sweep 0.0 to 1.0 (8), near-threshold cases 6.5e-5 to 1.3e-4 from a reference score (3), max width (2), normalization (4), splitter alternatives (4), non-ASCII Latin and offsets (10), descriptions (1), long inputs over 512 and 1,024 tokens (2), word-cap truncation (2).
- PyTorch reference stability: 6 threads against 1 thread moves hidden states by 9.5e-6 and span logits by 1.0e-3; entities (labels, texts, offsets, order) are unchanged on 58/58, confidences differ on 2.

## 5. Export spike

Scripts: `export_gliner2_onnx.py`, `parity_gliner2_ort.py`, `dump_gliner2_ort.py`. Graphs stay under `/home/mark/Projects/AI-models/gliner2/onnx/` on the workstation; none is committed.

| Route | What | Exporter | Result | Nodes | Size |
| --- | --- | --- | --- | --- | --- |
| A | ids, mask, word positions, schema positions → count logits, span logits (+ hidden state for diagnosis) | dynamo | exported, no repair; `onnx.checker` ok | 1,068 | 3.0 MB graph + 829 MB external data |
| A | same | TorchScript | failure 1, repaired; then failure 2, repaired (below) | 4,996 | 830 MB |
| B | backbone only → last hidden state | both | exported, no repair | 877 / 4,588 | 735 MB |
| H | heads only (word and schema vectors → logits); B + H is the split route | both | dynamo: no repair; TorchScript: same two failures | 188 / 373 | 94 MB |

TorchScript exporter failures (both in `count_embed`'s `nn.TransformerEncoder`, none in DeBERTa):

1. Export/operator. In eval mode `nn.TransformerEncoderLayer` takes the fused fast path `aten::_transformer_encoder_layer_fwd`, which has no ONNX symbolic (`UnsupportedOperatorError`). Repair 1: `torch.backends.mha.set_fastpath_enabled(False)` during export (same function, standard operators). Reproducer: `export_gliner2_onnx.py --routes H --exporters legacy --keep-mha-fastpath`.
2. Export/shape. After repair 1 the graph exports but fails at run time when the label count differs from the trace (`Reshape` `/heads/transformer/transformer/layers.0/self_attn/Reshape_4`: input {5,1,128} to requested {3,4,32}): the tracer recorded the trace's label count inside `nn.MultiheadAttention`'s reshapes. Repair 2: the encoder layers restated with explicit operators (`gliner2_entity.encoder_layer_explicit`, `--explicit-attention`). Reproducer: export H with repair 1 only and run it with a label count other than 3.

Two repairs on two different failures; the stop rule was not reached. The dynamo exporter needs neither (it decomposes the fast path and keeps the label axis symbolic). Re-exporting gives byte-identical weights; the dynamo graph file differs only in `metadata_props` that carry source stack traces, so a reproducible digest needs those stripped. The TorchScript export was byte-identical across runs.

### Parity (58 cases, PyTorch fp32 CPU reference)

Span logits are compared over valid spans; entity equality is the public output (labels, texts, offsets and order), decoded from ORT logits with the same decoder.

| Graph | Provider | Hidden max / p99 | Count logits max | Span logits max / mean / p99 | Scores max | Entities equal | Flips | Batch vs single (span logits max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A dynamo | CPU, 1 and 12 threads | 2.3e-5 / 3.1e-6 | 4.6e-5 | 7.8e-3 / 9.3e-5 / 4.5e-3 | 3.7e-6 | 58/58 | 0 | 0 (bit-identical) |
| A TorchScript (repaired) | CPU, 1 and 12 threads | 1.8e-5 / 2.9e-6 | 4.2e-5 | 3.5e-3 / 6.6e-5 / 2.0e-3 | 2.1e-6 | 58/58 | 0 | 0 |
| B + H dynamo | CPU | same as A dynamo | | | | 58/58 | 0 | 0 |
| A dynamo | CUDA, `use_tf32=0` | 2.5e-5 / 2.6e-6 | 5.0e-5 | 4.0e-3 / 7.6e-5 / 2.6e-3 | 2.5e-6 | 58/58 | 0 | 1.6e-3 |
| A dynamo | CUDA, `use_tf32=1` (ORT default) | 8.9e-3 / 1.1e-3 | 3.2e-2 | 2.4 / 3.5e-2 / 1.4 | 1.4e-3 | 56/58 | 1 | 2.3 |
| A TorchScript | CUDA, `use_tf32=0` | 2.2e-5 | 4.2e-5 | 2.7e-3 | 1.5e-6 | 58/58 | 0 | 2.1e-3 |

- Span logits reach 1,182 in magnitude. Relative to `max(1, |ref|)` the largest ORT error is 2.4e-5 on CPU and 1.4e-5 on CUDA; where `|logit| < 10` (unsaturated scores) the largest absolute error is 6.2e-5 (CPU) and 5.3e-5 (CUDA).
- The worst span-logit case is `label-duplicate`; the worst hidden-state case is a short one on CPU and `long-over-1024-tokens` on CUDA.
- Near-threshold cases (`threshold-near-*`, margins 6.5e-5 to 1.3e-4) decode the same on every `use_tf32=0` read. The degenerate thresholds 0.0 and 1.0 also matched.
- TF32 cases: `threshold-near-below-patient` drops `patient` (person; reference score 0.4370651, TF32 score 0.4367090, threshold 0.437), the flip the near-threshold cases exist to catch. `long-over-1024-tokens` keeps the same entity set but reorders two `person` and two `location` items whose reference confidences differ by less than 5e-7.
- No NaN or Inf anywhere. Same-session repeats are bit-identical. The batch check groups cases by label count (route A's input contract) in batches of up to 8 and compares 56 rows (cases alone in their group are skipped); on CUDA it leaves out the two cases over 600 tokens because GPU 1 was shared with another job.
- Placement on CUDA (ORT profile): A dynamo 759 nodes on CUDA, 33 on CPU (shape arithmetic: `Concat`, `Slice`, `Squeeze`, `Reshape`, `Mul`, `Add`), one Memcpy. The TorchScript graph puts 984 of 1,994 nodes on CPU (shape subgraphs), so the dynamo graph is the better CUDA artifact.
- Timing is indicative only (CPU and GPU 1 shared with other jobs): route A dynamo runs the 58 cases one by one in 20 s on 1 CPU thread, 12 s on 12, and 1.2 s on CUDA; session load 0.7 to 1.5 s. The C# smoke: 25 s (1 thread), 10 s (12 threads), 1.1 s (CUDA).

### C# smoke

`/home/mark/Projects/AI-models/gliner2/csharp/` (throwaway, outside `ai-dotnet/`): Starlings.Tokenizers from source (profile `gliner2-base-v1-unigram`, `tokenizer.json` `1b7fbabf...`), Microsoft.ML.OnnxRuntime 1.30.0 (CPU) and Microsoft.ML.OnnxRuntime.Gpu 1.30.0 (CUDA, `use_tf32=0`), route A dynamo. It ports text normalization, the Python-compatible splitter and lower-casing, the schema prefix and routing positions, the gate, thresholding, span mapping, greedy suppression and formatting (515 lines including the comparison harness).

| Check | CPU (1 and 12 threads) | CUDA `use_tf32=0` | CUDA `use_tf32=1` |
| --- | --- | --- | --- |
| Splitter and lower-casing (46 tokenizer fixture cases + 58 extraction cases, 4,810 words) | exact | | |
| Input ids, word and schema positions, character maps | 58/58 exact | 58/58 | 58/58 |
| Span logits vs Python ORT (same provider) | 0 (bit-identical) | 0 | 2.4 vs PyTorch (TF32) |
| Public entities vs PyTorch reference | 58/58 | 58/58 | 56/58 (one flip, one reorder) |
| Confidence max abs vs reference | 1.8e-6 | 1.9e-6 | 3.6e-4 |

Route B with the heads ported to C# was not needed to show feasibility: the full graph works, and the split B + H graphs give identical numbers. The heads are about 23.5M parameters (94 MB) of small MLPs, one GRU step and a 2-layer, 128-wide transformer; porting them is possible but would add a second numeric implementation to qualify.

## 6. Proposed tolerances (not a tolerances file)

From the observed numbers above, with headroom for other CPUs and GPUs. For the maintainer to set (plan 10.1); no `tolerances.yaml` is proposed.

| Key | Proposed bound | Observed worst (CPU / CUDA TF32 off) |
| --- | --- | --- |
| Normalized text, words, character maps, input ids, attention mask, word and schema positions | exact | exact |
| `last_hidden_state` max abs | 1e-4 | 2.3e-5 / 2.5e-5 |
| `count_logits` max abs; gate decision | 5e-4; exact | 4.6e-5 / 5.0e-5 |
| `span_logits` (valid spans), `|d| <= 1e-4 * max(1, |ref|)` | relative 1e-4 | 2.4e-5 / 1.4e-5 |
| Scores (sigmoid) max abs | 2e-5 | 3.7e-6 / 2.5e-6 |
| Public confidence max abs | 2e-5 | 1.8e-6 / 1.9e-6 |
| Entities (label, text, start, end) | exact, including order, except: a span may flip only if its reference score is within 2e-5 of the threshold, and items whose reference confidences differ by at most 2e-5 may swap places | 58/58 exact, 0 flips |
| Padded batch vs single, span logits | relative 1e-4 | 0 / 1.6e-3 abs (1.4e-6 relative) |
| NaN/Inf | none | none |
| Provider options | CUDA `use_tf32: 0` pinned | |

The order rule needs a decision (section 8): either it is declared semantically irrelevant within near-ties, as above, or order must match exactly and TF32-off is the only allowed CUDA profile. With TF32 off no reorder was seen.

## 7. P-015 estimate (route A)

| Work item | Days |
| --- | --- |
| Reproducible export step (pinned environment, metadata stripped, digest, published lock) and hosting of the ONNX artifact | 1 to 2 |
| GLiNER2 preprocessing in C# (normalization, Python-compatible splitter, lower-casing, schema prefix, routing positions, label dedup), tests against the 46 + 58 fixture cases | 3 to 4 |
| Execution on `Starlings.Runtime.Onnx` (session reuse, padded batches grouped by label count, CUDA profile) | 2 |
| Decoder (gate, fp32 threshold, scalar to UTF-16 offsets, greedy suppression, order, typed results, plain-format option) | 2 to 3 |
| Evaluator: `extraction` task in `model-pipeline evaluate` (stage results format for logits and entities, the tolerance keys above, flip and tie accounting); today it rejects non-embedding tasks | 3 to 4 |
| C# fidelity tool for stage 3 (ids path and text path), like the e5 one | 1 to 2 |
| Acquisition lock (ONNX artifact + `tokenizer.json`), manifest, capability statuses, README, sample | 2 to 3 |
| Long-input policy and its tests; malformed-input errors | 1 |
| CUDA parity tests (TF32 off) | 1 |
| **Total** | **16 to 22**, about 23 with review rounds |

Optional additions: label descriptions (+1 to 2 days, mostly fixtures and a description-ordering rule); classification (+4 to 6 days: classifier head output in the graph, softmax/sigmoid and multi-label rules, `[L]` prefix fixtures, a second task in the evaluator).

What made the estimate low: no custom operator, no TorchSharp, one graph, owned tokenizer already exact, C# smoke already matching. What remains costly is discrete fidelity (Python regex and case semantics, order under saturated scores) and the evaluator extension, which any extraction model needs.

## 8. Recommendation and decisions needed

Recommendation: proceed with the declared contract (entity extraction with labels and threshold) on route A. The spike landed low by the kickoff's measure. Numbers for the M2-slot question: P-015 is 16 to 22 working days with no open execution risk; Kev's spike found builder failures on the CUDA fp32 graph and an owned Qwen BPE profile still missing. This document does not decide the slot.

Decisions for the maintainer:

1. ONNX artifact source. No ONNX file exists on the Hub for this checkpoint. Options: Starling exports and hosts the graph (Apache-2.0 permits it with license and attribution; needs a hosting location and a reproducible-export check), or users run the export themselves (needs Python and torch at acquisition time, which conflicts with the no-Python install story).
2. Result order under near-ties: accept the tie rule in section 6, or require exact order (then CUDA TF32 is excluded by policy, as for e5).
3. Long inputs: GLiNER2 accepts any length; options are to reject above 512 tokens, follow upstream with no limit (tested to 1,329 tokens), or add the `max_len` word cap as an explicit option. The draft manifest says `truncation: error` at 512 as a placeholder.
4. Plain output format: GLiNER2 deduplicates repeated entities on lower-cased text when spans are not requested. Expose that mode, or return spans only.
5. Classification: the kickoff makes it a candidate if this spike lands low; it costs 4 to 6 more days.
6. Proposed fixtures: accept `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/` and `onnx-fp32-cuda/` (separate pull request, needs the `protected-change` label) as development fixtures, and set tolerances.

## 9. Not done

- No product code: no `ai-dotnet` project, no package, no evaluator change (P-015).
- No PyTorch CUDA reference read (ORT CUDA was compared against the CPU reference only).
- One GPU (RTX 4060 Ti, shared with another job, about 5.6 GB free) and one CPU; no Windows.
- Timing is not a benchmark (P-008 protocol not run).
- Route B with C#-ported heads was not built (not needed once route A passed).
- The closed count gate has no fixture.
