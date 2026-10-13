# M0: Reference runs

Run on 2026-10-05 in a cloud session: 4 vCPU Intel Xeon @ 2.10GHz, 15 GB RAM, no GPU, Linux x86-64. Every model file was pinned by commit SHA and hashed first (`docs/m0/hub-pins.md`). Runners and pinned requirements are in `model-pipeline/reference/` (`README.md` there describes the fixture format). Proposed protected fixtures are in `model-pipeline/protected/`, each directory with a `PROPOSED.md` for maintainer review.

Corpus: `model-pipeline/reference/corpus/tokenizer-corpus-v1.jsonl`, 49 cases (46 single texts, 3 pairs) covering the plan section 6.2 categories: ASCII, multilingual scripts, combining marks (precomposed, decomposed, stacked, Hangul jamo), astral characters (emoji with ZWJ, modifiers and flags, math alphanumerics, CJK Extension B), whitespace (leading, trailing, repeated, tabs, CR/LF, NBSP, ideographic and zero-width spaces), control characters, BOM, U+FFFD, empty input, characters outside the vocabulary, special-token look-alikes for XLM-R, DeBERTa, Qwen and GLiNER2, the e5 `query: `/`passage: ` prefixes, long repeated characters, and two inputs longer than 512 tokens. It is a proposal: the cases were written for this run, not taken from a reviewed external source.

## 1. multilingual-e5-small (P-002, P-003)

e5-small is the working P0 candidate. The maintainer has not chosen the P0 encoder (`docs/kickoff.md`).

- Manifest: `model-pipeline/registry/multilingual-e5-small/onnx-fp32-cpu/manifest.yaml`, validated by `model-pipeline validate registry`. Maturity experimental; G0 checklist, other gates not run.
- Runner: `model-pipeline/reference/encoder/run_e5.py`. Environment: Python 3.11.15, torch 2.14.1+cpu, transformers 5.18.0, sentence-transformers 6.1.0, tokenizers 0.23.2, onnxruntime 1.30.0, numpy 2.4.6.
- Fixtures: `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/`, generated at repository commit `001ced6`.

### Tokenizer reference

The fixtures encode with the shipped `tokenizer.json` read by tokenizers 0.23.2. Offsets are recorded in Unicode scalar values (the upstream unit), UTF-16 code units and UTF-8 bytes.

Other implementations, against that reference, on the 46 single-text cases (`tokenizer-implementations.*.json`):

| Implementation | Cases that differ | Which |
| --- | --- | --- |
| transformers 4.57.6, default (fast) class | 0 | |
| transformers 5.18.0, default class (what sentence-transformers 6.1.0 uses) | 6 | `space-only`, `ws-leading-trailing`, `control-chars`, `special-literal-s`, `special-literal-mask`, `long-mixed` |
| transformers 4.57.6, slow `XLMRobertaTokenizer` | 6 | `space-only`, `ws-leading-trailing`, `korean-jamo`, `special-literal-s`, `special-literal-mask`, `long-mixed` |
| SentencePiece 0.2.2, raw, fairseq id mapping | 6 | same as slow (plus no added-token matching, by construction) |

Causes:

- transformers 5.18.0 does not apply the pinned `tokenizer.json` pipeline. `XLMRobertaTokenizer` rebuilds it from the class: normalizer `Precompiled` without the file's `Replace(" {2,}", " ")`, pre-tokenizer `Sequence[WhitespaceSplit, Metaspace(prepend_scheme always, split true)]` instead of the file's bare legacy `Metaspace`. Vocabulary and added tokens are identical.
- The file's pipeline emits a lone `▁` (id 6) for trailing whitespace and for the space before a special-token literal such as `</s>` or `<mask>`. Slow SentencePiece and transformers 5.18.0 emit nothing there (`long-mixed` differs only in its trailing space, past token 512).
- transformers 5.18.0 also drops U+0085 (NEL), which the file and SentencePiece keep as a token.
- The file's `Precompiled` normalizer leaves decomposed Hangul jamo (`U+1100 U+1161 U+11A8`) uncomposed; SentencePiece composes them to `각`.
- The file maps ZWJ (U+200D) and other zero-width characters to a lone `▁`; all implementations agree on that for e5.

### Embedding reference and ONNX parity

The reference embedding is the sentence-transformers module stack (Transformer, mean Pooling, Normalize) in PyTorch fp32 on CPU, fed the `tokenizer.json` ids (truncated to 512). The candidate is Python ONNX Runtime 1.30.0 (CPU execution provider) on the repository's own `onnx/model.onnx`, fed the same ids. `parity.json`:

| Comparison (46 single texts) | Max abs | Mean abs | p99 abs |
| --- | ---: | ---: | ---: |
| ORT against PyTorch, last hidden state | 2.09e-6 | 1.53e-7 | 5.66e-7 |
| ORT (NumPy mean pooling and L2 norm) against ST embedding | 2.09e-7 | 1.98e-8 | 7.30e-8 |
| NumPy pooling on the PyTorch hidden state against ST embedding | 8.94e-8 | | |
| ST, padded batches of 16, against one row per call | 1.71e-7 | 1.98e-8 | 7.82e-8 |
| ORT padded batch of 6 rows against ST one row per call | 2.09e-7 | 2.17e-8 | 7.81e-8 |
| `SentenceTransformer.encode` end to end, where its ids equal the file's | 0 | | |

Minimum cosine similarity, ORT against ST: 0.99999994. No NaN or Inf. Largest hidden-state errors: `special-lookalike-spaced` (25 tokens) 2.09e-6, `long-english` (512 tokens) 1.70e-6. The tokenizer batch fixture (6 rows, right padding with id 1) matches single encodes row by row.

No tolerance file is proposed. These numbers are the observed baseline for the maintainer to set G2 tolerances from (plan 10.1).

## 2. GLiNER2 tokenizer fixtures (P-004, P-014 input)

- Runner: `model-pipeline/reference/gliner2/run_gliner2_tokenizer.py`. GLiNER2 source at `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf` (installed from Git), `fastino/gliner2-base-v1` at `f9634218e53580c56edf0de97ca1a7d3f1c2354e`, transformers 4.57.6, tokenizers 0.22.2 (GLiNER2 caps transformers below 5). No model weights are needed.
- Path reproduced: `SchemaTransformer.transform_record(text, schema)` with the checkpoint's tokenizer loaded through `load_extractor_tokenizer` (fast). The text is normalized (a `.` appended unless it ends in `.`, `!` or `?`), split by the regex word splitter, each word lower-cased and tokenized alone, and the subwords concatenated after the schema prefix. Schema: entities `person`, `location`. CJK and Thai cases also run the `char` splitter.
- Fixtures: `model-pipeline/protected/gliner2-base-v1/tokenizer/`, generated at repository commit `001ced6`. Word offsets are in Unicode scalars of the normalized text, with UTF-16 and UTF-8 conversions; offsets index the original characters while the word value is lower-cased (`İstanbul` becomes the 9-code-point `i̇stanbul` over an 8-code-point span).

`AutoTokenizer.from_pretrained(..., use_fast=False)` returns the fast class for this checkpoint, because its `tokenizer_config.json` names `DebertaV2TokenizerFast`. The slow comparison therefore builds `DebertaV2Tokenizer` directly.

Fast (`DebertaV2TokenizerFast`, what GLiNER2 uses) against slow (`DebertaV2Tokenizer`), per word, `gliner2-fast-vs-slow.json`: 11 of 46 cases differ. GLiNER2's 10 special tokens have the same ids (128001 to 128010) in both.

| Difference | Fast | Slow | Cases |
| --- | --- | --- | --- |
| Byte fallback | the character, which `convert_tokens_to_ids` maps to `[UNK]` (3) | byte pieces `<0x..>` (`spm.model` has `byte_fallback` true; `tokenizer.json` has it false) | `unknown-char` (ꙮ, U+1FAE8), `cjk-ext-b`, `combining-stack`, `rtl-mixed` (U+202B, U+202C), `ws-zero-width` (U+2060), `control-chars` (U+0000) |
| Zero-width and format characters | a lone `▁` (507) | nothing | ZWSP, ZWJ, ZWNJ, LRM, RLM, BOM, U+FFFD: `ws-zero-width`, `rtl-mixed`, `emoji-zwj`, `bom`, `replacement-char` |
| Hangul jamo | `▁ ᄀ ᅡᆨ` (507, 127449, `[UNK]`) | `▁각` | `korean-jamo` |
| Unigram tie-breaking | `▁a aa aaaaaaaaa ...` | `▁a aaaaaaaaa aaaaaaaaa aa ...` | `repeated-char` (300 × `a`): same pieces, different order of equal-score segments |

Words that produce no subwords in either tokenizer (U+0007, U+001B, U+007F, U+0085 as standalone words) make GLiNER2 log a warning and keep a placeholder position; the fixtures record that as an empty subword list.

Whole-text encodes (`tokenizer-implementations.transformers-4.json`, with `[CLS]`/`[SEP]`): transformers 4.57.6 fast matches the checkpoint's `tokenizer.json` on all 46 cases; slow differs on 8 and raw SentencePiece on 10 (the extra two are GLiNER2 and DeBERTa special-token look-alikes, which raw SentencePiece does not match as added tokens).

## 3. Kev-0.8B execution-route spike (P-027)

Full results: `model-pipeline/admissions/kev.md`, "Spike results". Thresholds are the provisional ones in `kev.md` step 8; no maintainer has set binding values.

- Environment: Kev code `6b719c3c`, Kev-0.8B weights `9a45d25e`, base `dc7cdfe2`; Kev's `uv.lock` versions (Python 3.12.3, transformers 5.17.0, peft 0.21.0) with the torch 2.8.0 CPU build; onnxruntime 1.30.0, onnxruntime-genai 0.17.1. `head.pt` loaded only with `weights_only=True`.
- Reference (CPU fp32, upstream `kev.benchmark`): decision-v7 development accuracy 0.8267405063291139 (n = 1,264) and transfer-v4 development 0.6478658536585366 (n = 656), both equal to the release record.
- Merge: fp32 merge bit-identical to Kev's own; reproduces the reference to 1.79e-7.
- Export (route A): succeeded after one repair (`exclude_mtp=true`). `model.onnx` `55dc88d2...`, `model.onnx.data` `b52e6562...`.
- Python ORT against PyTorch: decision-v7 1,468 questions max 8.43e-5, transfer-v4 764 questions max 5.83e-5, 0 flips, identical accuracy; cached against uncached 6.6e-7; padded batches 0; 7,552-token states max 3.1e-5 at 6.3 GB peak RSS; no NaN or Inf. A 32,768-token state did not complete (process killed, likely memory).
- C# (`Microsoft.ML.OnnxRuntime` 1.30.0, Linux x64): hidden states bit-identical to Python ORT, probabilities within 2.07e-7.
- Fallback routes were not needed; the stop rule was not triggered.
