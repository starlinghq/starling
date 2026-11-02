# ModernBERT on the encoder route: design note

Status: **proposed**, 2026-11-01 (#102, #101). Nothing in this note changes product code, a capability status, a manifest or a tolerance. It is input for a maintainer decision, not a support claim.

## Problem

ModernBERT (`answerdotai/ModernBERT-base` and `-large`, Apache-2.0) is a newer encoder backbone, and `Alibaba-NLP/gte-reranker-modernbert-base` (Apache-2.0) is a reranker fine-tuned on it. Neither runs on `bert-onnx-cpu-v1` today. The scoping of 2026-10-18 on #102 still holds:

1. The base repository's ONNX graph is the masked-LM export (output `logits`, `[batch, seq, 50368]`); there are no hidden states to pool. Base ModernBERT is a backbone, not a sentence embedder, so even with hidden states it gives no useful `ITextEmbedder` on its own.
2. The tokenizer is byte-level BPE with non-standard special ids (`[UNK]` 50280, `[CLS]` 50281, `[SEP]` 50282, `[PAD]` 50283, `[MASK]` 50284).
3. The graph has two inputs, `input_ids` and `attention_mask`, and no `token_type_ids`.
4. Rotary embeddings, alternating local (128-token window) and global attention, and unpadding make ONNX-versus-PyTorch drift likely, as with nomic; per-model bounds will be needed.

## What the route already has

Reading the code on `main` narrows the route work:

- **Two-input graphs are already supported.** `contracts.json` has a `token_type_ids` boolean, `BAAI/bge-m3` is an entry with `token_type_ids: false`, and `BertSession` passes `token_type_ids` only when the profile says so. No graph-shape change is needed.
- **Reranking is already a task.** `task: reranking`, `pooling: none`, `output_name: logits`, `IReranker` and `BertReranker` exist, and `bge-reranker-v2-m3` set the onboarding pattern (`reference/bert/export_reranker_onnx.py`).

What is missing:

- **The tokenizer profile enum.** `tokenizer_profile` allows only `bert-wordpiece-uncased` and `xlmr-unigram` (both readers refuse other values). A ModernBERT value has to be added to the table schema, `onboarding_bert.py` and `BertModelProfile`, backed by a real profile in starlinghq/tokenizers.
- **The tokenizer itself.** The gte-reranker `tokenizer.json` uses an `NFC` normalizer (supported), a `ByteLevel` pre-tokenizer with `use_regex: true` (the GPT-2 split regex; listed as unsupported in starlinghq/tokenizers `docs/supported-components.md` as last read, which accepts only `use_regex` false behind a supported `Split`; confirm against the 0.3.4 release before starting), a BPE model, and a `TemplateProcessing` post-processor `[CLS] A [SEP]` / `[CLS] A [SEP] B [SEP]` with all type ids 0. So the tokenizer repo needs the GPT-2 pattern in its scanner (a third pattern next to the Qwen ones and Llama 3) with conformance fixtures, including pair encoding, which the reranker needs.
- **Inspection.** `_BERT_MODEL_TYPES` in `onboarding_recipe.py` is `bert`, `xlm-roberta`, `nomic_bert`; `modernbert` and `ModernBertForSequenceClassification` have to be admitted for `inspect` to suggest the route.
- **Export and reference.** A sequence-classification export of our own (upstream ONNX is used only after comparison with the Python reference), and the transformers built-in `ModernBertForSequenceClassification` as reference (no remote code).

## Constraints

- **Contract values for the first candidate** (from the pinned `config.json` and `tokenizer.json`, to be re-read at the pinned revision): `task` reranking, `pooling` none (the head's `classifier_pooling: mean` happens inside the graph), `normalize` none, `hidden_size` 768, `vocabulary_size` 50368, `cls_token_id` 50281, `sep_token_id` 50282, `pad_token_id` 50283, `output_name` logits, `token_type_ids` false, a single logit (`id2label` has one label). `max_tokens` comes from the tokenizer's truncation (8000), not from `max_position_embeddings` (8192); which one the reference actually applies must be checked in the reference run, not assumed.
- **Long inputs.** The local/global attention switch means short-sequence parity says little about long sequences. The corpus needs pairs well beyond the 128-token local window, and the CPU cost of 8000-token inputs has to be measured before claiming that limit.
- **Bounded authority.** Per-model bounds are maintainer-owned; this note proposes none.
- CPU only; no GPU budget (#210) needed.

## Options

**A. Base ModernBERT first** (#102 as written). Exercises export, tokenizer and route, but ends with no usable product: no embedding head, and fill-mask is not a Starling task.

**B. gte-reranker-modernbert-base first** (#101), on the existing reranking task. Exercises the same tokenizer, export, two-input and reference work, and ships a usable `IReranker`.

**C. A ModernBERT embedding fine-tune first.** Also usable, but needs choosing a checkpoint and a sentence-transformers pooling check; nothing about the route favors it over B.

**D. A separate `modernbert` route or plugin.** Not needed: the gaps above are a tokenizer value and an inspection rule, not a new execution shape.

## Recommendation

Option B on `bert-onnx-cpu-v1`, extended rather than forked. The tokenizer profile (working name `modernbert-bpe`) is the long pole and comes first. A ModernBERT embedding fine-tune (option C) follows on the same route as a table entry once B has landed; base ModernBERT stays unadmitted as a product.

## Checklist (one PR each)

1. starlinghq/tokenizers: issue, then PR adding the GPT-2 split pattern for `ByteLevel` with `use_regex: true` (class tables generated from the reference), and the `modernbert-bpe` profile pinned to the gte-reranker `tokenizer.json`, with single and pair fixtures and a randomized differential run. Release.
2. This repo: raise the `Starlings.Tokenizers` pin and every manifest's `processing.tokenizer.version` in one change.
3. Route extension: `modernbert-bpe` as an allowed `tokenizer_profile` in the contract-table rules (`routes/README.md`, `onboarding_bert.py`, `BertModelProfile`), the `modernbert` model type and `ModernBertForSequenceClassification` head in inspection, with tests. No candidate yet.
4. `model-pipeline`: reranker export for `ModernBertForSequenceClassification` (based on `export_reranker_onnx.py`), the transformers built-in reference path, and a check of any upstream ONNX against it.
5. Candidate `gte-reranker-modernbert-base` via `model-pipeline scaffold` (contract entry, recipe, asset lock, README row), pinned by full commit SHA; corpus with long pairs.
6. Onboarding run on CPU, evidence under `docs/evidence/`, profile Experimental. The maintainer adds per-model bounds and the registry manifest.
7. Later, separately: a ModernBERT embedding fine-tune as a table entry; ModernBERT-large; CUDA.
