# M0: GLiNER2 backbone tokenizer and P0 encoder proposal

Status: proposal, checked 2026-10-05. Task: plan section 17.1, days 1-2 ("check GLiNER2's backbone tokenizer and fix the encoder ID and revision"). Kickoff row: "P0 encoder model ID and revision".

Environment limits: direct downloads from huggingface.co were blocked. A later check on 2026-10-05 read Hub text files (configs, `tokenizer.json` headers, file listings with sizes, license metadata) through the Hugging Face MCP connector, at the current `main`; it returns no commit SHAs or file digests. Other evidence comes from the GLiNER2 source at a pinned commit, PyPI, Python package source (transformers, optimum, fastembed), tokenizer assets vendored in third-party GitHub repositories, and web search. Each claim below is tagged:

- **[code]** read from source at a recorded commit or package version.
- **[run]** reproduced locally in a scratch venv.
- **[vendored]** read from a third-party GitHub copy of a Hub asset; content not confirmed against the Hub.
- **[search]** found only through web search; unverified.
- **[hub]** read from the Hub through the connector on 2026-10-05, at `main`; no revision SHA or digest.
- **[download]** downloaded on 2026-10-05 at a pinned commit SHA and hashed; see `docs/m0/hub-pins.md`.

Update, 2026-10-05: Hub downloads now work. Every file this proposal relies on was downloaded at a pinned commit and hashed (`docs/m0/hub-pins.md`). The connector-read facts above were rechecked against the downloaded files where section 5 asked for it. Reference runs on the pinned files are in `docs/m0/reference-runs.md`.

## 1. GLiNER2 facts

| Item | Value |
| --- | --- |
| Repository | https://github.com/fastino-ai/GLiNER2 |
| HEAD inspected | `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf` (2026-09-24, merge of PR #175) |
| Latest tag | `v2.0.0`, commit `3c913c7369301133d3b7699252074c4303ada50e` (2026-08-24). HEAD is ahead of the tag. |
| `gliner2.__version__` at HEAD | `2.0.0` [code] |
| PyPI | `gliner2` 2.0.0 uploaded 2026-08-24T15:13:41Z. Previous: 1.3.2 (2026-06-30), 1.3.1 (2026-05-06), 1.3.0 (2026-04-15). |
| License | Apache-2.0 (code). `fastino/gliner2-base-v1` and `fastino/gliner2-multi-v1`: Apache-2.0 in Hub metadata [hub]. Other checkpoints not checked. |
| Python | `>=3.10` |
| Core dependencies | `pydantic>=2,<3`, `requests`, `tqdm`, `urllib3`. Local inference extra `[local]`: `torch>=2.1,<3`, `transformers>=4.38,<5`, `safetensors`, `peft`, `numpy`. `sentencepiece` and `tokenizers` are not declared directly; `tokenizers` arrives through transformers. |
| Optional | `flashdeberta` (GPU attention kernels, DeBERTaV2 only). |

Released checkpoints, from the README model tables at HEAD [code]:

| Model ID | Architecture | Params (README) | Backbone (README) | Language |
| --- | --- | --- | --- | --- |
| `fastino/gliner2-base-v1` | span (GLiNER2) | 205M | DeBERTa-v3-base | English |
| `fastino/gliner2-large-v1` | span | 340M | DeBERTa-v3-large | English |
| `fastino/gliner2-multi-v1` | span | ~205M | mDeBERTa-v3-base | Multilingual |
| `fastino/gliner2.5-small-v1` | boundary (GLiNER2.5) | 74M | DeBERTa-v3-xsmall | English |
| `fastino/gliner2.5-base-v1` | boundary | 194M | DeBERTa-v3-base | English (README default) |
| `fastino/gliner2.5-multi-v1` | boundary | 287M | mDeBERTa-v3-base | Multilingual |
| `fastino/gliguard-LLMGuardrails-300M` | span fine-tune | ~300M | not stated | |
| `fastino/gliner2-privacy-filter-PII-multi` | span fine-tune | 205M | not stated (multilingual, likely mDeBERTa-v3) | Multilingual |
| `fastino/GLiNER2-Guardrails-PII-Multi` | span fine-tune | 205M | not stated | Multilingual |

The backbone column is the README's statement. Checked against the checkpoints' own files [hub]:

- `fastino/gliner2-base-v1`: `config.json` `model_type` `extractor`, `model_name` `microsoft/deberta-v3-base`, `token_pooling` `first`, `max_width` 8, transformers 4.51.0. `encoder_config/config.json`: `model_type` `deberta-v2`, 12 layers, hidden 768, 12 heads, `max_position_embeddings` 512, `position_buckets` 256, `relative_attention` true, `pos_att_type` [p2c, c2p], `vocab_size` 128,011 (128,000 pieces plus `[MASK]` plus the 10 GLiNER2 tokens; not the 128,100 of the original deberta-v3). Files: `spm.model` (2,464,616 bytes), `tokenizer.json` (8,650,701 bytes), `tokenizer_config.json`, `added_tokens.json`, `special_tokens_map.json`, `model.safetensors` (833,938,108 bytes; Hub reports 208.5M parameters).
- `fastino/gliner2-multi-v1`: `config.json` `model_name` `microsoft/mdeberta-v3-base`, `token_pooling` `first`, `max_width` 8, `counting_layer` `count_lstm`. `encoder_config/config.json`: `deberta-v2`, 12 layers, hidden 768, `vocab_size` 250,112. `added_tokens.json`: `[MASK]` 250101, `[SEP_STRUCT]` 250102 through `[DESCRIPTION]` 250111, in the same order as the base checkpoint. `spm.model` 4,305,025 bytes, `tokenizer.json` 16,337,353 bytes. Hub reports 307.1M parameters, not the README's ~205M.

The repository's real-encoder tests use `microsoft/deberta-v3-xsmall` (`tests/models/boundary/test_*_real_deberta.py`) [code].

Feature set at HEAD [code, README]: entity extraction, single and multi-label text classification (including a constrained `Classifier` builder), structured record extraction (`extract_json`/schema fields), relation extraction, combined multi-task schemas, joint IE (`JointIE`) with constraints and calibration, long-text chunking, span attributes (boundary checkpoints, when the head is enabled), LoRA adapters, training. Boundary checkpoints carry classification, records, and relations "when those heads are enabled", so the feature contract is per checkpoint, not per repository.

Tokenizer loading [code]:

- `gliner2/models/base.py` `load_extractor_tokenizer`: `AutoTokenizer.from_pretrained(repo_or_dir)` with default `use_fast=True`, retried with `extra_special_tokens={}` for older checkpoint metadata. Both architectures load the tokenizer saved inside the checkpoint (`processor.tokenizer.save_pretrained` on save).
- `gliner2/processor.py` `SchemaTransformer.__init__`: `AutoTokenizer.from_pretrained(model_name)` when no tokenizer is passed, then `add_special_tokens({"additional_special_tokens": [...]})` with 10 tokens: `[SEP_STRUCT] [SEP_TEXT] [P] [C] [E] [R] [L] [EXAMPLE] [OUTPUT] [DESCRIPTION]`.
- With a DeBERTa-v3 checkpoint that ships `tokenizer.json`, `AutoTokenizer` returns `DebertaV2TokenizerFast` [run, on the vendored xsmall assets]. On those assets the added tokens took IDs 128001 onward (after `[MASK]`=128000) [run]. `fastino/gliner2-base-v1` stores the same IDs [hub] (`added_tokens.json`, `tokenizer_config.json` `added_tokens_decoder`, and `tokenizer.json` `added_tokens` agree): `[SEP_STRUCT]` 128001, `[SEP_TEXT]` 128002, `[P]` 128003, `[C]` 128004, `[E]` 128005, `[R]` 128006, `[L]` 128007, `[EXAMPLE]` 128008, `[OUTPUT]` 128009, `[DESCRIPTION]` 128010, all `special` true and `normalized` false. Its `tokenizer_config.json` names `DebertaV2TokenizerFast`, `do_lower_case` false, `split_by_punct` false, `vocab_type` `spm`; `[UNK]` is the only added token with `normalized` true.

How GLiNER2 feeds the encoder [code]:

1. Text is split into words by a regex splitter (`gliner2/processing/word_splitter.py`, default `whitespace`; `char` for CJK-style text). Each word is lower-cased (`lower=True` at `processor.py:635`) before subword tokenization, although the DeBERTa-v3 tokenizer is cased (`do_lower_case: false`).
2. Each word and each schema token is tokenized on its own with `tokenizer.tokenize(word)`, cached by an LRU. Subwords are concatenated and mapped with `convert_tokens_to_ids`.
3. The tokenizer post-processor is not used: no `[CLS]`/`[SEP]` is inserted by the tokenizer on this path. Padding uses ID 0 and a 0/1 attention mask (`_pad_batch`).

Consequence for the owned tokenizer: GLiNER2 needs per-word `tokenize` (Metaspace prefix on every word), added-token matching, and `convert_tokens_to_ids`, but not the template post-processor or tokenizer offsets. The P0 encoder will need the post-processor, padding, truncation, and offsets. Both paths share the normalizer, pre-tokenizer, and Unigram model.

## 2. Backbone tokenizer

DeBERTa-v3 (xsmall, small, base, large) uses a SentencePiece **Unigram** model, distributed as `spm.model` plus a converted `tokenizer.json` for the fast tokenizer.

Evidence:

- `spm.model` for deberta-v3-xsmall, -small and -base, vendored in two unrelated GitHub repositories (`jesvijonathan/Snitch-GPT` HEAD `fe4c7f5324938114c3586825a8deb24b38035388`, `WissamAntoun/CamemBERTa`), has one SHA-256: `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd` [vendored]. The xsmall, small and base sizes share one vocabulary.
- Parsed `ModelProto` [run, vendored asset]: `model_type=UNIGRAM`, 128000 pieces, normalizer `nmt_nfkc` with a precompiled charsmap (237,539 bytes), `add_dummy_prefix=true`, `remove_extra_whitespaces=true`, `byte_fallback=true`.
- `tokenizer.json` (vendored xsmall, SHA-256 `05402ffae6dd382a8491b1d29bfc139bec5d332662e86a026f433ce54c25c202`) [vendored, run]:
  - model: `Unigram`, `unk_id=3`, `byte_fallback=false`, 128000 entries.
  - normalizer: `Sequence[Strip, Precompiled(charsmap), Replace(" {2,}" -> " ")]`.
  - pre-tokenizer: `Sequence[Metaspace(replacement="▁", prepend_scheme="always")]`. `split_by_punct=false`, so no `Punctuation` step.
  - post-processor: `TemplateProcessing` `[CLS] $A [SEP]` / `[CLS] $A [SEP] $B [SEP]`.
  - decoder: `Metaspace`.
  - added tokens: `[PAD]`=0, `[CLS]`=1, `[SEP]`=2, `[UNK]`=3, `[MASK]`=128000. `config.json` `vocab_size` is 128100 (padded embedding rows).
- A fresh conversion with transformers 4.57.6 (`DebertaV2Converter`) produced the same component list as the vendored file [run].
- `fastino/gliner2-base-v1` `tokenizer.json` [hub] has the same components: normalizer `Sequence[Strip(left, right), Precompiled, Replace(" {2,}" -> " ")]`, pre-tokenizer `Sequence[Metaspace(replacement "▁", prepend_scheme "always", split true)]`, post-processor `TemplateProcessing` `[CLS] $A [SEP]` / `[CLS] $A [SEP] $B:1 [SEP]:1` (`[CLS]`=1, `[SEP]`=2), decoder `Metaspace`, model `Unigram` with `byte_fallback` false. Added tokens: the five base specials plus the 10 GLiNER2 tokens. The charsmap was not compared byte for byte with the vendored one (its base64 prefix matches the XLM-R one, as expected for `nmt_nfkc`).
- mDeBERTa-v3 (multi checkpoints) is also a SentencePiece Unigram model. `gliner2-multi-v1` has an encoder `vocab_size` of 250,112 [hub], larger than XLM-R's 250,002 pieces, and its `[MASK]` is 250101 against XLM-R's `<mask>` 250001, so the multi vocabulary is not XLM-R's. Its `tokenizer.json` content was not read.

Known fidelity hazard [run]: `spm.model` has `byte_fallback=true` but the fast `tokenizer.json` has `byte_fallback=false`. transformers emits a warning on conversion. For an out-of-vocabulary character such as `ꙮ` (U+A66E), the slow tokenizer yields byte pieces `<0xEA> <0x99> <0xAE>` (IDs 238 157 178), while the fast tokenizer yields `ꙮ`, which `convert_tokens_to_ids` maps to `[UNK]` (ID 3). GLiNER2 uses the fast tokenizer, so the owned profile must reproduce the fast (`tokenizer.json`) behavior and record the slow divergence as a known difference. Other probes (accents, NFKC folding of `𝔘𝔫𝔦𝔠𝔬𝔡𝔢`, `Ⅻ ﬁ ½`, emoji, Japanese, repeated spaces) matched between slow and fast.

Offset semantics:

- GLiNER2 spans are `(start, end)` exclusive-end indices into the original Python `str`, which are **Unicode code points (scalar values)**, produced by the regex word splitter, not by the tokenizer. Entity character offsets map back through `start_idx_map`/`end_idx_map`. A C# port needs a code-point to UTF-16 conversion for slicing; text with characters outside the BMP (emoji, math alphanumerics) is the test case.
- Python's `\w` is Unicode-aware; the C# regex must match it on the conformance corpus, including the URL, email and `@handle` alternatives.
- The fast tokenizer's own `offset_mapping` (used by the P0 encoder path, if exposed) is also in code points via transformers, and Metaspace offsets include the preceding space: `"naïve 😀 café"` gives `▁😀` = (5, 7) [run].

## 3. Candidates

No established embedding model was found that uses the exact DeBERTa-v3 vocabulary. Web search found only individual SNLI fine-tunes of `microsoft/deberta-v3-small` (for example `bobox/DeBERTaV3-small-ST-AdaptiveLayerAllNormalized`) and DeBERTa-v3 cross-encoders (`cross-encoder/nli-deberta-v3-xsmall`), which are classifiers, not embedding models [search]. The practical match is one level up: same algorithm (SentencePiece Unigram) and same asset format (`ModelProto` plus `tokenizer.json` with `Precompiled` normalizer, `Metaspace`, `TemplateProcessing`), different vocabulary. XLM-R vocabulary models fit that.

XLM-R tokenizer check [vendored asset from `ailia-ai/ailia-models`, `multilingual-minilmv2/tokenizer/sentencepiece.bpe.model`, SHA-256 `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865`, `tokenizer_config.json` names `nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`; run]: `model_type=UNIGRAM` despite the `.bpe.model` file name, 250000 pieces, `nmt_nfkc`, `byte_fallback=false`. Converted `tokenizer.json`: `Unigram`, `Sequence[Precompiled, Replace]` normalizer, `Metaspace(prepend_scheme="always", split=true)`, `TemplateProcessing <s> $A </s>`, Metaspace decoder.

Published `tokenizer.json` files [hub]:

- `intfloat/multilingual-e5-small` (17,082,730 bytes, version 1.0): added tokens `<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3, `<mask>`=250001 (all `lstrip` false; `tokenizer_config.json` declares `<mask>` with `lstrip` true, so the slow and fast configs differ on that flag). Normalizer `Sequence[Precompiled, Replace(" {2,}" -> " ")]`. Pre-tokenizer a bare `Metaspace(replacement "▁", add_prefix_space true)` in the legacy serialization (no `prepend_scheme`, no `split` field, no `Sequence` wrapper); the reader must map `add_prefix_space: true` to `prepend_scheme: always`. Post-processor `TemplateProcessing` single `<s> $A </s>`, pair `<s> $A </s> </s> $B </s>`, all `type_id` 0, `<s>`=0, `</s>`=2. Decoder `Metaspace(add_prefix_space true)`. Model `Unigram`, `unk_id` 3, last vocab entry `<mask>`; no `byte_fallback` key (defaults to false). `truncation` and `padding` null. `onnx/tokenizer.json` has the same size (17,082,730 bytes) as the root file; byte identity needs a digest.
- Fallback `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (9,081,518 bytes, compact JSON): differs from e5 in ways a profile must handle. Normalizer is a bare `Precompiled` with no `Replace`. Pre-tokenizer `Sequence[WhitespaceSplit, Metaspace(add_prefix_space true)]`. `truncation` `max_length` 128 `LongestFirst` and `padding` `BatchLongest`, `pad_id` 1, are set in the file. `<mask>` has `lstrip` true. Post-processor, decoder and `Unigram` `unk_id` 3 match e5.

So the published e5 file uses an older serialization than a fresh conversion (bare legacy Metaspace instead of `prepend_scheme`/`split`), and the two XLM-R candidates are not interchangeable at the pipeline level, although they appear to share the vocabulary (same `vocab_size` 250,037 and the same final vocab entries; not compared in full).

| | A. `intfloat/multilingual-e5-small` | B. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | C. DeBERTa-v3 community embedding (e.g. `bobox/DeBERTaV3-small-*`) | D. WordPiece baseline: `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2` |
| --- | --- | --- | --- | --- |
| License | MIT [hub metadata; license file not read] | Apache-2.0 [hub metadata; code: fastembed 0.8.1 metadata] | not checked | MIT / Apache-2.0 [code: fastembed metadata] |
| Params | 117.7M [hub], most in the 250,037 x 384 embedding table (`BertModel`, 12 layers, hidden 384) [hub] | 117.7M [hub]; `BertModel`, 12 layers, hidden 384, vocab 250,037 [hub] | ~141M (deberta-v3-small) [search] | 33M / 22M [search] |
| Tokenizer | XLM-R SentencePiece Unigram, 250K; `tokenizer_class` `XLMRobertaTokenizer` [hub] | XLM-R SentencePiece Unigram, 250K [hub]; `tokenizer_class` `PreTrainedTokenizerFast`, so only `tokenizer.json` applies (its `do_lower_case: true` has no effect on that class) [hub] | DeBERTa-v3 SentencePiece Unigram, 128K (exact GLiNER2 English vocab) | BERT WordPiece, 30,522 |
| Shares with GLiNER2 backbone | algorithm, asset format, Precompiled normalizer, Metaspace, TemplateProcessing; not vocab | same as A | everything | nothing beyond the interpreter skeleton |
| Pooling | mean [hub: `1_Pooling`], then L2 normalize; inputs need `query: ` / `passage: ` prefixes [search] | mean [code: fastembed `PooledEmbedding`] | not checked | CLS + normalize (bge) / mean + normalize (MiniLM) [code: fastembed] |
| Max length | 512 [hub: `model_max_length`] | 128 (`tokenizer.json` truncation `max_length` 128, `LongestFirst`) [hub]; model positions 512 | 512 [search] | 512 / 256 [search] |
| Dim | 384 | 384 [code: fastembed] | 768 [search] | 384 |
| ONNX route | optimum registers `xlm-roberta` for `feature-extraction` [code: optimum 2.3.0]; Hub repo ships `onnx/model.onnx` (470,268,510 bytes), `onnx/model_O4.onnx` (235,052,531), `onnx/model_qint8_avx512_vnni.onnx` (118,346,824), and an `openvino/` folder [hub] | optimum `xlm-roberta` [code]; Qdrant export `qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` used by fastembed [code] | optimum registers `deberta-v2` for `feature-extraction` [code]; no published ONNX checked | fastembed ships ONNX for both [code] |
| Use case fit | multilingual retrieval and similarity; standard benchmark baseline | multilingual similarity, short inputs | weak: individual SNLI fine-tunes, low adoption, quality unknown | strongest English ecosystem, smallest |
| Could not verify | Hub revision SHA, file digests, license file text | Hub revision SHA, file digests, license file text | license, quality, maintenance | Hub revision |

Differences between the XLM-R profile (A, B) and the DeBERTa-v3 profile that M2 must still add:

- Vocabulary and special tokens: `<s> <pad> </s> <unk>` with fairseq ID offset (baked into `tokenizer.json`) versus `[CLS] [SEP] [PAD] [UNK] [MASK]`.
- Normalizer: DeBERTa-v3 adds a leading `Strip`.
- Metaspace serialization: e5 uses the legacy `add_prefix_space` form; GLiNER2 base uses `prepend_scheme: always, split: true` inside a `Sequence` wrapper [hub]. The reader must accept both.
- GLiNER2's 10 added special tokens and per-word tokenization without the post-processor.
- The `byte_fallback` divergence in section 2 (absent in XLM-R, whose spm has no byte fallback).

Tradeoff against WordPiece (D): a WordPiece encoder is the lowest-risk P0 (smallest, best tooling, `BertNormalizer` + `BertPreTokenizer` + `WordPiece` are simple and ML.NET already ships a WordPiece path for comparison). It carries nothing into GLiNER2: the Unigram lattice (Viterbi), the precompiled charsmap normalizer (a serialized double-array trie plus NFKC-style replacements), and Metaspace would all land in M2 next to the GLiNER2 port. The precompiled charsmap is the part most likely to cause conformance failures, so moving it into M1 is the reason to accept a larger P0 model.

## 4. Recommendation

Adopt **`intfloat/multilingual-e5-small`** as the P0 encoder, with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` as the fallback if its license, file layout, or ONNX parity fails at admission.

Reasons:

- It exercises the same tokenizer algorithm and asset format as every GLiNER2 checkpoint (SentencePiece Unigram, `nmt_nfkc` precompiled charsmap, Metaspace, TemplateProcessing). Swapping the vocabulary and adding `Strip`, added tokens and per-word mode is the M2 increment, not a new algorithm.
- Multilingual input gives the conformance corpus non-ASCII, CJK and astral-plane cases from day one, which the GLiNER2 multi checkpoints and the offset work (plan 6.2) need.
- Max length 512 matches DeBERTa-v3's position limit; MIT license (Hub metadata [hub]; license file text still to pin); ONNX export is a registered optimum route.
- Costs: 117.7M parameters [hub], dominated by a ~96M-parameter embedding table (250,037 x 384); mandatory `query: `/`passage: ` prefixes, which belong in the task contract; mean pooling must be implemented with the attention mask.

Do not choose option C. Its exact vocabulary match does not offset unknown license, quality and maintenance. If the maintainer wants the exact GLiNER2 English vocabulary in M1, the alternative is a tokenizer-only conformance profile for `microsoft/deberta-v3-xsmall` (MIT [search]) alongside the e5 model, without shipping a DeBERTa embedding model.

## 5. Download results (formerly "What still needs a full download")

All five items were resolved by download on 2026-10-05 [download]. Full tables are in `docs/m0/hub-pins.md`.

1. `intfloat/multilingual-e5-small` at `614241f622f53c4eeff9890bdc4f31cfecc418b3`: every listed file hashed. `onnx/tokenizer.json` is byte-identical to `tokenizer.json` (`0b44a9d7...`). The repository has no `LICENSE` file; MIT comes from the card front matter (`README.md` line 97). Parameter count from `model.safetensors`: 117,653,760 fp32 parameters plus a 512-element int64 `position_ids` buffer (470,641,600 bytes on disk).
2. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`: tokenizer and config files hashed; weights recorded from the Hub API without download. No `LICENSE` file; Apache-2.0 from the card front matter. It ships the same `sentencepiece.bpe.model` as e5 (`cfc8146a...`).
3. `fastino/gliner2-base-v1` at `f9634218e53580c56edf0de97ca1a7d3f1c2354e`: `tokenizer.json` `1b7fbabf...`, `spm.model` `c679fbf9...`, `tokenizer_config.json`, `added_tokens.json`, `encoder_config/config.json` hashed; `model.safetensors` recorded from the Hub API (`845fc4bd...`, 833,938,108 bytes) without download. The content checks in section 1 hold on the downloaded files.
4. `microsoft/deberta-v3-base` at `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`: `spm.model` is `c679fbf9...`, equal to the vendored hash, so the section 2 findings stand. The repository has no `tokenizer.json`; the vendored xsmall `tokenizer.json` has no base counterpart. `microsoft/deberta-v3-xsmall` was not pinned.
5. GLiNER2 source: the tokenizer fixtures use `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf` (HEAD inspected in M0). `processor.py` changed by 557 lines between `v2.0.0` and that commit; the feature contract must still name one.

## 6. Open questions

1. Which GLiNER2 checkpoint and feature set is the M2 target? English (DeBERTa-v3, 128K vocab) and multi (mDeBERTa-v3, ~250K vocab) are two tokenizer profiles. The multi vocabulary is not XLM-R's (`vocab_size` 250,112, `[MASK]` 250101) [hub]; its `tokenizer.json` content is still unread.
2. Should the owned tokenizer reproduce the fast tokenizer's `[UNK]` behavior for byte-fallback pieces (matches GLiNER2 in practice) and mark slow-tokenizer parity as unsupported? Proposed: yes, with the divergence recorded in the profile.
3. GLiNER2 lower-cases words before a cased tokenizer. This is upstream behavior to preserve, not fix; confirm the fidelity API documents it.
4. Is ~118M parameters (mostly embeddings) acceptable for the M1 CPU benchmark budget, or should P0 trade the shared tokenizer for a 22M to 33M WordPiece model?
5. Does the published `multilingual-e5-small` `tokenizer.json` match a fresh conversion of its `sentencepiece.bpe.model` under the pinned transformers version? The component list matches in type, but the serialization is older (legacy Metaspace `add_prefix_space`) [hub]. Compare token output at admission; pin the shipped file either way. Answered [run, download]: no implementation agrees with the shipped file everywhere. On the 46 single-text corpus cases, transformers 4.57.6's fast class reproduces the file exactly. transformers 5.18.0 rebuilds the pipeline from the class (no `Replace` normalizer, `WhitespaceSplit` before `Metaspace`) and differs on 5 cases. The slow SentencePiece tokenizer differs on 6. Details in `docs/m0/reference-runs.md`. The fixtures pin the shipped file, read by tokenizers 0.23.2.
6. GLiNER2 caps `transformers<5`. Tokenizer reference runs should use the same major version. Done: the GLiNER2 fixtures use transformers 4.57.6 (the newest 4.x) with tokenizers 0.22.2; the e5 runner uses the current releases (transformers 5.18.0, sentence-transformers 6.1.0, tokenizers 0.23.2).

## Sources

- GLiNER2 at `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`: `README.md`, `pyproject.toml`, `gliner2/models/base.py`, `gliner2/processor.py`, `gliner2/processing/word_splitter.py`, `tests/models/boundary/test_*_real_deberta.py`.
- PyPI JSON API for `gliner2`.
- Hugging Face Hub through the MCP connector, 2026-10-05, at `main`: `intfloat/multilingual-e5-small` (listing, `onnx/` listing, `tokenizer.json` header and tail, `tokenizer_config.json`, metadata), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (listing, `config.json`, `tokenizer_config.json`, `tokenizer.json` header and tail, metadata), `fastino/gliner2-base-v1` (listing, `added_tokens.json`, `tokenizer_config.json`, `encoder_config/config.json`, `tokenizer.json` header and tail, metadata), `fastino/gliner2-multi-v1` (listing, `config.json`, `added_tokens.json`, `encoder_config/config.json`, metadata).
- transformers 4.57.6 wheel: `convert_slow_tokenizer.py` (`DebertaV2Converter`, `SpmConverter`).
- optimum 2.3.0 wheel: `exporters/onnx/model_configs.py`.
- fastembed 0.8.1 wheel: `text/onnx_embedding.py`, `text/pooled_embedding.py`, `text/pooled_normalized_embedding.py`.
- Vendored assets: `jesvijonathan/Snitch-GPT` (`models/debertav3{xsmall,small,base}`), `WissamAntoun/CamemBERTa` (`vocab/debertav3-base`), `ailia-ai/ailia-models` (`natural_language_processing/multilingual-minilmv2/tokenizer`).
- Web search, unverified: https://huggingface.co/bobox/DeBERTaV3-small-ST-AdaptiveLayerAllNormalized, https://mixpeek.com/model/cross-encoder/nli-deberta-v3-xsmall, https://thegtmdirectory.com/models/intfloat-multilingual-e5-small/multilingual-e5-small, https://huggingface.co/intfloat/multilingual-e5-small/tree/main/1_Pooling.
