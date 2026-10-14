# Proposed: P-030 tokenizer and text-encoder release set, EmbeddingGemma 2

Status: **proposed**. Not accepted and not under `model-pipeline/protected/`. Intended destination if accepted: `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/release/` (the CUDA text profile would evaluate the same rows).

## Source

Input: the 232-row holdout `../tokenizer-holdout/corpus.jsonl` (MASSIVE CC-BY-4.0, HumanEval MIT, oasst1 Apache-2.0; see `../tokenizer-holdout/PROPOSED.md`). Output of the existing reference runner with the new `--batch-ids` option:

```sh
cd model-pipeline/reference/embeddinggemma2
<eg2 venv>/bin/python run_embeddinggemma2.py --model-dir <google/embeddinggemma-2@914f7f89 snapshot> \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
  --corpus ../../proposed/stable/embeddinggemma-2/tokenizer-holdout/corpus.jsonl \
  --out ../../proposed/stable/embeddinggemma-2/onnx-fp32-cpu-release \
  --batch-ids massive-km-KH-9760,massive-ja-JP-2798,massive-ar-SA-15109,massive-th-TH-13186,humaneval-120,oasst1-06f21312-d9c1-4970-9d7a-b129e8069aba
```

Environment: `embeddinggemma2/requirements.txt` (torch 2.14.1 CPU, transformers 5.19.0, sentence-transformers 6.1.0, tokenizers 0.23.3), Python 3.11; 6.5 minutes on the reference workstation.

## Contents

| File | Contents |
| --- | --- |
| `tokenizer.jsonl`, `tokenizer-batch.json` | 232 cases, exact ids, offsets, decode (G1) |
| `tokenizer-implementations.json` | transformers `GemmaTokenizer` and sentence-transformers preprocessing against `tokenizer.json`: **no differences** on the holdout |
| `reference.json` | reference stability: 696 rows (each text without prompt, with the query prompt and with the document prompt; up to 595 tokens); padded batch of 16 vs single 5.4e-7, eager vs sdpa 3.0e-7, text-only load vs full 1.5e-7, one thread vs six 1.5e-7, no NaN/Inf |
| `environment.json` | packages, asset and corpus digests |

`embeddings.jsonl` (696 rows, 7.0 MB, SHA-256 `c2ffe5e45be11904b7ad0277758c0d8bafc48c6c65dcc87ff932ccbaf574285c`) is **not committed** because of its size: regenerate it with the command above and check the digest, or commit it with the accepted set if the maintainer prefers (the development `embeddings.jsonl` is committed). The reference is deterministic on this machine; another CPU may differ in the last bits, so compare against the digest only on the same environment and otherwise within the profile's tolerance.

## Decisions for the reviewer

1. Accept as the EmbeddingGemma 2 text P-030 release set (G1/G2).
2. Commit the 7 MB `embeddings.jsonl` with it, or keep it outside Git pinned by digest.
