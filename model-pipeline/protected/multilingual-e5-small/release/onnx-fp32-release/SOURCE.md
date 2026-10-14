# Proposed: P-030 tokenizer and encoder release set, multilingual-e5-small

Status: **proposed**. Not accepted and not under `model-pipeline/protected/`. Intended destination if accepted: `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/release/`, byte-identical, with the profile's manifest and the evaluator pointed at it for G1/G2 (the development fixtures stay as development data).

## Source

Input: the 232-row holdout `../tokenizer-holdout/corpus.jsonl` (MASSIVE CC-BY-4.0, HumanEval MIT, oasst1 Apache-2.0; sources, licenses and selection in `../tokenizer-holdout/PROPOSED.md`). Output of the existing reference runner, unchanged except for the new `--batch-ids` option (the runner's padded-batch rows were hard-coded to development ids):

```sh
cd model-pipeline/reference/encoder
<encoder venv>/bin/python run_e5.py --model-dir <intfloat/multilingual-e5-small@614241f6 snapshot> \
  --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --corpus ../../proposed/stable/multilingual-e5-small/tokenizer-holdout/corpus.jsonl \
  --out ../../proposed/stable/multilingual-e5-small/onnx-fp32-cpu-release \
  --batch-ids massive-am-ET-6600,massive-ja-JP-5590,massive-ar-SA-4081,massive-th-TH-10816,humaneval-85,oasst1-f974fa98-8707-49c4-8ac1-fb23e84f5919
```

Environment: `encoder/requirements.txt` (torch 2.14.1 CPU, transformers 5.18.0, sentence-transformers 6.1.0, tokenizers 0.23.2, onnxruntime 1.30.0), Python 3.11, Intel i7-5820K. `environment.json` records the asset and corpus digests; its repository commit carries `-dirty` because the run preceded the commit that adds the corpus.

## Contents and reference stability

| File | Contents |
| --- | --- |
| `tokenizer.jsonl` | 232 cases: ids, tokens, masks, offsets in three units, decode (G1, exact) |
| `tokenizer-batch.json` | the six `--batch-ids` rows right-padded together; equivalent to single encodes |
| `embeddings.jsonl` | sentence-transformers fp32 embeddings, one per case (G2) |
| `parity.json` | Python ONNX Runtime on `onnx/model.onnx` against PyTorch on the same ids |

Python ORT against PyTorch on the 232 holdout rows: hidden state max abs 2.56e-6, embedding max abs 1.51e-7, min cosine 0.99999988; padded batch of 16 against single 8.9e-8. These sit well inside the profile's protected bounds, the same margins as the development fixtures, so the holdout does not motivate any tolerance change.

Finding: on 8 of 232 rows (HumanEval prompts ending in whitespace) sentence-transformers' own tokenizer class gives different ids from `tokenizer.json` (it drops the trailing `▁` token). This is the known behavior already recorded for the development corpus (`space-only`, `ws-leading-trailing`); the fixtures follow `tokenizer.json`, as before.

## Decisions for the reviewer

1. Accept as the e5 P-030 release set (G1 and G2), and decide whether the CUDA and browser profiles evaluate against the same rows.
2. The `--batch-ids` runner option (a two-line change in `run_e5.py` and `run_embeddinggemma2.py`, default unchanged).
