# Proposed: G3 retrieval set, reference scores and budgets (e5, EmbeddingGemma 2, browser int8)

Status: **proposed**. Not accepted and not under `model-pipeline/protected/`. If accepted: the dataset directory moves to `model-pipeline/protected/datasets/miracl-dev-hn-8lang-v1/` (or wherever the maintainer keeps G3 sets), and each candidate in `budgets/` becomes `model-pipeline/protected/<family>/<profile>/task-quality.yaml`.

## Dataset: `miracl-dev-hn-8lang-v1/`

R5 retrieval format (`queries.jsonl`, `corpus.jsonl`, `qrels.jsonl`, `dataset.json`). 200 queries (25 per language: ar, de, en, es, hi, ja, ru, zh), 2,778 passages, 2,000 judgments (relevance 1, and 0 for judged non-relevant passages). Evaluator dataset digest: `614a8d0d4d43a3930cc7ea4fea56a9835c7db108b68b9c071de8e116995d2405`.

| Source | Revision | License (evidence) |
| --- | --- | --- |
| `mteb/MIRACLRetrievalHardNegatives` (MIRACL dev topics, judgments and a hard-negative passage pool per language; 24 Parquet files, SHA-256 in `reference/stable/sources.lock.json`) | `332a9acb49f5e83d5397683f79d23e588f685916` | CC-BY-SA-4.0 (card front matter). Upstream `miracl/miracl` topics and judgments: Apache-2.0 (card at `5be20db9509754dadad47689368639fcec739c00`). Passages: Wikipedia, CC BY-SA. |

**Share-alike.** `corpus.jsonl` is Wikipedia text and stays under CC BY-SA 4.0 with attribution if committed; it is not relicensed by the repository's MIT/Apache terms. Evaluating with it and publishing scores is unrestricted. If the maintainer prefers no SA text in the repository, keep only `dataset.json`, the digest and the selection script, and regenerate the files at evaluation time (byte-identical).

Selection (`reference/stable/select_retrieval.py`, seed 20261014): per language, queries sorted by id, filtered to those with a relevant passage whose judged passages are all in the pool, 25 drawn with `random.Random("20261014:<lang>:queries")`; the corpus is every judged passage of those queries plus 100 passages drawn from the rest of the pool with `random.Random("20261014:<lang>:docs")`. Passage text is `title + " " + text`. Regenerate:

```sh
python -I model-pipeline/reference/stable/fetch_sources.py --cache <cache> --only miracl-hn
python -I model-pipeline/reference/stable/select_retrieval.py --cache <cache> --out <dir>
```

Caveats: the Multilingual E5 technical report (arXiv:2402.05672) lists MIRACL training data in e5's fine-tuning mix; this is the dev split, so the absolute score is optimistic for e5 but the budget compares e5 with itself. recall@100 is 1.0 by construction (each query's judged pool is about 10 passages) and is not a budget metric.

## Reference scores (`reference-scores/`)

Computed with `reference/stable/reference_retrieval.py` (the evaluator's own ranking and metric functions) on CPU, fp32. Embedding files stay outside Git; each scores file records their SHA-256 and the environment.

| Run | nDCG@10 | MRR@10 | recall@10 | top-10 overlap vs PyTorch (mean / min) | CPU time |
| --- | ---: | ---: | ---: | --- | --- |
| e5 sentence-transformers PyTorch (reference) | 0.7978 | 0.7717 | 0.9789 | | 3.3 min |
| e5 Python ORT fp32, `onnx/model.onnx` | 0.7978 | 0.7717 | 0.9789 | 1.000 / 1.0 | 8.4 min (padded batches of 16) |
| e5 Python ORT int8-embed (browser default variant, `374a3cc0…fb19`) | 0.7996 | 0.7759 | 0.9789 | 0.9995 / 0.9 | 5.3 min |
| EmbeddingGemma 2 sentence-transformers PyTorch (reference) | 0.8329 | 0.8142 | 0.9889 | | 88 min (6 threads, shared machine at load 9 to 12) |

No Python ONNX Runtime run for EmbeddingGemma 2 here: its ORT parity (PyTorch vs `embeddinggemma2-text-fp32.onnx`) is already a protected fidelity result, and an ORT pass over this set would cost another hour of CPU.

```sh
P=<encoder venv>/bin/python; S=model-pipeline/reference/stable/reference_retrieval.py; D=<dataset dir>
$P $S --model e5 --model-dir <e5 snapshot> --dataset $D --embeddings e5-st.jsonl --scores e5-st.json --threads 6
$P $S --model e5 --model-dir <e5 snapshot> --dataset $D --backend ort --embeddings e5-ort.jsonl --scores e5-ort.json --baseline e5-st.jsonl
$P model-pipeline/reference/encoder/quantize_e5_browser.py --model <e5 snapshot>/onnx/model.onnx --out <q>
$P $S --model e5 --model-dir <e5 snapshot> --dataset $D --backend ort --onnx <q>/model-int8-embed.onnx \
  --embeddings e5-int8.jsonl --scores e5-int8.json --baseline e5-st.jsonl
<eg2 venv>/bin/python $S --model embeddinggemma2 --model-dir <eg2 snapshot> --dataset $D --embeddings eg2-st.jsonl \
  --scores eg2-st.json --threads 6 --batch 8
```

The .NET candidate is `Starlings.Fidelity.Tool retrieval --model e5|embeddinggemma2 --dataset $D --out <file>` and the gate is `model-pipeline evaluate <manifest> --g3 retrieval --dataset $D --results <file>` (with `--baseline <fp32 embeddings>` for the int8 profiles).

## Budgets (`budgets/`)

| Profile | Minimum | Margin rationale |
| --- | --- | --- |
| e5 `onnx-fp32-cpu`, `-cuda`, `-wasm`, `-webgpu` | `ndcg@10: 0.792`, `mrr@10: 0.765` | reference minus 0.006; fp32 ORT on the same graph reproduces the reference exactly, so the margin only covers tie order |
| e5 `onnx-int8embed-wasm`, `-webgpu` | `top10_overlap_vs_baseline.mean: 0.95`, `ndcg@10: 0.78` | 0.95 is kickoff decision 3; nDCG@10 floor at reference minus 0.02 for a quantized variant |
| EmbeddingGemma 2 `onnx-fp32-cpu`, `-cuda` | `ndcg@10: 0.826`, `mrr@10: 0.808` | reference minus 0.006, as for e5 |

Cost: EmbeddingGemma 2 needs about 90 minutes of CPU for this set on the reference workstation under shared load (the .NET run of the same graph is expected to cost the same order; not measured here). If that is too slow for scheduled CI, the alternative is a 4-language subset (en, es, ja, zh) drawn by the same script, about half the cost, with its own reference score; this proposal keeps one set for both models.

## Decisions for the reviewer

1. Accept the dataset (and the share-alike handling).
2. Set the budgets; the margins are proposals.
3. Where accepted G3 datasets live under `protected/` (the evaluator takes any `--dataset` path; the budget pins the digest).
