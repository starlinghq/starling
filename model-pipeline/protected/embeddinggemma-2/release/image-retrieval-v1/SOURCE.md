# EmbeddingGemma 2 G3 text-to-image retrieval set (`image-retrieval-v1`)

The G3 task-quality set for `embeddinggemma-2/image-fp32-cpu` and `image-fp32-cuda`: 149 text queries against 149 images from Wikimedia Commons. Each query is its image's own Commons description. The image is the query's only relevant document (relevance 1). Metrics are nDCG@10 and MRR@10 from `model-pipeline evaluate --g3 retrieval`. The budgets in the two profiles' `task-quality.yaml` were set from the upstream reference's score on this exact set, before any .NET run on it.

**No media bytes are committed.** `corpus.jsonl` names each image by file name and SHA-256, so the dataset digest pins the bytes. The images are fetched by script.

## Selection

No agent wrote or chose any query or image. `model-pipeline/reference/embeddinggemma2/select_eg2_retrieval.py select --kind image --exclude release/media/media.json` (seed 20261015, queried 2026-10-15T19:23Z) did the following:

1. It read `Category:Featured pictures on Wikimedia Commons` from each sort-key prefix A to Z, in API batches of 50 (1,270 members).
2. It kept JPEG and PNG files of at most 15 MB and at most 50,000,000 pixels. The pixel limit is the package's default decode limit, so no row needs `--max-image-pixels`.
3. It kept the license allowlist: public domain, CC0 and CC BY. CC BY-SA, GFDL, NC, ND and unknown licenses were rejected.
4. It dropped the 22 P-030 release media titles, so the sets are disjoint.
5. It required an English `ImageDescription` of at least 20 characters after cleaning: HTML entities unescaped, tags removed, whitespace collapsed. Longer descriptions were cut to 300 characters at the last space. Descriptions shared by two files were dropped.
6. That left 242 eligible files. It drew 150 with `random.Random("20261015:image-retrieval")`.

`image-media.json` is the recorded selection, with each description verbatim. Commons is a live site, so everything is rebuilt from this record and never by re-querying.

`fetch_eg2_release_media.py --media image-media.json` checked every license again through the Commons API (`extmetadata` and the description page's license templates; `licenses.json`, 2026-10-15). It also checked each revision's SHA-1 and size. It rejected one file, `commons-image-133` (U.S. Navy diver, 140218-N-IC111-156): the recorded revision is a truncated JPEG that PIL cannot decode in full, so upstream has no embedding for it. `select_eg2_retrieval.py build` then wrote `queries.jsonl`, `corpus.jsonl`, `qrels.jsonl` and `dataset.json` from the 149 kept files.

```sh
S=model-pipeline/reference/embeddinggemma2
python -I $S/fetch_eg2_release_media.py --media image-media.json --out <media dir>     # numpy and pillow needed
python -I $S/select_eg2_retrieval.py build --media image-media.json --fetched <media dir> --out <dir> \
  --name embeddinggemma2-commons-image-retrieval-v1                                     # byte-identical to this directory
```

## Licenses

**Images:** 89 public domain, 14 CC0, 4 CC BY 2.5, 12 CC BY 2.0, 20 CC BY 3.0 and 10 CC BY 4.0. Each license is as Commons states it for the file (`licenses.json`; data, not a legal review). Authors and credits per file are in `dataset.json` (`files`) and `licenses.json`. The images are not redistributed.

**Queries:** the descriptions are text from the files' Commons description pages. Wikimedia's text license is CC BY-SA 4.0, and descriptions copied from a public-domain source (for example NASA or US military captions) are public domain. `queries.jsonl`, and the descriptions in `image-media.json` and `dataset.json`, are therefore committed under **CC BY-SA 4.0**, with attribution to each file's description page (`source` in `dataset.json`), as for the MIRACL passages. Evaluating with them and publishing scores is unrestricted.

## Format and prompts

The format is the evaluator's retrieval format with a media corpus (`model-pipeline/README.md`, "Task quality (G3)"). `dataset.json` declares `"modality": "image"`. Queries are embedded with the model's `query` prompt (`task: search result | query: `). Images are embedded alone without a prompt, as the model card says: prefixes apply to text only.

## Reference score and budget

Dataset digest: `55fead382caaa09aea32d5cca8ad4a115f801641d7c5bcdd1a28d629f903823e` over the data files. With `dataset.json` included it is `43b3a8d41907ce5fda042554eb8485378b5cacacfa4d994cfebd6bc2631f0607`, which is the form the budgets bind.

The reference is `reference/embeddinggemma2/reference_eg2_media_retrieval.py` at `9a16ec8` (#53). It ran `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b` on sentence-transformers 6.1.0, transformers 5.19.0 and PyTorch 2.14.1+cpu, in fp32 on CPU with 14 threads. The machine was a RunPod EPYC 7H12 pod on 2026-10-15. Each input went through one forward call, and the scoring used the evaluator's own functions. The run took 471 s. The embeddings stay outside Git, with SHA-256 `18694dbb61e80e7ca10b5280f67c2905f00dee98ba524a2afec1185d71130d15`.

| nDCG@10 | MRR@10 | recall@1 | recall@5 | recall@10 |
| ---: | ---: | ---: | ---: | ---: |
| 0.9222 | 0.9008 | 0.8456 | 0.9732 | 0.9866 |

Two queries rank their image below 10. The smallest score gap between a query's image and its best other image is 7.8e-5, on one query; the next is 1.0e-3, and 21 queries have a gap below 1e-2.

**Budget** for `image-fp32-cpu` and `image-fp32-cuda`: `ndcg@10 ≥ 0.916` and `mrr@10 ≥ 0.893`. These are the reference minus 0.006 and minus 0.007, rounded down to three decimals. The margin rule follows the text sets: admit about two adjacent swaps at the top of the ranking. With one relevant document per query, one swap of ranks 1 and 2 lowers mean nDCG@10 by (1 − 1/log2 3)/149 = 0.0025 and mean MRR@10 by 0.5/149 = 0.0034. Two such swaps cost 0.005 and 0.007. For nDCG the text sets' 0.006 is kept, which is slightly wider. For a fidelity-equivalent fp32 profile, these margins absorb float-level near-ties only. Development G2 on these profiles is within 1e-5 max abs per embedding, far below the 1e-3 gaps.
