# Task quality (G3): embeddinggemma-2/video-fp32-cuda (retrieval)

Date 2026-10-17. Evaluator model-pipeline evaluate --g3 0.1.0 at `6ea1bfa218555d4360a81afb48c4050f46e71824`. Exit code 0.

**G3: passed**: every budgeted metric meets its minimum.

Dataset `fe9b83c107c717d55d78e464f0c47ff8b53f05d380423b79c28a8b893c2fac08` (embeddinggemma2-commons-video-retrieval-v1, status None): modality video, queries 50, documents 50, judgments 50. With `dataset.json`: `c1573b0f0715d310875399c48510c2275fc92adb78b1bb618d785bf8af349ed9`. The budget names the data+metadata digest.

| Metric | Value |
| --- | ---: |
| ndcg@10 | 0.8126 |
| recall@1 | 0.6800 |
| recall@5 | 0.9000 |
| recall@10 | 0.9400 |
| recall@100 | 1.0000 |
| mrr@10 | 0.7714 |

| Budgeted metric | Minimum | Value | Status |
| --- | ---: | ---: | --- |
| mrr@10 | 0.7510 | 0.7714 | passed |
| ndcg@10 | 0.7970 | 0.8126 | passed |
