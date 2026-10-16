# Task quality (G3): embeddinggemma-2/image-fp32-cuda (retrieval)

Date 2026-10-16. Evaluator model-pipeline evaluate --g3 0.1.0 at `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8`. Exit code 0.

**G3: passed**: every budgeted metric meets its minimum.

Dataset `55fead382caaa09aea32d5cca8ad4a115f801641d7c5bcdd1a28d629f903823e` (embeddinggemma2-commons-image-retrieval-v1, status None): modality image, queries 149, documents 149, judgments 149. With `dataset.json`: `43b3a8d41907ce5fda042554eb8485378b5cacacfa4d994cfebd6bc2631f0607`. The budget names the data+metadata digest.

| Metric | Value |
| --- | ---: |
| ndcg@10 | 0.9222 |
| recall@1 | 0.8456 |
| recall@5 | 0.9732 |
| recall@10 | 0.9866 |
| recall@100 | 1.0000 |
| mrr@10 | 0.9008 |

| Budgeted metric | Minimum | Value | Status |
| --- | ---: | ---: | --- |
| mrr@10 | 0.8930 | 0.9008 | passed |
| ndcg@10 | 0.9160 | 0.9222 | passed |
