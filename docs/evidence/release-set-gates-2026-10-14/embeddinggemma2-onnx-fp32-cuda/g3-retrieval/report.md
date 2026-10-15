# Task quality (G3): embeddinggemma-2/onnx-fp32-cuda (retrieval)

Date 2026-10-15. Evaluator model-pipeline evaluate --g3 0.1.0 at `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023`. Exit code 0.

**G3: passed**: every budgeted metric meets its minimum.

Dataset `4791590718868341a87ef94f11c3829023386741835947b8305855864c83471e` (miracl-dev-hn-8lang-v1, status PROPOSED: candidate G3 retrieval set, not accepted; not gate evidence until the maintainer commits it under model-pipeline/protected/): queries 200, documents 2778, judgments 2000. With `dataset.json`: `614a8d0d4d43a3930cc7ea4fea56a9835c7db108b68b9c071de8e116995d2405`. The budget names the data+metadata digest.

| Metric | Value |
| --- | ---: |
| ndcg@10 | 0.8325 |
| recall@1 | 0.3229 |
| recall@5 | 0.7875 |
| recall@10 | 0.9889 |
| recall@100 | 1.0000 |
| mrr@10 | 0.8136 |

| Budgeted metric | Minimum | Value | Status |
| --- | ---: | ---: | --- |
| mrr@10 | 0.8080 | 0.8136 | passed |
| ndcg@10 | 0.8260 | 0.8325 | passed |
