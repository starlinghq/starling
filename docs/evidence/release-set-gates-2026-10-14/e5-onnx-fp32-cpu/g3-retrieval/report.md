# Task quality (G3): multilingual-e5-small/onnx-fp32-cpu (retrieval)

Date 2026-10-14. Evaluator model-pipeline evaluate --g3 0.1.0 at `b469654b17262920457e810af410063b13cce8ea`. Exit code 4.

**G3: not-run**: no budget: not a gate result (the dataset digest 4791590718868341 is not the budgeted set 614a8d0d4d43a393).

Dataset `4791590718868341a87ef94f11c3829023386741835947b8305855864c83471e` (miracl-dev-hn-8lang-v1, status PROPOSED: candidate G3 retrieval set, not accepted; not gate evidence until the maintainer commits it under model-pipeline/protected/): queries 200, documents 2778, judgments 2000.

| Metric | Value |
| --- | ---: |
| ndcg@10 | 0.7978 |
| recall@1 | 0.2885 |
| recall@5 | 0.7227 |
| recall@10 | 0.9789 |
| recall@100 | 1.0000 |
| mrr@10 | 0.7717 |

| Budgeted metric | Minimum | Value | Status |
| --- | ---: | ---: | --- |
| mrr@10 | 0.7650 | 0.7717 | passed |
| ndcg@10 | 0.7920 | 0.7978 | passed |
