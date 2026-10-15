# Task quality (G3): gliner2-base-v1/onnx-fp32-cpu (ner)

Date 2026-10-14. Evaluator model-pipeline evaluate --g3 0.1.0 at `10c016c1e8724983306d79a5db02b84326109c32`. Exit code 0.

**G3: passed**: every budgeted metric meets its minimum.

Dataset `0c86bed9f0d7da417d71600dcadeed706aca2d16d22be72ddf82679f5150ecb2` (wnut17-test-v1, status PROPOSED: candidate G3 NER set, not accepted; not gate evidence): rows 300, labels 6, gold_entities 262. With `dataset.json`: `c4322bab27638a6c2b65a3bb306802d4c5f67851d4aa6a90471b32e90d949a55`. The budget names the data+metadata digest.

| Metric | Value |
| --- | ---: |
| exact.micro.precision | 0.2599 |
| exact.macro.precision | 0.2398 |
| exact.micro.recall | 0.5992 |
| exact.macro.recall | 0.5142 |
| exact.micro.f1 | 0.3626 |
| exact.macro.f1 | 0.3206 |
| partial.micro.precision | 0.2964 |
| partial.macro.precision | 0.2787 |
| partial.micro.recall | 0.6832 |
| partial.macro.recall | 0.5990 |
| partial.micro.f1 | 0.4134 |
| partial.macro.f1 | 0.3704 |

| Label | Exact P | Exact R | Exact F1 | Partial F1 | Gold | Predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| person | 0.3636 | 0.7706 | 0.4941 | 0.5471 | 109 | 231 |
| location | 0.3827 | 0.7750 | 0.5124 | 0.5620 | 40 | 81 |
| corporation | 0.2368 | 0.5000 | 0.3214 | 0.3214 | 18 | 38 |
| product | 0.0615 | 0.3636 | 0.1053 | 0.1579 | 22 | 130 |
| creative-work | 0.1818 | 0.2286 | 0.2025 | 0.3291 | 35 | 44 |
| group | 0.2125 | 0.4474 | 0.2881 | 0.3051 | 38 | 80 |

| Budgeted metric | Minimum | Value | Status |
| --- | ---: | ---: | --- |
| exact.micro.f1 | 0.3520 | 0.3626 | passed |
| partial.micro.f1 | 0.4030 | 0.4134 | passed |
