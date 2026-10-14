# Gates G1, G2 and G3 on the P-030 release sets (2026-10-14)

These runs are the first .NET evaluation on the release sets, G3 datasets and quality budgets that the maintainer accepted on 2026-10-14 (`docs/kickoff.md`, "Release sets, G3 datasets and budgets"; layout in `model-pipeline/protected/README.md`). Each profile directory has a README with commands, digests, metrics against budgets, maximum errors against bounds, and wall clock. Next to it are the unedited `report.md` and `report.json` files of `model-pipeline evaluate`. The candidate outputs (stage results, embeddings, predictions) stay outside Git, and each README lists their SHA-256.

How the release sets are addressed: `model-pipeline evaluate --fixtures <protected release dir>` (G1/G2 staged), `--tokenizer-fixtures <protected .jsonl>` (G1 on a tokenizer holdout or release file), and `--g3 <task> --dataset <protected dataset>` (G3). The C# producers take the same paths. Both evaluator options came in #24. The evaluator reads only `model-pipeline/protected/` and verifies every protected file it reads against Git, and the bounds and budgets come only from the profile's protected `tolerances.yaml` and `task-quality.yaml`. Nothing under `model-pipeline/protected/` was changed, and nothing was tuned.

A manifest gate entry becomes `passed` only where the evaluator reported that gate as passed on the release set, with this evidence path and the date 2026-10-14. Other gates keep their status and get a note. G4 to G7 and maturity are unchanged.

| Profile | G1 | G2 | G3 | README |
| --- | --- | --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-cpu` | passed | passed | not-run: dataset digest form (metrics above the minimums) | [e5-onnx-fp32-cpu](e5-onnx-fp32-cpu/README.md) |

Workstation: Intel i7-5820K (12 threads, AVX2), 94 GB RAM, Linux x64, .NET 10.0.12, ONNX Runtime 1.30.0. It was shared with a one-hour soak measurement and other services, so the wall-clock times are not benchmarks.
