# multilingual-e5-small `onnx-fp32-cpu` on the P-030 release sets

Run on 2026-10-14 at `b469654` (main after #24). All inputs come from the accepted release sets (kickoff 2026-10-14, "Release sets, G3 datasets and budgets"). The evaluator checked the protected files against Git (`protected_evidence: verified`, base `b469654`), and the binding tolerances and budget come from the profile's protected directory. Nothing under `model-pipeline/protected/` was changed.

| Gate | Set | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- | --- |
| G1 | tokenizer holdout `multilingual-e5-small/release/onnx-fp32-release/tokenizer.jsonl` (232 rows: MASSIVE 208, HumanEval 8, oasst1 16) | **passed** | ids and attention mask exact on 232/232. Tokens, type ids, special-tokens mask, decode (with and without special tokens) and offsets in all three units are also exact on 232/232, measured but not bound (`offsets_unit: none`). | [`g1-tokenizer-holdout/report.md`](g1-tokenizer-holdout/report.md) |
| G1, G2 | text-encoder release set `multilingual-e5-small/release/onnx-fp32-release` (the same 232 rows, `embeddings.jsonl`, `tokenizer-batch.json`) | **G1 passed, G2 passed**, first divergence none | stage 1 ids and masks exact 232/232; .NET from ids vs reference embedding max abs 1.46e-7 (1e-5), min cosine 1.0; .NET from text 1.46e-7 (1e-5); C# vs Python ORT, same machine, 8.0e-8 (1e-6); Python ORT vs PyTorch hidden state 2.56e-6 (1e-4), embedding 1.51e-7 (1e-5); every padded batch vs single 0 (1e-5) | [`g1-g2-staged/report.md`](g1-g2-staged/report.md) |
| G3 | MIRACL dev, 8 languages, 200 queries, 2,778 passages (`release-shared/retrieval/miracl-dev-hn-8lang-v1`) | **not-run** (no budget bound: dataset digest form, below) | nDCG@10 **0.7978** (minimum 0.792), MRR@10 **0.7717** (minimum 0.765); recall@10 0.9789. Both metrics clear their minimums and equal the PyTorch reference (`reference-scores/e5-st.json`: 0.7978 / 0.7717) to four decimals. | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

The G1 holdout and the staged G1 check the same 232 rows; the holdout run also compares the fields the staged evaluator does not read (tokens, decode, offsets).

## G3: why it is not a gate result

The evaluator's dataset digest for retrieval covers `queries.jsonl`, `corpus.jsonl` and `qrels.jsonl` (`model-pipeline/README.md`, "Task quality"). On this set that digest is `4791590718868341a87ef94f11c3829023386741835947b8305855864c83471e`. The accepted budget (`protected/multilingual-e5-small/onnx-fp32-cpu/task-quality.yaml`) names `614a8d0d4d43a3930cc7ea4fea56a9835c7db108b68b9c071de8e116995d2405`. That is the same function applied to the three data files plus `dataset.json`, because the proposal's scoring script called `dataset_digest(data.files)` on every file the loader returned. The data bytes are the ones the budget was set on: the four files' Git blobs are identical in the proposal commit (`e50cb3d`) and on main. So the evaluator reports "no budget: not a gate result" (exit 4) and G3 stays `not-run`. The same mismatch affects the EmbeddingGemma 2 retrieval budgets and the GLiNER2 NER budgets. The Qwen GSM8K budget uses the data-only form and is not affected. This needs a fix in the evaluator, or a maintainer correction of the budgets; G3 is rerun after it.

## Commands

Scripts outside the repository (`~/.cache/starling-release-gates/scripts/e5-cpu-*.sh`). One model process at a time; 4 ONNX Runtime intra-op threads.

```sh
REL=model-pipeline/protected/multilingual-e5-small/release/onnx-fp32-release
MAN=model-pipeline/registry/multilingual-e5-small/onnx-fp32-cpu/manifest.yaml
TOOL="dotnet ai-dotnet/tools/Starlings.Fidelity.Tool/bin/Release/net10.0/Starlings.Fidelity.Tool.dll"   # Release build at b469654

# G1, tokenizer holdout
$TOOL tokenizer --model e5 --tokenizer <e5 snapshot>/tokenizer.json --fixtures $REL/tokenizer.jsonl --out <run>/g1-holdout
model-pipeline evaluate $MAN --tokenizer-fixtures $REL/tokenizer.jsonl --results <run>/g1-holdout --out <report>

# G1/G2, staged
$TOOL e5 --out <run>/stages --fixtures $REL --hidden --threads 4
<encoder venv>/bin/python model-pipeline/reference/encoder/stage_e5_export.py --model-dir <e5 snapshot with model.safetensors> \
  --fixtures $REL --profile multilingual-e5-small/onnx-fp32-cpu --out <run>/stages --intra-op-threads 4
model-pipeline evaluate $MAN --fixtures $REL --results <run>/stages --out <report> --require-gates G1,G2

# G3, retrieval
DS=model-pipeline/protected/release-shared/retrieval/miracl-dev-hn-8lang-v1
$TOOL retrieval --model e5 --dataset $DS --out <run>/g3/embeddings.jsonl --threads 4
model-pipeline evaluate $MAN --g3 retrieval --dataset $DS --results <run>/g3/embeddings.jsonl --out <report>
```

## Digests and environment

| Input | SHA-256 |
| --- | --- |
| `tokenizer.json` (manifest `processing.tokenizer`) | `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39` |
| `onnx/model.onnx` (manifest `origin.files`, checked by the evaluator) | `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665` |
| release `tokenizer.jsonl`, `embeddings.jsonl` | in each report's protected-evidence table |
| candidate `tokenizer.candidate.jsonl` (outside Git) | `4d73242b90bf50170e4c853f1570ebcc084c452bb481b40359cf570e1a068796` |
| stage results `tokenizer`, `export`, `dotnet` (outside Git) | `86eb82de…`, `85afa223…`, `9f045ada…` (full values in `g1-g2-staged/report.json`) |
| G3 candidate embeddings, 2,978 rows (outside Git) | `fb96e66e28c6e96a47e6462d15011ba98de71081c7d17f093469f61a9ed92c9d` |

Environment: Intel i7-5820K (12 threads, AVX2), Linux x64, .NET 10.0.12, ONNX Runtime 1.30.0 CPU (managed 1.30.0), 4 intra-op threads. Python stage 2 ran in a fresh environment from `encoder/requirements.txt` (Python 3.11.16, torch 2.14.1+cpu, transformers 5.18.0, sentence-transformers 6.1.0, onnxruntime 1.30.0) on the snapshot at `614241f6` with `model.safetensors`. The machine was also running a one-hour EmbeddingGemma 2 soak (`Starlings.Soak.Tool`, about 5 cores) and other services. Wall-clock times below are not benchmarks.

| Step | Wall clock |
| --- | ---: |
| G1 holdout producer (232 rows) | 0.6 s |
| .NET stages 1 and 3 (232 rows, hidden states, batches) | 28.0 s |
| Python stage 2 | 37.3 s |
| G3 producer (200 queries, 2,778 passages) | 406.8 s |
| each evaluator run | under 10 s |
