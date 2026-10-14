# gliner2-base-v1 `onnx-fp32-cpu` on the P-030 release sets

Run on 2026-10-14 at `10c016c` (main after #26). The evaluator verified the protected evidence against Git, and the binding tolerances and the budget come from `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/`. Nothing under `protected/` was changed.

| Gate | Set | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- | --- |
| G1 | tokenizer release `gliner2-base-v1/release/tokenizer-release/gliner2-tokenizer.jsonl` (232 holdout rows) | **passed** | `Gliner2Preprocessor` against the reference whitespace splitter, exact on 232/232: normalized text, words, word character offsets (Unicode scalars), input ids, word and schema positions. Schema tokens, also exact, are measured only. | [`g1-tokenizer-release/report.md`](g1-tokenizer-release/report.md) |
| G1, G2 | extraction release set `gliner2-base-v1/release/extraction-release` (30 WNUT-17 sentences, 6 labels, threshold 0.5) | **G1 passed, G2 passed**, first divergence none | processor outputs exact on 30/30. .NET: span logits relative 1.78e-5 (1e-4), count logits 4.0e-5 (5e-4), scores 2.91e-6 (2e-5), entities 30/30 with 0 exceptions, confidence 2.88e-6 (2e-5); text path entities 30/30, confidence 2.88e-6. C# vs Python ORT span logits 0 (1e-6). Python ORT vs PyTorch hidden state 2.48e-5 (1e-4). Every padded batch vs single: 0. | [`g1-g2-extraction-release/report.md`](g1-g2-extraction-release/report.md) |
| G3 | WNUT-17 test, 300 sentences (`gliner2-base-v1/release/ner/wnut17-test-v1/gold.jsonl`) | **passed** | exact micro F1 **0.3626** (minimum 0.352), partial micro F1 **0.4134** (minimum 0.403). Both equal the GLiNER2 PyTorch reference (0.3626 / 0.4134). The budget's digest is the data+metadata form (#26). | [`g3-ner/report.md`](g3-ner/report.md) |

## Commands

Scripts outside the repository (`~/.cache/starling-release-gates/scripts/g2-cpu*.sh`). One model process at a time, 4 threads.

```sh
TOK=model-pipeline/protected/gliner2-base-v1/release/tokenizer-release/gliner2-tokenizer.jsonl
REL=model-pipeline/protected/gliner2-base-v1/release/extraction-release
MAN=model-pipeline/registry/gliner2-base-v1/onnx-fp32-cpu/manifest.yaml
TOOL="dotnet ai-dotnet/tools/Starlings.Fidelity.Tool/bin/Release/net10.0/Starlings.Fidelity.Tool.dll"

$TOOL tokenizer --model gliner2 --tokenizer <snapshot>/tokenizer.json --fixtures $TOK --out <run>/g1-tokenizer
model-pipeline evaluate $MAN --tokenizer-fixtures $TOK --results <run>/g1-tokenizer --out <report>

$TOOL gliner2 --out <run>/stages --fixtures $REL --threads 4
<gliner2 venv>/bin/python model-pipeline/reference/gliner2/stage_gliner2.py export --model-dir <snapshot f9634218 with model.safetensors> \
  --graph <snapshot>/onnx/gliner2-entities-fp32.onnx --graph-asset-prefix onnx/gliner2-entities-fp32 \
  --fixtures $REL --profile gliner2-base-v1/onnx-fp32-cpu --out <run>/stages --threads 4
model-pipeline evaluate $MAN --fixtures $REL --results <run>/stages --out <report> --require-gates G1,G2

DS=model-pipeline/protected/gliner2-base-v1/release/ner/wnut17-test-v1/gold.jsonl
$TOOL ner --dataset $DS --out <run>/g3/predictions.jsonl --threads 4
model-pipeline evaluate $MAN --g3 ner --dataset $DS --results <run>/g3/predictions.jsonl --out <report>
```

## Digests and environment

| Input | SHA-256 |
| --- | --- |
| `tokenizer.json` | `1b7fbabfb4c690bed84c6793bfecae9b8dfe205751b04f9ffd1e76a1e7df9c16` |
| `onnx/gliner2-entities-fp32.onnx` (manifest `origin.files`, checked by the evaluator) | in `g1-g2-extraction-release/report.json` |
| G1 candidate `tokenizer.candidate.jsonl` (outside Git) | `e397015a294ba96f52bd81e8993cf034f36bac833f8f2001d44b5b7cb6d14084` |
| stage results `tokenizer`, `export`, `dotnet` (outside Git) | `75cc65e0…`, `0c76415c…`, `64a3f9f8…` |
| G3 predictions, 300 rows (outside Git) | `650757a8b76841bc63c8815cddcba0143cf316584680bdc4cbecb5f5bb52141e` |
| WNUT-17 dataset digest (data / data+metadata) | in `g3-ner/report.json` (`dataset.sha256`, `sha256_with_metadata`) |

Environment: Intel i7-5820K, Linux x64, .NET 10.0.12, ONNX Runtime 1.30.0 CPU, 4 intra-op threads. Python stage 2 used a fresh environment from `gliner2/requirements-export.txt` (Python 3.11, torch 2.14.1+cpu, transformers 4.57.6, tokenizers 0.22.2, onnxruntime 1.30.0, gliner2 2.0.0 at `55656fbf`). The machine also ran the G4 soak measurements, so the wall-clock times below are not benchmarks.

| Step | Wall clock |
| --- | ---: |
| G1 tokenizer producer (232 rows) | 1.3 s |
| .NET stages 1 and 3 (30 rows, batches) | 28.0 s |
| Python stage 2 | 54.6 s |
| G3 producer (300 sentences) | 42.5 s |
