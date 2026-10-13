# Staged fidelity evaluation: multilingual-e5-small/onnx-fp32-cpu (P-007)

`report.json` and `report.md` are the unedited output of `model-pipeline evaluate` on real model output, 2026-10-09. Result: first divergence none, G1 passed, G2 passed, protected evidence verified against `origin/main` at `559dd440`. Exit code 0 with `--require-gates G1,G2`.

This is development evidence from one workstation, on the agent-written development fixtures (not the protected release set, P-030). The registry manifest's gates are not changed by this pull request; copying G1 and G2 into it is a maintainer decision.

## What ran

| Stage | Producer | Environment |
| --- | --- | --- |
| 1 tokenizer | `ai-dotnet/tools/Starlings.Fidelity.Tool` at `e2ede1b7` (`E5Tokenizer`, owned tokenizer) | .NET 10.0.12 |
| 2 export | `model-pipeline/reference/encoder/stage_e5_export.py` at `e2ede1b7` | Python 3.11.16 from `reference/encoder/requirements.txt`: onnxruntime 1.30.0 (CPU EP, default threads), torch 2.14.1+cpu, sentence-transformers 6.1.0, transformers 5.18.0 |
| 3 dotnet | `Starlings.Fidelity.Tool --hidden` at `e2ede1b7` (`E5Encoder`, `E5TextEmbedder`) | .NET 10.0.12, ONNX Runtime 1.30.0 CPU, default threads |

All on one Intel i7-5820K (AVX2, Linux x64), model snapshot `intfloat/multilingual-e5-small@614241f6`, digests checked against the manifest. Evaluator at `c8f2af70` (re-run on the same stage results after the review fixes, which added per-output padded-batch evidence checks).

## Numbers (46 cases, absolute error)

| Comparison | Bound (protected) | Max | Mean | p99 |
| --- | ---: | ---: | ---: | ---: |
| Tokenizer ids and masks vs `tokenizer.jsonl` | exact | 46/46 exact | | |
| Python ORT vs PyTorch, last hidden state | 1e-4 | 2.89e-6 | 2.0e-7 | 7.6e-7 |
| Python ORT vs reference embeddings | 1e-5 | 2.30e-7 | 1.99e-8 | 7.4e-8 |
| C# from ids vs reference embeddings | 1e-5 | 2.32e-7 | 1.95e-8 | 7.14e-8 |
| C# vs Python ORT embeddings, same machine | 1e-6 | 1.0e-7 | 2.97e-9 | 2.0e-8 |
| C# from text vs reference embeddings | 1e-5 | 2.32e-7 | 1.95e-8 | 7.14e-8 |
| Padded batches vs own single rows (Python and C#, 3 batch shapes; each output path has a batch that pads) | 1e-5 | 0 | 0 | 0 |
| C# vs Python ORT, last hidden state (report only) | none | 0 | 0 | 0 |

C# hidden states are bit-identical to Python ONNX Runtime in 46 of 46 cases, so the remaining 1e-7 between C# and Python embeddings is pooling arithmetic (C# accumulates in double, numpy in float32). Offsets are `not-applicable`: the manifest declares `offsets_unit: none` for the embedding task.

## Stage attribution on the same real outputs

Two faults injected into copies of these results (not committed; the commands are a few lines of Python over the JSON):

| Injection | Evaluator verdict |
| --- | --- |
| C# embeddings scaled by 1 + 1e-5 (a .NET pooling-style fault) | First divergence `dotnet-ids`, ".NET runtime binding, preprocessing or pooling". C# vs reference still passes (max 1.57e-6 under 1e-5); the same-machine bound against Python ORT catches it (1.66e-6 over 1e-6). |
| Python ORT and C# embeddings both scaled by 1 + 1e-3 (a fault in the shared artifact) | First divergence `export`, "export/operator or precision"; the C# failures are reported as consequences of the export, because C# still agrees with Python ORT within 1e-7. |

The synthetic tests (`model-pipeline/tests/test_evaluate.py`, `test_protection.py`) cover every attribution case and the tamper attempts.

## Reproduce

See `model-pipeline/reference/README.md`, "Staged fidelity evaluation of the e5 CPU profile". About 25 s for stage 2 and 10 s for the C# tool on this machine; the evaluator takes about 1 s.
