# Staged fidelity evaluation, entity extraction: gliner2-base-v1/onnx-fp32-cpu, stages 1 to 4 (P-015)

`report.md` and `report.json` are the unedited output of

```sh
model-pipeline evaluate model-pipeline/registry/gliner2-base-v1/onnx-fp32-cpu/manifest.yaml --results <dir> --out <dir> --require-gates G1,G2
```

run on 2026-10-10 in this repository at `6cf3b95` (protected evidence verified against Git), with the binding tolerances (`model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/tolerances.yaml`, PR #21) and the protected development fixtures (`extraction.jsonl`, 58 agent-written cases, PR #19; not the P-030 release set). Result: exit 0, first divergence none, **G1 passed, G2 passed**, 0 failed checks. One workstation (Intel i7-5820K, Linux x64, ORT 1.30.0 CPU, 1 thread); development evidence, the manifest's gate entries are left to the maintainer.

## Stage results

| Stage | Producer |
| --- | --- |
| 1 processor | `ai-dotnet/tools/Starlings.Fidelity.Tool -- gliner2 --threads 1` at `6cf3b95`: the owned C# processor, every case (the two `max_len` cases through the internal word-cap path) |
| 2 export | `model-pipeline/reference/gliner2/stage_gliner2.py export --graph <release>/onnx/gliner2-entities-fp32.onnx --graph-asset-prefix onnx/gliner2-entities-fp32 --threads 1` (P-014 `venv-cpu`, `PYTHONPATH=model-pipeline/src`): Python ORT on the hosted single-file graph whose digest `origin.files` pins |
| 3, 4 .NET | the same tool run: stage 3 from the fixture inputs, stage 4 (`text_entities`) end to end from text. The four cases outside the contract (two over 512 tokens, two with the `max_len` word cap) go through internal paths of the package and are listed in `environment.cases_outside_contract` |

## Numbers (58 cases; bounds from the protected file)

| Check | Bound | Stage 2 (Python ORT) | Stage 3 (C# from fixture inputs) |
| --- | ---: | ---: | ---: |
| Processor outputs (normalized text, words, maps, ids, mask, positions) | exact | | stage 1: 58/58 exact |
| Count logits max abs | 5e-4 | 4.8e-5 | 4.8e-5 |
| Gate decision | exact | 58/58 | 58/58 |
| Span logits (valid spans), relative | 1e-4 | 2.4e-5 (max abs 7.76e-3) | 2.4e-5 (max abs 7.76e-3) |
| Scores max abs | 2e-5 | 3.6e-6 | 3.6e-6 |
| Entities (identity with swap and flip exceptions of 2e-5) | | 58/58, 0 exceptions | 58/58, 0 exceptions |
| Confidence of matched entities max abs | 2e-5 | 1.82e-6 | 1.81e-6 |
| C# vs Python ORT span logits, same machine and provider, relative | 1e-6 | | 0 (bit-identical) |
| Padded batches vs own singles | 1e-4 relative | 0 | 0 (span logits); text entities identical |
| Stage 4: entities from text | | | 58/58, 0 exceptions, confidence 1.81e-6 |

Reproduce: `ai-dotnet/README.md` ("GLiNER2 entity extraction") for the C# stages, `model-pipeline/reference/README.md` for stage 2, then the command above.
