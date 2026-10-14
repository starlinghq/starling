# gliner2-base-v1/onnx-fp32-cuda

Evaluator exit 0, first divergence none, **G1 passed, G2 passed** on development fixtures (PR #19 extraction corpus, 58 cases). This is not the P-030 release set. Hardware and commands are in [`../README.md`](../README.md).

| Stage | Producer | Result |
| --- | --- | --- |
| 1 processor | `Starlings.Fidelity.Tool.Cuda -- gliner2 --threads 1 --device cuda` at `2db3dce` (the owned C# processor) | normalized text, words, character maps, ids, mask, word and schema positions exact on 58/58 |
| 2 export | `stage_gliner2.py export --provider cuda --threads 1` on the hosted graph `onnx/gliner2-entities-fp32.onnx` (onnxruntime-gpu 1.30.0, `use_tf32=0`). The reference hidden states come from PyTorch fp32 on CPU. | count logits 4.9e-5 (5e-4); span logits 1.42e-5 relative (1e-4); scores 2.26e-6 (2e-5); confidence 1.13e-6 (2e-5); hidden 2.48e-5 (1e-4); gate decisions and entities exact; padded batches 1.59e-5 relative (1e-4) |
| 3 .NET from inputs | same C# tool, ORT 1.30.0 CUDA | count logits 4.9e-5; span logits 1.42e-5 relative; scores 2.26e-6; confidence 1.11e-6; entities exact with no listed exception; C# against Python span logits 0 (1e-6) |
| 4 .NET from text | same | entities exact on 58/58; confidence 1.11e-6 (2e-5) |

All 77 checks passed. 4 cases are outside the public C# contract (the max_len word cap, and inputs over 512 tokens). They ran through internal paths, as on CPU, and are listed in `report.json` under `cases_outside_contract`.

Files: `report.md`, `report.json` (unedited), `export.log`, `dotnet.log`.
