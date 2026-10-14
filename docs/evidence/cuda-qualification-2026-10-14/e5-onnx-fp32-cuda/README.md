# multilingual-e5-small/onnx-fp32-cuda

Evaluator exit 0, first divergence none, **G1 passed, G2 passed** on development fixtures (M0 corpus, 46 cases). This is not the P-030 release set. Hardware, environment and the full set of commands are in [`../README.md`](../README.md).

| Stage | Producer | Result |
| --- | --- | --- |
| 1 tokenizer | `Starlings.Fidelity.Tool.Cuda -- e5 --hidden --device cuda` at `2db3dce` | ids and attention masks exact on 46/46 |
| 2 export | `stage_e5_export.py --provider cuda` (onnxruntime-gpu 1.30.0, `use_tf32=0`). The reference hidden states come from PyTorch fp32 on CPU. | hidden state 2.27e-6 (bound 1e-4); embedding 1.70e-7 (1e-5); padded batches against single rows 1.23e-7 (1e-5) |
| 3a .NET from ids | same C# tool, ORT 1.30.0 CUDA | embedding 1.65e-7 (1e-5); C# against Python ORT on the same GPU 9.0e-8 (1e-6); C# and Python hidden states bit-identical (report only) |
| 3b .NET from text | same | embedding 1.65e-7 (1e-5); text batch against single rows 1.05e-7 (1e-5) |

Minimum cosine against the reference was 0.9999999999995 (report only). All 29 bound checks passed. The 2 not-applicable checks are offsets (`offsets_unit: none`) and C# hidden states, which have no protected bound.

The earlier proposed Python-only figures for this profile, measured on an RTX 4060 Ti (`protected/multilingual-e5-small/onnx-fp32-cuda/PROPOSED.md`), were hidden state 2.1e-6 and embedding 1.9e-7. This run is the first evaluator run of the C# path on CUDA for this profile.

Files: `report.md`, `report.json` (unedited evaluator output), `export.log`, `dotnet.log`.
