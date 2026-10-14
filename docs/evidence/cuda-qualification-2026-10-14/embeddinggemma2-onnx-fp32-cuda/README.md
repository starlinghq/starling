# embeddinggemma-2/onnx-fp32-cuda

Evaluator exit 0, first divergence none, **G1 passed, G2 passed** on development fixtures (PR #29 corpus, 138 rows of 46 texts × 3 prompts). This is not the P-030 release set. Hardware and commands are in [`../README.md`](../README.md).

| Stage | Producer | Result |
| --- | --- | --- |
| 1 tokenizer | `Starlings.Fidelity.Tool.Cuda -- embeddinggemma2 --device cuda` at `2db3dce` | ids exact on 138/138 |
| 2 export | `stage_embeddinggemma2_export.py --provider cuda` on the release text graph (onnxruntime-gpu 1.30.0, `use_tf32=0`). The reference token embeddings come from `eg2_text.TextPath`, fp32 on CPU. | embedding 5.1e-7 (5e-6); min cosine 0.999999999993 (0.9999999); token embeddings 7.2e-6 relative per case (1e-4); padded batches against single rows 3.4e-7 (5e-6) |
| 3 .NET from ids and text | same C# tool, ORT 1.30.0 CUDA | embedding 4.71e-7 (5e-6), and 5.31e-7 in batches of 8; min cosine 0.999999999992; C# against Python on the same GPU 4.6e-7 (1e-6); prompt text batches (none, query, document) against single rows up to 2.7e-7 |

All 43 bound checks passed. The 2 not-applicable checks are offsets and hidden states, which have no protected bound.

Files: `report.md`, `report.json` (unedited), `export.log`, `dotnet.log`.
