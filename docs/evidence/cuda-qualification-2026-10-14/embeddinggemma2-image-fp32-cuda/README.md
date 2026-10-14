# embeddinggemma-2/image-fp32-cuda

Evaluator exit 0, first divergence none, **G1 passed, G2 passed** on development fixtures (PR #42 image set, 42 rows, 35 images). This is not the P-030 release set. Hardware and commands are in [`../README.md`](../README.md).

| Stage | Producer | Result |
| --- | --- | --- |
| 1 rendering and preprocessing | `Starlings.Fidelity.Tool.Cuda -- embeddinggemma2-images --device cuda` at `2db3dce` (SkiaSharp decoder) | ids exact on 42/42; decoded RGB, resized pixels, pixel values and positions exact on 35/35 images. `mode-gray16` (16-bit) is excluded by contract. |
| 2 export | not applicable | The image tolerances bind the C# path, as on CPU. |
| 3 .NET from decoded RGB | same C# tool, ORT 1.30.0 CUDA (vision and multimodal text graphs) | embedding 3.64e-6 (1e-5); min cosine 0.99999999965 (0.9999995); soft tokens 1.44e-4 row-relative (1e-3, worst `photo-camera/image0`); token embeddings 7.7e-5 (1e-3); vision batch of 8 and padded batch of 3 against single rows 7.0e-7 and 4.3e-7 (1e-5) |

The reference tensors are the spike's `ref-clean-tensors` `.npz` files, which are outside Git. The evaluator checked them against the protected `tensors.json`. All 15 bound checks passed. Compared with CPU (`docs/evidence/embeddinggemma2-images/`: embedding 2.24e-6, soft tokens 1.57e-4), CUDA is within the same order.

Files: `report.md`, `report.json` (unedited), `dotnet.log`.
