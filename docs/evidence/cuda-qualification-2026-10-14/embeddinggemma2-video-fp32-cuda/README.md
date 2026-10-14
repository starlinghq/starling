# embeddinggemma-2/video-fp32-cuda

Evaluator exit 0, first divergence none, **G1 passed, G2 passed** on development fixtures (PR #47 video set, 36 rows, 32 videos). This is not the P-030 release set. Hardware and commands are in [`../README.md`](../README.md).

FFmpeg: the pinned LGPL shared build `n8.1.3-17-gc389840894` (FFmpeg 8.1, from `fetch_eg2_ffmpeg.sh`), passed with `--ffmpeg`. Stage 1 records `ffmpeg_major` 8, which equals `decoder_ffmpeg.supported_ffmpeg_major`, so the per-frame decoder check is gated. Decoding is on the CPU; only the model runs on CUDA.

| Stage | Producer | Result |
| --- | --- | --- |
| 1 decoding, sampling, rendering, preprocessing | `Starlings.Fidelity.Tool.Cuda -- embeddinggemma2-videos --device cuda` at `2db3dce` | ids exact on 36/36. Frame indices, per-frame RGB (against torchcodec), stacked frames, resized frames, pixel values and positions exact on 32/32 videos. The still image's RGB is exact. `vid-array-no-metadata` is excluded by contract. |
| 2 export | not applicable | |
| 3 .NET from decoded frames | same C# tool, ORT 1.30.0 CUDA | embedding 1.13e-6 (1e-5); min cosine 0.99999999996 (0.9999995); soft tokens 6.1e-5 row-relative (1e-3); token embeddings 1.09e-4 (1e-3); batches (2 frames per call with a text batch of 4, and a padded batch of 3) against single rows 1.8e-7 and 1.95e-7 (1e-5) |

The reference tensors are the spike's video `ref-clean-tensors`, checked against the protected `tensors.json`. All 18 bound checks passed.

Files: `report.md`, `report.json` (unedited), `dotnet.log`.
