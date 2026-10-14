# W1: CUDA frame-call size, measurement protocol

Status: **protocol and script only, not run** (2026-10-13, R7). The run uses the shared GPU and is the maintainer's (GPU allowance). The package default stays 2 frames per vision call; D7 decides on these numbers ("keep 2 unless 4 fits the shared GPU with headroom").

## What is measured

For 2, 3 and 4 frames per vision call (`EmbeddingGemma2MultimodalOptions.MaxVideoFramesPerVisionCall`, with `MaxVisionBatchPatches` = frames x 1,260), one run each of the CUDA video parity test on GPU 1:

| Quantity | Source |
| --- | --- |
| Embedding max abs and min cosine (35 default-contract rows vs the CPU PyTorch reference) | `CudaVideoParityTests.RowsMatchTheReference` output; the bounds are the protected `video-fp32-cuda` tolerances, so a run over the bound fails and is recorded as failed |
| Peak GPU memory of the test process | `nvidia-smi --query-compute-apps=pid,used_memory`, sampled every 200 ms |
| What the other workload leaves | GPU memory used before each run (other processes, listed) and the headroom at the run's peak (total minus the GPU's peak used memory) |
| Wall clock | The test process, start to exit (session creation, decoding and 35 rows) |
| Frames per call actually used | `SessionsUseCudaWithTf32Disabled` output line `frames per vision call: N` |

## How to run

Prerequisites, as for any CUDA test here (`ai-dotnet/README.md`): the CUDA 13 runtime and cuDNN 9 on the loader path; the image graphs (`STARLING_MODELS` or `STARLING_EMBEDDINGGEMMA2_SNAPSHOT`); the video set (`STARLING_EMBEDDINGGEMMA2_VIDEOS`, from `make_eg2_videos.py`); FFmpeg 8 (`STARLING_FFMPEG_DIR`, `fetch_eg2_ffmpeg.sh`); optionally the reference tensors (`STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS`) for the soft-token lines.

```sh
tools/measure-eg2-cuda-frame-calls.py --gpu 1 --out docs/evidence/embeddinggemma2-video/frame-calls
```

The script builds `ai-dotnet/tests/Starlings.Models.EmbeddingGemma2.Tests.Cuda` (Release), then for each size sets `STARLING_CUDA_TESTS=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1` and `STARLING_EG2_CUDA_FRAMES_PER_CALL=<n>` (read only by the CUDA test fixture), waits 5 s, samples the baseline, runs the two tests and samples the GPU while they run. Sizes run one after another, never in parallel. It writes `frame-calls.json` (every sample, the parsed test output, GPU name, driver, commit), `frame-calls.md` (the table below) and one log per size.

Run when the other GPU workload is in its usual state, and note it in the record: the headroom depends on it. If a size fails with an out-of-memory error, the table records the failed run; do not retry with other settings in the same record.

## Record (to fill from `frame-calls.md`)

| Frames per call | Embedding max abs | Min cosine | Process peak GPU memory (MiB) | GPU used before run, other workload (MiB) | Headroom at peak (MiB) | Wall clock (s) | Tests |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | not run | | | | | | |
| 3 | not run | | | | | | |
| 4 | not run | | | | | | |

Known points of comparison: frame calls of 2 gave embedding max abs 3.27e-6 and min cosine 0.99999999968 on CUDA (`README.md` in this directory); the spike's C# prototype with calls of 4 gave 4.5e-7; the batch test with calls of 4 gave 8.9e-8 against single rows.
