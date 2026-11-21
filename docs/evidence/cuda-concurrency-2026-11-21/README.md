# Concurrent CUDA EmbedBatch calls on the RTX 3090, 2026-11-21

Issue #257, items 1 and 2: the rerun of the four CUDA concurrency tests (e5 and EmbeddingGemma 2 text from #254; EmbeddingGemma 2 image and video from the 2026-11-01 run) on the RTX 3090 that the CUDA contract covers. The 2026-11-01 run used the RTX 4060 Ti because the 3090 was occupied by another workload; the 3090 was idle for this run.

## Result

All four tests passed, 8 separate processes each, every vector bit-for-bit equal to the sequential run:

| Profile | Test | Per process | Processes | Vectors compared | Max abs |
| --- | --- | --- | --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-cuda` | `E5.Tests.Cuda` `CudaParityTests.ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit` | 8 threads x 8 inputs x 5 rounds | 8 of 8 passed | 2,560 | 0 |
| `embeddinggemma-2/onnx-fp32-cuda` | `EmbeddingGemma2.Tests.Cuda` `CudaParityTests.ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit` | 8 threads x 4 rows x 5 rounds | 8 of 8 passed | 1,280 | 0 |
| `embeddinggemma-2/image-fp32-cuda` | `CudaImageParityTests.ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit` | 8 threads x 4 rows x 5 rounds | 8 of 8 passed | 640 | 0 |
| `embeddinggemma-2/video-fp32-cuda` | `CudaVideoParityTests.ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit` | 8 threads x 2 rows x 5 rounds | 8 of 8 passed | 640 | 0 |

Logs: [e5](logs/e5-cuda-concurrency-8x.log), [text](logs/eg2-text-cuda-concurrency-8x.log), [image](logs/eg2-image-cuda-concurrency-8x.log), [video](logs/eg2-video-cuda-concurrency-8x.log). Each process ran the provider-check method (ORT version, providers, `use_tf32`, `CUDA_VISIBLE_DEVICES`) and the concurrency test. The first line of the e5 log records the GPU, its memory use and the driver at the start.

**The image test ran with 8 threads, not 4.** The 2026-11-01 run used 4 threads because 8 overlapping image calls exhausted the 16 GB 4060 Ti's CUDA arena (allocation failures, logged there). The issue proposed trying 8 on the 24 GB 3090, via `STARLING_EG2_CUDA_IMAGE_CONCURRENT_THREADS=8`; it fit and passed 8 of 8. The `concurrent_calls: supported` status of `image-fp32-cuda` is unchanged, and 8-thread concurrency on the 3090 is now exercised evidence rather than an untried claim. On a 16 GB GPU 8 overlapping image calls still fail with `OnnxRuntimeException` (allocation failure), as the 2026-11-01 logs show; that remains caller-visible behavior of the 16 GB card, unchanged by this run.

**No status change.** `concurrent_calls` stays supported on all four CUDA profiles, and neither encoder nor the multimodal embedder serializes CUDA runs.

## Environment

- The maintainer's workstation: Intel Core i7-5820K (6 cores, 12 threads, AVX2/FMA), 94 GiB RAM, Omarchy, kernel 7.2.8-9-omarchy, .NET runtime 10.0.14, Release build.
- NVIDIA GeForce RTX 3090 (Ampere, the contract GPU), driver 615.78.08, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`. The GPU was idle (0.3 GiB of 24 GiB in use before the runs); no soak or other job ran alongside.
- CUDA 13 runtime from `/opt/cuda`; cuDNN 9 from the `nvidia-cudnn-cu13` wheel in the model-pipeline reference venv.
- ONNX Runtime 1.30.0 (`Microsoft.ML.OnnxRuntime.Gpu.Linux`).
- Models from the Starlings model cache (`$STARLING_MODELS`), SHA-256 enforced by the fixtures and the embedder. The image set is the protected fixture's 29 images (SHA-256 checked by the test assets), the video set the protected fixture's 22 videos; the video decoder used the pinned FFmpeg 8 LGPL shared build (`n8.1.3-17-gc389840894`, from `Starlinghq/ffmpeg-lgpl-rebuilds` via `model-pipeline/reference/embeddinggemma2/fetch_eg2_ffmpeg.sh`; the decoder checks `libavutil` major 60).

## Relation to the 2026-11-01 run

Same tests, same fixture rows, same bit-for-bit byte comparison (`MemoryMarshal.AsBytes(...).SequenceEqual`); only the GPU differs (the contract's RTX 3090 instead of the RTX 4060 Ti) and the image test's thread count (8 instead of 4). Both runs together cover the contract GPU and a consumer 16 GB card. With this run, issue #257's items 1 and 2 are done; nothing else was open on the issue.
