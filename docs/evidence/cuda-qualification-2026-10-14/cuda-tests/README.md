# CUDA test projects

Each log is the output of `dotnet test --project ai-dotnet/tests/<project> -c Release --output Detailed` on the RTX 3090 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`, `STARLING_CUDA_TESTS=1`, models from the acquisition cache at `STARLING_MODELS`). Hardware and software are listed in [`../README.md`](../README.md).

| Project | Total | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: |
| `Starlings.Models.E5.Tests.Cuda` | 7 | 6 | 0 | 1 |
| `Starlings.Models.Gliner2.Tests.Cuda` | 4 | 4 | 0 | 0 |
| `Starlings.Models.EmbeddingGemma2.Tests.Cuda` (text, image, video) | 18 | 18 | 0 | 0 |
| `Starlings.Models.Qwen35.Tests.Cuda` (fp32x and fp16x) | 75 | 75 | 0 | 0 |

Inputs outside Git, each checked by the tests against protected digests:

- EmbeddingGemma 2 image set and video set (`STARLING_EMBEDDINGGEMMA2_IMAGES`, `_VIDEOS`) and their reference tensors (`_IMAGE_TENSORS`, `_VIDEO_TENSORS`).
- The FFmpeg 8.1.3 LGPL build (`STARLING_FFMPEG_DIR`).
- The Qwen snapshot with both bundles (`STARLING_QWEN35_SNAPSHOT`).

Skipped: `E5.Tests.Cuda.CudaUnavailableTests.MissingCudaProviderLibraryFailsClearly`. It needs `STARLING_EXPECT_CUDA_UNAVAILABLE=1` in an environment where the CUDA provider cannot load.
