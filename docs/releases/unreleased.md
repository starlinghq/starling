# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## EmbeddingGemma 2 image and video CUDA: concurrent-call tests (#257)

New CUDA tests run overlapping `EmbedBatch` calls on one `EmbeddingGemma2MultimodalEmbedder` for `image-fp32-cuda` (4 threads by default, `STARLING_EG2_CUDA_IMAGE_CONCURRENT_THREADS` to change it) and `video-fp32-cuda` (8 threads), and compare every vector with the sequential result byte for byte. Both passed on an RTX 4060 Ti (experimental evidence; the RTX 3090 rerun is open in #257). `concurrent_calls` stays supported. The package README now warns that 8 overlapping image calls on one embedder ran out of device memory on a 16 GB GPU (`OnnxRuntimeException`, not a wrong result) until memory-aware limits land (#172). Evidence: `docs/evidence/cuda-concurrency-2026-11-01/`.
