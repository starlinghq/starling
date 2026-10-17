## CUDA qualification on RunPod

Commit `6ea1bfa218555d4360a81afb48c4050f46e71824`, GPU NVIDIA GeForce RTX 3090, 580.126.20, wall 106 min.

| Profile | Check | Result |
| --- | --- | --- |
| `all` | models-acquired-and-verified | passed |
| `Starlings.Models.E5.Tests.Cuda` | tests (total: 10, failed: 0, succeeded: 9, skipped: 1) | passed |
| `Starlings.Models.Gliner2.Tests.Cuda` | tests (total: 8, failed: 0, succeeded: 8, skipped: 0) | passed |
| `Starlings.Models.EmbeddingGemma2.Tests.Cuda` | tests (total: 25, failed: 0, succeeded: 18, skipped: 7) | passed |
| `Starlings.Models.Qwen35.Tests.Cuda` | tests (total: 106, failed: 0, succeeded: 106, skipped: 0) | passed |
| `multilingual-e5-small/onnx-fp32-cuda` | g1-tokenizer-holdout G1 | passed |
| `multilingual-e5-small/onnx-fp32-cuda` | g1-g2-staged G1,G2 | passed |
| `multilingual-e5-small/onnx-fp32-cuda` | g3-retrieval G3 | passed |
| `gliner2-base-v1/onnx-fp32-cuda` | g1-tokenizer-release G1 | passed |
| `gliner2-base-v1/onnx-fp32-cuda` | g1-g2-extraction-release G1,G2 | passed |
| `gliner2-base-v1/onnx-fp32-cuda` | g3-ner G3 | passed |
| `embeddinggemma-2/onnx-fp32-cuda` | g1-tokenizer-holdout G1 | passed |
| `embeddinggemma-2/onnx-fp32-cuda` | g2-release G1,G2 | passed |
| `embeddinggemma-2/onnx-fp32-cuda` | g3-retrieval G3 | passed |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | g1-tokenizer-release G1 | passed |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | g1-g2-release-prompts G1,G2 | passed |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | g3-gsm8k G3 | passed |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | g1-tokenizer-release G1 | passed |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | g1-g2-release-prompts G1,G2 | passed |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | g3-gsm8k G3 | passed |
| `embeddinggemma-2/image-fp32-cuda` | g1-g2-image-release G1,G2 | passed |
| `embeddinggemma-2/image-fp32-cuda` | g3-retrieval G3 | passed |
| `embeddinggemma-2/video-fp32-cuda` | g1-g2-video-release G1,G2 | passed |
| `embeddinggemma-2/video-fp32-cuda` | g3-retrieval G3 | passed |
| `multilingual-e5-small/onnx-fp32-cuda` | soak 10m smoke | passed |
| `gliner2-base-v1/onnx-fp32-cuda` | soak 10m smoke MALLOC_MMAP_THRESHOLD_=65536 | passed |
| `embeddinggemma-2/onnx-fp32-cuda` | soak 10m smoke | passed |
| `embeddinggemma-2/image-fp32-cuda` | soak 10m smoke | passed |
| `embeddinggemma-2/video-fp32-cuda` | soak 10m smoke MALLOC_MMAP_THRESHOLD_=65536 | passed |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | soak 10m smoke | passed |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | soak 10m smoke | passed |
