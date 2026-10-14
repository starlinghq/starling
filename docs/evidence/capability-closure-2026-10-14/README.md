# Capability closure, 2026-10-14

The manifest validator's Stable rule refuses any capability marked `untested` (`model_pipeline.manifest.stable_blockers`). On `main` at `697c978`, 39 capability entries in 12 of the 13 profiles were `untested`. This change closes every one of them in two ways:

- **(a) Evidence, then `experimental`.** The capability was compared with the pinned upstream reference, or with the profile's existing protected evidence, and judged with that profile's protected tolerances.
- **(b) A narrower claim.** Where evidence showed a limit, the capability was split. The part with evidence is `experimental`. The rest is `unsupported`, and the code rejects it explicitly with a typed exception.

No capability was set to `supported`. No gate or maturity changed, and nothing under `model-pipeline/protected/` or in any `tolerances.*` file changed.

**Where new reference data was needed, it is proposed, not accepted.** Each set is under `model-pipeline/proposed/` with a `PROPOSED.md`, produced by the pinned upstream Python reference, never by the C# code. The `experimental` statuses that rest on them say so in the manifest comments. They are judged with the *protected* bounds of each profile, but the rows themselves await the maintainer:

- `embeddinggemma-2/task-prompts`, `image-extra` and `video-edge`
- `gliner2-base-v1/capability-cases`
- the earlier `embeddinggemma-2/long-input` (R7 W3)
- the earlier `qwen3.5-0.8b/long-context` (R8)

Hardware: Intel i7-5820K (12 threads), RTX 3090 (GPU 0 only, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`), driver 610.57.04, CUDA 13 runtime and cuDNN 9 from the `nvidia-*` wheels. Software: .NET SDK 10.0.401 and runtime 10.0.12, ONNX Runtime 1.30.0, ONNX Runtime GenAI 0.17.1. TF32 was off on every CUDA session (`use_tf32=0`, forced by `Starlings.Runtime.Onnx` and `GenAIModelOptions`). The Python references used the pinned environments: `embeddinggemma2/requirements-video.txt` (torch 2.14.1 CPU, transformers 5.19.0, sentence-transformers 6.1.0, torchcodec 0.17.0, Pillow 12.3.0) and `gliner2/requirements.txt` (gliner2 at `55656fbf`, transformers 4.57.6). One machine; this is not qualification.

## Before and after

| Capability | Profile | Before | After | Evidence or reason |
| --- | --- | --- | --- | --- |
| `aot` | e5 `onnx-fp32-cuda` | untested | experimental | `tools/aot-check-ai.sh --cuda` (new): Native AOT against `Microsoft.ML.OnnxRuntime.Gpu.Linux`, 0 trim/AOT warnings, AOT on CUDA equal to JIT on CUDA, 12 vectors max abs 0 ([aot-cuda/](aot-cuda/)) |
| `extensions_ai` | e5 `onnx-fp32-cuda` | untested | experimental | `E5.Tests.Cuda` `CudaAdapterTests`: generator over a CUDA session, 46 protected texts, max abs 1.66e-7 (bound 1e-5) ([log](logs/e5-cuda-tests.log)) |
| `mlnet` | e5 `onnx-fp32-cuda` | untested | experimental | Same test class: transform row-wise and in look-ahead batches of 16, max abs 1.66e-7 |
| `label_descriptions` | gliner2 CPU, CUDA | untested | experimental | 11 proposed public-text cases with descriptions, run by the pinned GLiNER2. C# inputs, gate and entities exact on 22/22 cases; count logits 4.2e-5, span-logit relative 1.7e-5 (CPU) and 2.5e-5 (CUDA) ([CPU](logs/gliner2-capability-cases-cpu.log), [CUDA](logs/gliner2-cuda-tests.log)) |
| `count_gate_closed` | gliner2 CPU, CUDA | untested | experimental | 12 proposed cases where the reference's count gate closes. Margins 1.1 to 8.1 logits, from a scan of 230 public paragraphs x 8 label sets in which 586 of 1,840 gates closed. C# gate closed and no entities on all 12 |
| `cancellation` | gliner2 `onnx-fp32-cuda` | untested | experimental | `CudaParityTests`: a batch of 4,096 texts cancelled after 50 ms threw at 53 ms; the extractor then gave the reference entities |
| `concurrent_calls` | gliner2 `onnx-fp32-cuda` | untested | experimental (serialized); new `concurrent_calls_unserialized`: unsupported | **Defect found.** Four threads on one CUDA session returned wrong entities in 6 of 8 runs ([before](logs/gliner2-cuda-tests-before-serialization.log)). CPU passed 4 of 4 and e5 and EmbeddingGemma 2 on CUDA passed 8 of 8. `Gliner2EntityExtractor` now serializes graph calls on CUDA by default (`Gliner2ExtractorOptions.SerializeNativeCalls`, a cancellable `SemaphoreSlim`), as `Starlings.Models.Qwen35` already does. Serialized: 8 of 8 runs equal to the sequential results ([log](logs/gliner2-cuda-concurrency-serialized-8x.log)) |
| `aot` | gliner2 `onnx-fp32-cuda` | untested | experimental | `aot-check-ai.sh --cuda`: 0 warnings, entities exact, confidences max abs 0 |
| `long_context_over_1666_tokens` | qwen fp32x CPU | untested | split: `long_context_1667_to_8192_tokens` experimental, `long_context_over_8192_tokens` unsupported | R8 evidence ([r8-qwen35-long-context](../r8-qwen35-long-context/README.md), proposed fixture). At 4,112 and 8,213 tokens greedy ids are exact and cached vs uncached is within the protected 2e-4. At 16,391 tokens it reached **4.1e-4**. `Qwen35ChatModel` now rejects prompt plus `MaxNewTokens` above **8,192** (`MaxContextTokens`) with `GenerationContextLengthException` before any native call. `Qwen35ModelInfo.ContextLength` reports 8,192; `ModelContextLength` reports the model's 262,144. CI tests: `SequencesOverTheAcceptedContextAreRejectedBeforeInference` and the 13,702-token template-corpus case |
| same | qwen fp32x CUDA | untested | split, as above | `Fp32xCudaTests.LongContextGreedyAndLogitsEqualTheReference`, 4,112 and 8,213 tokens: greedy exact, step-0 and cached vs uncached within this profile's protected bounds ([log](logs/qwen-cuda-long-context.log)) |
| same | qwen fp16x CUDA | untested | split, as above | `Fp16xCudaTests.LongContextGreedyAndLogitsMatchTheReference`: greedy within the profile's near-tie rule, step-0 within the protected bounds (same log) |
| `aot` | qwen fp32x CUDA, fp16x CUDA | untested | experimental | `aot-check-ai.sh --cuda` against `Microsoft.ML.OnnxRuntimeGenAI.Cuda` with both variants: typed API and `IChatClient` text, ids, finish reason and usage exact. GenAI version diagnostics report unknown under AOT, as on CPU |
| `extensions_ai` | qwen fp16x CUDA | untested | experimental | `Fp16xCudaChatClientTests`: on 21 text prompts the response and streamed updates equal the typed fp16x model, with usage and finish reason; cancellation; serialized concurrency (23/23 passed, [log](logs/qwen-fp16x-cuda-chat-client.log)) |
| `prompts_task` | eg2 `onnx-fp32-cpu`, `-cuda` | untested | experimental | Proposed `task-prompts`: all 20 prompt names x 4 public texts, sentence-transformers `prompt_name`. Ids exact on 80/80; max abs 3.6e-7 (CPU) and 1.9e-7 (CUDA) against a bound of 5e-6 ([CPU](logs/eg2-task-prompts-cpu.log), [CUDA](logs/eg2-cuda-CudaParityTests.log)) |
| `input_2050_to_8192_tokens` | eg2 `onnx-fp32-cpu` | untested | experimental | Proposed `long-input` (R7 W3): 8 text rows of 2,600 to 8,192 tokens, ids exact, max abs 1.0e-7, rejections over 8,192 exact (`long-input/csharp-compare.md`) |
| same | eg2 `onnx-fp32-cuda` | untested | experimental | Same rows on CUDA (`Starlings.Fidelity.Tool.Cuda embeddinggemma2-long --device cuda:0`): text max abs 2.0e-7, media 6.5e-7, ids and rejections exact ([eg2-long-input-cuda/](eg2-long-input-cuda/csharp-compare-cuda.md)). The two padded batches of two long rows did not fit next to the tool's first embedder (BFC arena). The tool now records that instead of aborting. With the default `MaxBatchTokens` of 8,192, two such rows never share a call |
| `aot` | eg2 `onnx`, `image`, `video` CUDA | untested | experimental | `aot-check-ai.sh --cuda` targets `eg2` (text, image, interleaved, 3-frame video) and `eg2-meai`: 0 warnings, 13 vectors max abs 0 |
| `extensions_ai` | eg2 `onnx-fp32-cuda` | untested | experimental | `CudaAdapterTests`: text generator, prompts none, query and document, max abs 4.3e-7 ([log](logs/eg2-cuda-CudaAdapterTests.log)) |
| `mlnet` | eg2 `onnx-fp32-cuda` | untested | experimental | `CudaAdapterTests`: transform, 44 document rows, max abs 3.3e-7 |
| `extensions_ai` | eg2 `image-fp32-cuda` | untested | experimental | `CudaAdapterTests`: multimodal generator, 28 images from bytes (SkiaSharp), max abs 3.8e-6 (bound 1e-5) |
| `extensions_ai` | eg2 `video-fp32-cuda` | untested | experimental | Same test: 6 videos from bytes (FFmpeg 8 adapter), max abs 2.7e-7 |
| `image_budget_other` | eg2 `image-fp32-cpu`, `-cuda` | untested | experimental | Proposed `image-extra`: 6 images at 70, 140, 560 and 1,120 soft tokens, with the processor's `max_soft_tokens` set. Preprocessing digests and ids exact on 24/24. Max abs **9.2e-6** on CPU (`photo-camera@560`), close to the 1e-5 bound; 5.7e-6 on CUDA ([CPU](logs/eg2-image-budgets-cpu.log), [CUDA](logs/eg2-cuda-image-budgets.log)). On CUDA the 1,120 budget needs several GB per image: it ran out of memory when it shared the process with the other image test fixtures, and passed alone |
| `image_decode_gif_bmp_webp` | eg2 `image-fp32-cpu`, `-cuda` | untested | experimental; new `image_decode_gif_transparency`: unsupported | 13 files written by Pillow from fixture images. SkiaSharp RGB equals PIL's `convert("RGB")` on 12. **The transparent GIF differs**: PIL keeps the palette colour of its 77,042 transparent pixels, SkiaSharp zeroes them. The decoder now rejects a GIF with transparent pixels (`EmbeddingGemma2ImageRejection.GifTransparency`), covered by a CI test ([log](logs/eg2-image-formats-skiasharp.log)) |
| `video_decode_rotation_other` | eg2 `video-fp32-cpu`, `-cuda` | untested | experimental | Proposed `video-edge`: rotation 180 (two videos) and -90. Metadata, sampled indices and every frame equal to torchcodec 0.17.0 on FFmpeg 8.1 ([FFmpeg 8](logs/eg2-edge-videos-ffmpeg8.log)) and with the FFmpeg 9 adapter ([FFmpeg 9](logs/eg2-edge-videos-ffmpeg9.log)) |
| `video_hdr` | eg2 video CPU, CUDA | untested | experimental | HEVC Main 10 BT.2020 PQ (HDR10 SEI) and HLG: all 180 frames equal to torchcodec. As upstream, there is no tone mapping. Only the signalling is HDR; the content is synthetic |
| `video_interlaced` | eg2 video CPU, CUDA | untested | experimental | H.264 interlaced, top and bottom field first (`ffprobe`: every frame interlaced): all 165 frames equal. As upstream, no deinterlacing |
| `video_variable_frame_rate` | eg2 video CPU, CUDA | untested | experimental | Frame durations 1/30 then 1/15 s, and alternating 1/20 and 1/60 s (checked with `ffprobe -show_frames`). Metadata (average rate), sampled indices and all 210 frames equal |

The video decoder bound is the existing `decoder_ffmpeg.decoded_rgb_vs_torchcodec: exact`. Equal frames and indices mean the embedding path gets upstream's input, and that path is parity-tested on the accepted videos on CPU and CUDA. Decoding is CPU-side for both video profiles, as it is for the image profiles' SkiaSharp decoding.

## Other changes this needed

- `tools/aot-check-ai.sh --cuda` and the AOT consumer's CUDA build: the GPU natives, every session on CUDA device 0, and both Qwen variants. The CPU mode is unchanged (rerun: e5 and gliner2 pass).
- New CUDA test coverage, local and opt-in:
  - `E5.Tests.Cuda/CudaAdapterTests.cs`
  - `EmbeddingGemma2.Tests.Cuda/CudaAdapterTests.cs`
  - task prompts, budgets, concurrency and the long-context tests in the existing CUDA classes
  - `Fp16xCudaChatClientTests`
  - GLiNER2 CUDA cancellation and concurrency
- New CPU tests: task prompts, budgets, GLiNER2 capability cases, the GIF/BMP/WebP decoder test, and the edge-case video decoder test, compiled for the FFmpeg 8 and 9 adapters.
- `Starlings.Fidelity.Tool embeddinggemma2-long` records a padded batch that runs out of GPU memory instead of aborting the run, and the compare script reports it as not run.
- A test fix found on the way: `RealModelTests.FunctionInvocationLoopCompletesOnTheRealModel` (PR #17) disposed the class fixture's shared `Qwen35ChatClient` through its `using` pipeline. Every later test in the class then failed with `ObjectDisposedException`, depending on order. The pipeline is no longer disposed.

## Runs

| Command | Result |
| --- | --- |
| `dotnet test --solution ai-dotnet/AI.slnx -c Release --no-build --max-parallel-test-modules 2` (empty model cache, as CI) | 1,232 total, 0 failed |
| `Starlings.Models.Qwen35.Tests` with models (CPU) | 376 total, 0 failed ([log](logs/qwen-cpu-tests.log)) |
| `Starlings.Models.Qwen35.Extensions.AI.Tests` with models (CPU) | 223 total, 0 failed ([log](logs/qwen-extensions-ai-cpu-tests.log)) |
| `Starlings.Models.Gliner2.Tests` with models (CPU) | 75 total, 0 failed ([log](logs/gliner2-cpu-tests.log)) |
| `Starlings.Models.E5.Tests.Cuda` | 9 total, 0 failed, 1 skipped (the unavailable-provider check) |
| `Starlings.Models.Gliner2.Tests.Cuda` | 7 total, 0 failed |
| `Starlings.Models.EmbeddingGemma2.Tests.Cuda`, one process per class | 9 + 6 + 6 + 3 total. The only failure was `OtherBudgetsMatchTheReference`, out of GPU memory next to the class's other fixtures; it passed alone |
| `Starlings.Models.Qwen35.Tests.Cuda` long-context and fp16x chat-client classes | 4 and 23 total, 0 failed |
| `tools/aot-check-ai.sh --cuda`, targets e5, e5-meai, gliner2, eg2, eg2-meai, qwen35, qwen35-meai | all pass ([aot-cuda/](aot-cuda/)) |

Paths in the logs are shortened: `<repo>` is the checkout, `<work>` the scratch directory with the reference environments and generated media, and `<model-cache>` the Starlings model cache.
