# G7 integration, 2026-10-14

Plan section 10, G7: adapter behavior, representative downstream use, and persistence if advertised. Release rule: integration badges only for tested capabilities. Plan section 15 names the integrations: Microsoft.Extensions.AI (15.1) and ML.NET (15.2).

G7 is recorded as `passed` for the 11 profiles below. The two EmbeddingGemma 2 video profiles followed on 2026-10-15 ([Video, 2026-10-15](#video-2026-10-15)).

## What each profile advertises

These are taken from the manifests' `capabilities`. Missing never means supported. Nothing is `supported`; the advertised integrations are `experimental`.

| Profile | `extensions_ai` | `mlnet` | `tool_calling` | `persistence` |
| --- | --- | --- | --- | --- |
| e5 CPU, CUDA | experimental | experimental | n/a | unsupported |
| GLiNER2 CPU, CUDA | unsupported | unsupported | n/a | unsupported |
| EmbeddingGemma 2 text CPU, CUDA | experimental | experimental | n/a | unsupported |
| EmbeddingGemma 2 image CPU, CUDA | experimental (multimodal generator) | unsupported (no image column) | n/a | unsupported |
| EmbeddingGemma 2 video CPU, CUDA | experimental (multimodal generator) | unsupported (no video column) | n/a | unsupported |
| Qwen 3.5 fp32x CPU, fp32x CUDA, fp16x CUDA | experimental (`IChatClient`) | unsupported | experimental | unsupported |

**Persistence** is `unsupported` everywhere. It is advertised nowhere, and the one API that could imply it is rejected explicitly. Saving an ML.NET transformer throws `NotSupportedException`, also inside a chain: `SaveIsUnsupportedAlsoInsideAChain` in `Starlings.Models.E5.MLNet.Tests` and `Starlings.Models.EmbeddingGemma2.MLNet.Tests` (`TransformContractTests`). The release drills confirm it from the packed packages:

- `ML.NET Save of the e5 transformer is rejected (persistence unsupported)` (AiConsumer) and `e5: ML.NET Save of the transformer is rejected` (MlNetConsumer), in [cpu-drill.log](../g6-distribution-2026-10-14/cpu-drill.log) lines 291 and 307;
- `EmbeddingGemma 2: ML.NET Save of the transformer is rejected (persistence unsupported)`, in [eg2-image-cpu-drill.log](../g6-distribution-2026-10-14/eg2-image-cpu-drill.log) line 300.

The Qwen and GLiNER2 packages expose no save API.

## Adapter behavior and downstream use, per profile

The drill's sample consumers are the representative downstream use: E5Search, Gliner2Ner, QwenChat, EmbeddingGemma2Search, AiConsumer, MlNetConsumer, and the new CudaConsumer and QwenChatCuda. They are built outside the repository from the packed packages and run offline ([G6](../g6-distribution-2026-10-14/README.md)). Test results with the weights present: the preview.6 record (988 passed, 0 failed; image and video fixtures absent then) and the test run of this change (see the PR).

| Profile | Adapter tests | Downstream use (drill) | G7 |
| --- | --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-cpu` | Extensions.AI: `E5.Extensions.AI.Tests` `GeneratorContractTests` (weight-free) and `RealModelTests` (protected embeddings, cancellation, ownership and DI disposal). ML.NET: `E5.MLNet.Tests` `TransformContractTests` and `RealModelTests` (row-wise and look-ahead batching against the protected tolerance) | E5Search (Extensions.AI equals the typed API), AiConsumer (generator and transform against the protected fixtures), MlNetConsumer | **passed** |
| `multilingual-e5-small/onnx-fp32-cuda` | `E5.Tests.Cuda` `CudaAdapterTests`: generator and transform over a CUDA session, 46 protected texts, max abs 1.66e-7 (#19, [log](../capability-closure-2026-10-14/logs/e5-cuda-tests.log)) | CudaConsumer: Extensions.AI generator over CUDA against the protected embeddings, 1.66e-7 | **passed** |
| `gliner2-base-v1/onnx-fp32-cpu` | No adapter is advertised (`extensions_ai` and `mlnet` unsupported; no adapter package exists), so there is no adapter behavior to test. The typed API is the integration surface. | Gliner2Ner: expected entities and offsets, long input rejected | **passed** |
| `gliner2-base-v1/onnx-fp32-cuda` | as CPU | CudaConsumer: 54/54 fixture cases equal the reference through the public API on CUDA | **passed** |
| `embeddinggemma-2/onnx-fp32-cpu` | Extensions.AI: `EmbeddingGemma2.Extensions.AI.Tests` `GeneratorContractTests` and `RealModelTests` (per prompt against the protected tolerance, Matryoshka dimensions). ML.NET: `EmbeddingGemma2.MLNet.Tests` `TransformContractTests` and `RealModelTests` | EmbeddingGemma2Search (Extensions.AI equals the typed API, Matryoshka 256), MlNetConsumer (transform equal to the typed API, Matryoshka column) | **passed** |
| `embeddinggemma-2/onnx-fp32-cuda` | `EmbeddingGemma2.Tests.Cuda` `CudaAdapterTests`: text generator (prompts none, query, document) max abs 4.3e-7; ML.NET transform, 44 rows, 3.3e-7 (#19, [log](../capability-closure-2026-10-14/logs/eg2-cuda-CudaAdapterTests.log)) | CudaConsumer: generator over CUDA, 46 query rows, 2.6e-7 | **passed** |
| `embeddinggemma-2/image-fp32-cpu` | `MultimodalGeneratorContractTests` (weight-free: `DataContent` images through the decoder hook, errors, usage) | EmbeddingGemma2Search `images`: the multimodal generator with the SkiaSharp decoder equals the typed API (max abs 0) on the photos, from the seeded and the publicly downloaded graphs | **passed** |
| `embeddinggemma-2/image-fp32-cuda` | `CudaAdapterTests`: multimodal generator, 28 images from bytes, max abs 3.8e-6 (bound 1e-5) (#19) | CudaConsumer: the multimodal generator over CUDA against the protected image rows, 3.75e-6 | **passed** |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | `Qwen35.Extensions.AI.Tests`: `ChatClientContractTests`, `RealModelTests` (21 prompts, streaming equals non-streamed equals reference, cancellation, concurrency), and tool calling (`ToolCallingTests`: 54 core and 21 client tests; `TemplateCorpusChatClientTests`; the real-model call and the `UseFunctionInvocation` loop, [qwen35-tool-calling](../qwen35-tool-calling/README.md)) | QwenChat: typed API and `--chat-client` give the same answer, finish reason and usage | **passed** |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | `Fp32xCudaChatClientTests` (P-017 on CUDA, [log](../cuda-qualification-2026-10-14/cuda-tests/Starlings.Models.Qwen35.Tests.Cuda.log)). **New**: tool calling on CUDA. The tools prompt gives the reference `get_weather(Paris)` call, with all 25 completion ids equal to the fp32 reference, and the `UseFunctionInvocation` loop completes ([qwen-cuda-tool-calling.log](qwen-cuda-tool-calling.log)) | QwenChatCuda: the protected reference answer through the typed API and IChatClient | **passed** |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | `Fp16xCudaChatClientTests` (#19, [log](../capability-closure-2026-10-14/logs/qwen-fp16x-cuda-chat-client.log)). **New**: tool calling on CUDA, the same two checks. The fp16x completion ids also equal the reference (reported, not required) | QwenChatCuda: as fp32x, from the publicly downloaded fp16x bundle | **passed** |
| `embeddinggemma-2/video-fp32-cpu` | Extensions.AI video is tested: `MultimodalGeneratorContractTests.VideosUseTheVideoDecoderAndEqualTheEmbedder` (fake graphs). The preview.4 drill ran the sample's video search with the user's FFmpeg 8: `extensions.ai videos: … max abs difference from the typed API 0` ([excerpt](../embeddinggemma2-video/release-drill-excerpt.txt)). ML.NET unsupported. | EmbeddingGemma2Search `videos` and EmbeddingGemma2SearchFFmpeg9 from the packed adapters: the multimodal generator with the adapter as `VideoDecoder` equals the typed API (max abs 0); the packed ML.NET adapter rejects a video column ([below](#video-2026-10-15)) | **passed** (2026-10-15) |
| `embeddinggemma-2/video-fp32-cuda` | `CudaAdapterTests`: multimodal generator over CUDA, 6 videos from bytes, max abs 2.7e-7 (#19). ML.NET unsupported. | CudaConsumer `eg2-video` (generator over CUDA, 6 videos from bytes through each packed adapter, 3.89e-6 from the protected rows) and EmbeddingGemma2SearchCuda `videos --cuda` (generator 1.88e-7 from the typed batch) ([below](#video-2026-10-15)) | **passed** (2026-10-15) |

## What stays open

- ~~**EmbeddingGemma 2 video.**~~ Closed on 2026-10-15 ([below](#video-2026-10-15)): D2 cleared the FFmpeg adapters, and the drill now runs the sample's video path and the Extensions.AI video generator from the packed packages on CPU and CUDA.
- **Tool use quality** is not part of G7 and stays unqualified. One prompt on the 0.8B model shows the mechanics, not reliable tool choice ([qwen35-tool-calling](../qwen35-tool-calling/README.md), "Not done").
- **No named downstream consumer yet** (M7, P-028 in the support policy). The drill's samples stand in for one. A first named consumer remains a maintainer action before a Stable label.

## Changed for this record

- `Starlings.Models.Qwen35.Tests.Cuda`: `CudaToolCallingChecks` and two facts per variant (`ToolsPromptGivesTheReferenceCallAsFunctionCallContent`, `FunctionInvocationLoopCompletesOnCuda`). The project references `Microsoft.Extensions.AI` for the function-invocation middleware. These are local GPU tests, outside `AI.slnx`, run on GPU 0 with TF32 off.
- The fp32x and fp16x CUDA manifests' `tool_calling` comments now cite this run instead of "not on CUDA".

## Video, 2026-10-15

G7 is now `passed` for `embeddinggemma-2/video-fp32-cpu` and `video-fp32-cuda`. Legal review D2 cleared the FFmpeg adapters, so representative downstream use, a user installing the packages, is possible. The runs are the 2026-10-15 drills of [G6](../g6-distribution-2026-10-14/README.md#video-2026-10-15). All three were built outside the repository from packages packed at `5b04d83`, and each run was offline.

What the video profiles advertise is unchanged: `extensions_ai: experimental` (the multimodal generator), `mlnet: unsupported`, `persistence: unsupported`.

| Integration | CPU (`video-fp32-cpu`) | CUDA (`video-fp32-cuda`) |
| --- | --- | --- |
| The sample's video path (EmbeddingGemma2Search `videos`) | The packed `…Video.FFmpeg` on the user's LGPL FFmpeg 8.1.3, and the same sample on the packed `…Video.FFmpeg9` on LGPL FFmpeg 9.0.2: the expected clip first for 3/3 queries; frames without timing rejected (`EmbeddingGemma2VideoTimingRequiredException`). Also from the anonymously downloaded graphs ([eg2-video-cpu-drill.log](../g6-distribution-2026-10-14/eg2-video-cpu-drill.log), [eg2-video-public-drill.log](../g6-distribution-2026-10-14/eg2-video-public-drill.log)) | EmbeddingGemma2SearchCuda `videos --cuda` (GPU natives only, `use_tf32=0`, `device_id=0`): the same top clips; frames without timing rejected ([eg2-video-cuda-drill.log](../g6-distribution-2026-10-14/eg2-video-cuda-drill.log) lines 656 to 671) |
| Microsoft.Extensions.AI video generator (`EmbeddingGemma2MultimodalEmbeddingGenerator` with the adapter as `VideoDecoder`, a video `DataContent` from bytes) | Equal to the typed API, max abs 0, through both adapters | The sample: 1.88e-7 from the typed batch (bound 1e-5, `padded_batch_vs_single_embedding_max_abs`). CudaConsumer: 6 videos from bytes through each adapter, 3.89e-6 from the protected embeddings (bound 1e-5) |
| ML.NET | `unsupported`. The packed `Starlings.Models.EmbeddingGemma2.MLNet` rejects a video column explicitly: `Schema mismatch for input column 'Video': expected String …, got Vector<Byte>` (`ArgumentOutOfRangeException` from `Fit`; MlNetConsumer, cpu log line 327). The weight-free `WrongInputColumnFailsWithASchemaError` test covers the same error for a non-text vector column | as CPU (one adapter, no provider-specific path) |
| Persistence | `unsupported`; Save of the transformer rejected (MlNetConsumer, the same run) | as CPU |

The adapter behavior is tested as recorded above. `MultimodalGeneratorContractTests.VideosUseTheVideoDecoderAndEqualTheEmbedder` runs in CI on fake graphs, and `CudaAdapterTests` ran on six videos (#19). The FFmpeg 9 adapter's tests (the FFmpeg 8 tests compiled for FFmpeg 9) ran on the LGPL FFmpeg 9.0.2 build with the fixture videos: 26 passed, and 2 were skipped (the edge-case videos were not set, and the release-video test is FFmpeg 8 only). That includes every decoded frame of the 22 fixture videos equal to torchcodec's.

The named downstream consumer (M7) is still missing, as for the other profiles.
