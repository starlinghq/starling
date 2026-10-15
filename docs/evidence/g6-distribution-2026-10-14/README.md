# G6 distribution, 2026-10-14

Plan section 10, G6: clean install from built packages, native dependency resolution, offline execution, and version diagnostics. Release rule: no reliance on the developer's source checkout or local caches.

G6 is recorded as `passed` for the 11 CPU and CUDA profiles below. It stays open for the two EmbeddingGemma 2 video profiles.

## Runs

All runs used `tools/release-drill-ai.sh` (#39 added the CUDA mode, the image and fp16x public-hosting steps, and package reuse). Each run started from a clean clone of the committed HEAD. It then pushed the packages to a local folder feed and built consumers outside the repository, with an isolated empty NuGet cache and a NuGet.config that knows only that feed and nuget.org. Every `Starlings.*` package in each restore graph is checked against the packed `.nupkg` (SHA-512), and the tokenizer against the locked nuget.org bytes. Each run used fresh isolated model caches (seeded files verified by the packages' own download step, or downloaded anonymously), and ran every model offline in a network namespace with no interfaces (`unshare -rn`).

| Log | Command (abridged) | What it covers |
| --- | --- | --- |
| [cuda-drill.log](cuda-drill.log) | `--seed-model-from <e5> --seed-gliner2-from <GLiNER2> --seed-embeddinggemma2-from <graphs + images> --seed-qwen35-from <fp32x + fp16x> --cuda GPU-9e21a72e… --cuda-lib-path <CUDA 13 + cuDNN 9> --cuda-only --public-hosting-only "Qwen 3.5 fp16x" --packages-from <pack>` | Step 6: two CUDA consumers. Step 11: the fp16x bundle downloaded anonymously, then a CUDA answer. Step 13: six CUDA profiles. 336 s. |
| [eg2-image-cpu-drill.log](eg2-image-cpu-drill.log) | `--no-model --seed-embeddinggemma2-from <graphs + images> --public-hosting-only "EmbeddingGemma 2 images" --packages-from <pack>` | EmbeddingGemma 2 text (9b), ML.NET (9e) and image search (9c) on CPU from seeded graphs. Then the text, vision and multimodal graphs downloaded anonymously into an empty cache, and the image search run offline from it. |
| [cpu-drill.log](cpu-drill.log) | `--seed-model-from <e5> --seed-gliner2-from <GLiNER2> --seed-qwen35-from <fp32x> --public-hosting-only "GLiNER2,Qwen 3.5" --packages-from <pack>` | The e5 sample and checker, with the Extensions.AI and ML.NET paths against the protected fixtures; GLiNER2; Qwen fp32x through the typed API and IChatClient. Then GLiNER2 and Qwen fp32x downloaded anonymously and run offline. |
| [e5-download-drill.log](e5-download-drill.log) | `--packages-from <pack>` (default mode: no seed) | e5 acquired with a real anonymous Hub download into an empty cache (`onnx/model.onnx`, 470,268,510 bytes, and `tokenizer.json`, SHA-256 verified), then the sample, the checker and the ML.NET checker run offline. |
| [preview.6 rc-drill.log](../../releases/ai-0.1.0-preview.6/rc-drill.log) | the full preview.6 drill, `--public-hosting --aot` | Release-candidate evidence at `5204eea`: every CPU sample, public hosting of GLiNER2, EmbeddingGemma 2 text and Qwen fp32x, and the Native AOT check. Its image search was built but not run, which is why the image run above exists. |

**Packages.** One drill packed the 15 packages from a clean clone of `b690703` (locked restore, empty package cache). The later runs reused those files (`--packages-from`). Each reuse checks that the nuspecs' commit differs from HEAD in nothing that is packed: `ai-dotnet/src`, the ai-dotnet and root build files, `global.json` and the licenses. It then reruns every package check of step 4: version, license, readme, repository commit, SourceLink, exact dependency list, and no model assets or FFmpeg libraries. Nothing that is packed changed between `b690703` and `e656703`. Those branch commits are not on GitHub after the squash merge, so the tree hash makes it checkable: `ai-dotnet/src` at the packed commit is tree `d29ccf6463b2581c0144d6d7d16f199e9bd9b6ed`, the same as `main` at `c726568`. `main` has moved since then. #41 changed only the two held FFmpeg video adapters, which no profile with G6 `passed` uses. The logs name HEAD `e656703`, this change's branch commit on top of `main` at `c726568` (#39, the drill tooling); the PR is squash-merged.

**Machine.** Intel i7-5820K, Linux x64, .NET SDK 10.0.100 in the clone (`global.json`, pinned exactly) and 10.0.401 for the consumers outside it, RTX 3090 (GPU 0, by UUID `GPU-9e21a72e-3a6f-e082-3d74-4b8d361de70c`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`), driver 610.57.04. The machine was shared; GPU 1 was not used.

**CUDA prerequisites.** The CUDA 13 runtime, cuBLAS and cuDNN 9 belong to the machine, as `ai-dotnet/README.md` documents. No package ships them. These runs took them from the `nvidia-cuda-runtime` 13.0.96, `nvidia-cublas` 13.1.1.3 and `nvidia-cudnn-cu13` 9.24.0.43 wheels, put on `LD_LIBRARY_PATH` for the CUDA runs only. Each CUDA log lists the mapped libraries. Without cuDNN, the GLiNER2 graph fails on CUDA at its Einsum node (`libcudnn.so` not found). That is a documented prerequisite, not a packaging gap.

## The CUDA mode

`tools/release-drill-ai.sh --cuda <device>` builds two consumers from the packed packages, with **only GPU native packages, swapped in place of the CPU ones**:

- `tools/release-drill/cuda-consumer`: e5, GLiNER2 and EmbeddingGemma 2 (with the Extensions.AI and SkiaSharp adapters), plus `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0.
  The e5 CUDA manifest names the binding `Microsoft.ML.OnnxRuntime.Gpu`, the meta-package whose Linux natives are `Microsoft.ML.OnnxRuntime.Gpu.Linux`. The other CUDA manifests name `Gpu.Linux` directly. The consumer references `Gpu.Linux` 1.30.0, the same native binaries.
- `QwenChatCuda`: the QwenChat sample with `Microsoft.ML.OnnxRuntimeGenAI.Cuda` 0.17.1. That package brings `Microsoft.ML.OnnxRuntime.Gpu` 1.30.0, with its Linux and Windows natives.

The restore-graph check requires the GPU natives exactly once, at the pinned versions, and no `Microsoft.ML.OnnxRuntime` or `Microsoft.ML.OnnxRuntimeGenAI` CPU package beside them (`cuda-drill.log` lines 290-293). Step 13 then seeds a new cache, verifies every file through the packages' download steps, and runs each profile offline on the one GPU.

## Per profile

"Diagnostics" are the lines each run prints:

- the package versions of the loaded assemblies;
- ONNX Runtime native and managed versions, and the available providers;
- the session providers and provider options (`use_tf32=0`, `device_id=0`) with the model SHA-256;
- the ONNX Runtime and CUDA libraries mapped into the process (ONNX Runtime from the consumer's own build output);
- for Qwen: the profile, provider and GenAI version, plus the packages and the GenAI native package from the dependency manifest.

| Profile | Clean install and native resolution | Offline run and correct output | Public download (no token) | G6 |
| --- | --- | --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-cpu` | e5-download-drill, cpu-drill, preview.6 | Sample top passages 4/4. Checker against the protected embeddings within `float.embedding_max_abs`, also through Extensions.AI and ML.NET; ML.NET Save rejected | model and tokenizer downloaded from the upstream Hub, then run offline (e5-download-drill) | passed (refreshed) |
| `multilingual-e5-small/onnx-fp32-cuda` | cuda-drill: CudaConsumer, Gpu.Linux only | 46 protected embeddings, max abs 1.66e-7 (bound 1e-5), also through Extensions.AI | as CPU | **passed** |
| `gliner2-base-v1/onnx-fp32-cpu` | cpu-drill, preview.6 | Expected entities; 600-word input rejected | graph and tokenizer downloaded, then run offline (cpu-drill step 11) | **passed** |
| `gliner2-base-v1/onnx-fp32-cuda` | cuda-drill | 54/54 in-contract fixture cases: entities equal the reference; confidence max abs 1.13e-6 (bound 2e-5); inputs over 512 tokens rejected | as CPU | **passed** |
| `embeddinggemma-2/onnx-fp32-cpu` | eg2-image-cpu-drill, preview.6 | Top documents 4/4; Extensions.AI equals the typed API; Matryoshka 256; over 8,192 tokens rejected; ML.NET checker passes and Save is rejected | text graph downloaded and run offline (preview.6; again in eg2-image-cpu-drill) | **passed** |
| `embeddinggemma-2/onnx-fp32-cuda` | cuda-drill | 138 protected rows, max abs 4.7e-7 (bound 5e-6), min cosine 1.0; padded batches equal single rows; Extensions.AI over CUDA 2.6e-7 | as CPU | **passed** |
| `embeddinggemma-2/image-fp32-cpu` | eg2-image-cpu-drill: EmbeddingGemma2Search with the SkiaSharp adapter | Expected photo first for 5/5 queries; Extensions.AI images equal the typed API (max abs 0); interleaved input | **vision and multimodal graphs downloaded** (671,256,072 and 1,084,327,898 bytes, SHA-256 verified), then the image search run offline from that cache | **passed** |
| `embeddinggemma-2/image-fp32-cuda` | cuda-drill | 30 image rows (SkiaSharp decode) and 8 text rows against the protected image fixtures, max abs 3.6e-6 (bound 1e-5); multimodal Extensions.AI over CUDA 3.75e-6; the 16-bit image rejected by contract | as CPU | **passed** |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | cpu-drill, preview.6 | "The capital of France is Paris.", Stop, 24 + 7 tokens, also through IChatClient | bundle downloaded and run offline (cpu-drill step 11) | **passed** |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | cuda-drill: QwenChatCuda, GenAI.Cuda only | Exactly the protected reference answer (row `en-short` of `qwen3.5-0.8b/reference/generation.jsonl`): text, Stop, 24 + 7 tokens; IChatClient gives the same answer and usage | as CPU | **passed** |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | cuda-drill | The same protected reference answer, through the typed API and IChatClient | **fp16x bundle downloaded** (1,572,143,104-byte data file, SHA-256 verified) and answered offline on CUDA from that cache | **passed** |
| `embeddinggemma-2/video-fp32-cpu` | not drilled here | — | the graphs are public; decoding needs the FFmpeg adapter | **open** |
| `embeddinggemma-2/video-fp32-cuda` | not drilled here | — | as CPU | **open** |

## What stays open

- **EmbeddingGemma 2 video.** Both FFmpeg adapters are held from publishing until legal review D2, so no clean install from published packages is possible for a video profile. The drill does build the adapter: the preview.4 video run (`docs/evidence/embeddinggemma2-video/`) used the user's own FFmpeg 8. G6 waits for D2.
- **Windows and other platforms.** These runs cover Linux x64 only, the one platform row the manifests record as `experimental`. Windows x64 stays `untested` beyond CI's build and weight-free tests.
- **Native AOT** is a capability, not part of G6. The preview.6 drill and `docs/evidence/aot/` cover CPU; `docs/evidence/capability-closure-2026-10-14/aot-cuda/` covers CUDA. These runs did not repeat it.
- **Qwen GenAI diagnostics** report ONNX Runtime GenAI from the dependency manifest. The GenAI native library exports no version function, so this is the best the binding offers. The ONNX Runtime it loads is `Microsoft.ML.OnnxRuntime.Gpu` 1.30.0 from the checked restore graph.
