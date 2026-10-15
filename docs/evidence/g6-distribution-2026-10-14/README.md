# G6 distribution, 2026-10-14

Plan section 10, G6: clean install from built packages, native dependency resolution, offline execution, and version diagnostics. Release rule: no reliance on the developer's source checkout or local caches.

G6 is recorded as `passed` for the 11 CPU and CUDA profiles below. The two EmbeddingGemma 2 video profiles followed on 2026-10-15, after legal review D2 cleared the FFmpeg adapters ([Video, 2026-10-15](#video-2026-10-15)).

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
| `embeddinggemma-2/video-fp32-cpu` | eg2-video-cpu-drill: EmbeddingGemma2Search (FFmpeg adapter) and EmbeddingGemma2SearchFFmpeg9, CPU natives only | Expected clip first for 3/3 queries through each adapter; Extensions.AI videos equal the typed API (max abs 0); frames without timing rejected; GPL builds refused ([below](#video-2026-10-15)) | **vision and multimodal graphs downloaded** with no token, then the video search run offline from that cache (eg2-video-public-drill) | **passed** (2026-10-15) |
| `embeddinggemma-2/video-fp32-cuda` | eg2-video-cuda-drill: CudaConsumer (both adapters) and EmbeddingGemma2SearchCuda, Gpu.Linux only | 22 fixture videos through each adapter: sampled indices 23/23 and 181/181 frames exact; 28 rows max abs 6.5e-7 (bound 1e-5); Extensions.AI 3.9e-6; the sample's video search on CUDA ([below](#video-2026-10-15)) | as CPU | **passed** (2026-10-15) |

## What stays open

- ~~**EmbeddingGemma 2 video.**~~ Closed on 2026-10-15 ([below](#video-2026-10-15)): D2 cleared both FFmpeg adapters, and the drill now runs the video path from the packed adapters on CPU and CUDA.
- **Windows and other platforms.** These runs cover Linux x64 only, the one platform row the manifests record as `experimental`. Windows x64 stays `untested` beyond CI's build and weight-free tests.
- **Native AOT** is a capability, not part of G6. The preview.6 drill and `docs/evidence/aot/` cover CPU; `docs/evidence/capability-closure-2026-10-14/aot-cuda/` covers CUDA. These runs did not repeat it.
- **Qwen GenAI diagnostics** report ONNX Runtime GenAI from the dependency manifest. The GenAI native library exports no version function, so this is the best the binding offers. The ONNX Runtime it loads is `Microsoft.ML.OnnxRuntime.Gpu` 1.30.0 from the checked restore graph.

## Video, 2026-10-15

G6 is now `passed` for `embeddinggemma-2/video-fp32-cpu` and `video-fp32-cuda`. Legal review D2 cleared both FFmpeg adapters on 2026-10-15, so consumers can install them from the published set. The drill gained the video steps it lacked (this change), and all three runs below passed from one set of packages.

**Packages.** The CPU run packed all 15 packages from a clean clone of `5b04d83` (this change's branch commit, on `main` after #64), with a locked restore and an empty package cache. The CUDA and public-hosting runs reused those files (`--packages-from`), and every package check of step 4 ran again. The adapter packages hold their DLL, XML docs, README and `THIRD-PARTY-NOTICES.md`, and no FFmpeg library.

**The user's FFmpeg.** No FFmpeg is packed, restored or downloaded by any package. The drill passes the user's own shared libraries with `--ffmpeg-dir`:

- FFmpeg 8: the pinned BtbN LGPL shared build `n8.1.3-17-gc389840894`, the one the earlier video evidence used (`fetch_eg2_ffmpeg.sh`, archive SHA-256 `bf3db18a…`).
- FFmpeg 9 (`--ffmpeg9-dir`, new): the BtbN LGPL shared build `n9.0.2-28-g669f5ab083`. It is `ffmpeg-n9.0.2-28-g669f5ab083-linux64-lgpl-shared-9.0.tar.xz` from release `autobuild-2026-10-15-13-03`, SHA-256 `5838264a2cff06ddc6b7b2fc43edeb51358ead38fd56d82f83f49c270fe0f5ff`. Autobuild releases expire upstream, so the archive is kept outside Git.
- GPL builds to refuse (`--ffmpeg-gpl-dir`, `--ffmpeg9-gpl-dir`, new): BtbN GPL shared `n8.1.3-17-gc389840894`, and the distribution's GPL FFmpeg 9.0.1 (`/usr/lib`).

| Log | Command (abridged) | What it covers |
| --- | --- | --- |
| [eg2-video-cpu-drill.log](eg2-video-cpu-drill.log) | `--no-model --seed-embeddinggemma2-from <graphs + images + videos> --ffmpeg-dir <8.1.3 LGPL> --ffmpeg9-dir <9.0.2 LGPL> --ffmpeg-gpl-dir <8.1.3 GPL> --ffmpeg9-gpl-dir /usr/lib` | Pack (step 3). Restore graphs: EmbeddingGemma2Search holds exactly `…Video.FFmpeg`, and EmbeddingGemma2SearchFFmpeg9 exactly `…Video.FFmpeg9` (lines 255 to 258). The ML.NET video rejection (line 327). Video search through each adapter: 9d (line 352) and 9f (line 370). The GPL refusals (lines 387 to 390). 384 s. |
| [eg2-video-cuda-drill.log](eg2-video-cuda-drill.log) | `--seed-* … --ffmpeg-dir … --ffmpeg9-dir … --cuda 0 --cuda-lib-path <CUDA 13 + cuDNN 9> --cuda-only --packages-from <pack>` | Step 6: EmbeddingGemma2SearchCuda and CudaConsumer, with GPU natives only (lines 343 and 345). Step 13: the six CUDA profiles of 2026-10-14 again, plus `video-fp32-cuda`: CudaConsumer `eg2-video` with each adapter (lines 554 to 655), then the sample's video search with `--cuda` (lines 656 to 671). 275 s. |
| [eg2-video-public-drill.log](eg2-video-public-drill.log) | the CPU command plus `--public-hosting-only "EmbeddingGemma 2 videos" --packages-from <pack>` | The CPU video steps again. Step 11: the text, vision and multimodal graphs downloaded into an empty cache with no token and no seed (lines 399 to 403; 671,256,072 and 1,084,327,898 bytes, SHA-256 verified), then the video search run offline from that cache (lines 405 to 420). 365 s. |

**Machine.** As above: i7-5820K, RTX 3090 as GPU 0 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`), driver 610.57.04. GPU 1 was not used. CUDA 13 and cuDNN 9 came from the `nvidia-*` wheels of the EmbeddingGemma 2 CUDA venv, on `LD_LIBRARY_PATH` for the CUDA runs only. The machine was shared.

### New in the drill

- `EmbeddingGemma2SearchFFmpeg9`: the sample with `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` in place of `.FFmpeg`. Only the package id and the `using` line differ; the adapters share their code and type names. Step 9f runs its video search on the FFmpeg 9 build.
- GPL refusal: with `--ffmpeg-gpl-dir` or `--ffmpeg9-gpl-dir`, the matching sample must exit 3 with "is not LGPL".
- `EmbeddingGemma2SearchCuda`: the sample with `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 in place of `Microsoft.ML.OnnxRuntime`. The restore graph is checked like CudaConsumer's: GPU natives exactly once and no CPU natives. The sample gained `--cuda` (device 0, package defaults) and now prints its session diagnostics.
- CudaConsumer `eg2-video <cache> <fixtures> <tolerances> <video dir> <FFmpeg dir> 8|9`. It checks the 22 fixture videos against the protected `videos.json` (SHA-256), decodes them with the adapter of that major (an LGPL build only, with no opt-in), and checks the sampled indices and every sampled frame's SHA-256 (exact). It then checks the embeddings of the 23 video rows and 5 text queries, and the Extensions.AI generator on the 6 smallest videos from their bytes, within the protected `video-fp32-cuda` bounds.
- Step 11 `EmbeddingGemma 2 videos`: the video path's graphs (the image lock) downloaded or verified anonymously, then the video search run offline from that cache.
- Every video search log is checked the same way. The FFmpeg major is reported as LGPL. The expected clip is first for each of the three queries. Frames without timing are rejected. Extensions.AI must equal the typed API exactly on CPU. On CUDA the generator embeds one video alone while the typed API embeds a batch, so it must stay within the profile's protected `padded_batch_vs_single_embedding_max_abs`.

### Per profile

| Profile | Clean install and native resolution | Offline run and correct output | Version diagnostics | Public download (no token) |
| --- | --- | --- | --- | --- |
| `video-fp32-cpu` | EmbeddingGemma2Search and EmbeddingGemma2SearchFFmpeg9 restore only packages, and every `Starlings.*` package is the packed bytes. No GPU natives and no FFmpeg library | FFmpeg 8 and FFmpeg 9: `h264-1fps-64x48-8s`, `h264-1.9s-320x240-30fps` and `h264-portrait-360x640-30fps-4s` ranked first for their queries (0.7780, 0.7831, 0.7420, the same on both majors). Extensions.AI max abs 0 from the typed API. Frames without timing rejected | `ffmpeg: <version> (LGPL) from <dir>; <adapter package> 0.1.0-preview.6`, ONNX Runtime native and managed 1.30.0, the session providers and model SHA-256 | vision and multimodal graphs downloaded, then the search run from them |
| `video-fp32-cuda` | EmbeddingGemma2SearchCuda and CudaConsumer: `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 only. ONNX Runtime and its CUDA provider mapped from the consumer's build output | CudaConsumer, each adapter (FFmpeg 8.1.3 and FFmpeg 9.0.2 LGPL): fixture files 22/22, indices 23/23 and frames 181/181 exact. 28 rows max abs 6.50e-7 (bound 1e-5), min cosine 1.0 (bound 0.9999995). Extensions.AI over CUDA, 6 videos from bytes, 3.89e-6. The sample with `--cuda`: the same top clips, Extensions.AI 1.88e-7 from the typed batch (bound 1e-5) | as CPU, plus `use_tf32=0` and `device_id=0` on both sessions, and the CUDA 13, cuBLAS and cuDNN libraries the process mapped | as CPU (one lock) |

### Not covered

- **Windows.** The video adapters target x86-64 Linux; the generated bindings record the target, and the adapter refuses other process architectures. Windows stays `untested`, as for the other profiles.
- **Native AOT** of the FFmpeg adapters is untested (the manifests say so). These runs did not use `--aot`.
- **CPU video on the release-candidate drill.** `--public-hosting` without a video seed or `--ffmpeg-dir` downloads and verifies the graphs, but it skips the video search with a note, as the image entry does without images.
