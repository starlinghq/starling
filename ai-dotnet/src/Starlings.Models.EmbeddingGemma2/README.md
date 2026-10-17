# Starlings.Models.EmbeddingGemma2

Embeddings from [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, on ONNX Runtime 1.30.0, in-process and offline after an explicit download. The package is versioned 0.x, so the API may still change between minor versions; maturity is per profile (below).

- **Text** (`EmbeddingGemma2TextEmbedder`, `EmbeddingGemma2Encoder`): owned tokenizer or token ids, on the released text graph. The text encoder rejects the ids of the media placeholder and boundary tokens (`<|image|>`, `<|audio|>`, `<|video|>`, `<|image>`, `<image|>`, `<|audio>`, `<audio|>`).
- **Images** (`EmbeddingGemma2MultimodalEmbedder`, experimental): an image alone, or text and images interleaved, from **already-decoded RGB** (`EmbeddingGemma2Image`), default budget of 280 soft tokens per image. See [Images](#images).
- **Videos** (`EmbeddingGemma2MultimodalEmbedder`, `EmbeddingGemma2Video`, experimental): a video alone, or text, images and videos interleaved, from **already-decoded RGB frames**, at the checkpoint's settings (1 frame per second, at most 32 frames, 140 soft tokens per frame, no timestamps). Audio is not supported. See [Videos](#videos).
- Right-padded batching, mean pooling over every attended token (prompt and image tokens included), L2 normalization: 768-dimensional unit vectors, or a Matryoshka size of 512, 256 or 128 (truncate, then normalize again, as sentence-transformers `truncate_dim`). Other sizes are rejected.
- Prompts are explicit and never added for you: `EmbeddingGemma2Prompt.Query` (`"task: search result | query: "`) for queries, `EmbeddingGemma2Prompt.Document` (`"title: none | text: "`) for documents, any of the 20 names in the model's `config_sentence_transformers.json` through `EmbeddingGemma2Prompt.FromName`, or `None` (the default). The model card: "Prefixes apply to text only. Pass images, video, and audio without any prefix."
- Inputs above 8,192 tokens (the card's context window; special tokens, prompt and image tokens included) are rejected with `EmbeddingGemma2InputTooLongException`, never truncated.
- CUDA runs with TF32 disabled (`use_tf32=0`); with TF32 the text embeddings drift by up to 1.9e-4 and the image embeddings by up to 1.4e-3.
- The package ships the asset locks (repository, commit, sizes and SHA-256), not the model files (1.1 GB for text, 2.9 GB with images). Acquisition is an explicit call; inference never downloads.

## Maturity

<!-- catalog:maturity profiles=embeddinggemma-2/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/image-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/image-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `embeddinggemma-2/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/video-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `embeddinggemma-2/video-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Tokenizer

Text goes through the owned tokenizer `Starlings.Tokenizers` (profile `gemma-bpe-embeddinggemma2`, exact against huggingface/tokenizers on the published fixtures): `EmbeddingGemma2Tokenizer` adds `<bos>` and `<eos>` and never truncates. No Microsoft.ML.Tokenizers and no native tokenizer library. Any other `IEmbeddingGemma2Tokenizer` must reproduce the pinned `tokenizer.json` exactly.

## Install

```sh
dotnet add package Starlings.Models.EmbeddingGemma2
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # CPU natives
# or, for CUDA on Linux x64: Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0 (reference exactly one)
# optional, to decode image files: Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp
# optional, to decode video files on your own FFmpeg 8.x: Starlings.Models.EmbeddingGemma2.Video.FFmpeg
```

## Use

```csharp
using Starlings.Models.EmbeddingGemma2;
using Starlings.Runtime.Onnx;

await EmbeddingGemma2ModelAssets.DownloadAsync();                       // once, explicit; verifies SHA-256
EmbeddingGemma2ModelFiles files = EmbeddingGemma2ModelAssets.OpenLocal(); // offline, verified, never downloads

using var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions
{
    Session = new OnnxSessionOptions { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256 },
    Dimension = 768,                                                    // or 512, 256, 128
});

var tokenizer = EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath);   // verifies the pinned SHA-256
var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
float[] q = embedder.Embed("how much protein should a female eat", EmbeddingGemma2Prompt.Query);
float[][] docs = embedder.EmbedBatch(["As a general guideline, ..."], EmbeddingGemma2Prompt.Document);
```

Vectors are normalized, so cosine similarity is the dot product.

The graph is hosted as a GitHub release asset of starlinghq/starling (`embeddinggemma-2-onnx-r1`). Until that release is published, `DownloadAsync` fetches the Hub files and fails on the graph; produce the graph with `model-pipeline/reference/embeddinggemma2/release_embeddinggemma2_onnx.py` (the digest must match) and use `OpenDirectory`.

## Images

```csharp
await EmbeddingGemma2ModelAssets.DownloadImageAssetsAsync();            // text files + vision graph + multimodal text graph
var imageFiles = EmbeddingGemma2ModelAssets.OpenLocalImageAssets();
using var images = EmbeddingGemma2MultimodalEmbedder.Create(imageFiles);  // SHA-256 of each graph checked while loading

// Decoded RGB from your own stack (3 bytes per pixel, rows top to bottom, no padding), or the SkiaSharp adapter:
EmbeddingGemma2Image photo = EmbeddingGemma2Image.FromRgb24(width, height, rgbBytes);
float[] e = images.Embed(photo);                                          // an image alone, no prompt (the card)
float[] mixed = images.Embed(EmbeddingGemma2MultimodalInput.Create("A cat resting on a blanket. ", photo));
float[] text = images.Embed(EmbeddingGemma2MultimodalInput.FromText("a photo of a cat"), EmbeddingGemma2Prompt.Query);
```

- **Pipeline (upstream, restated from decoded RGB).** Aspect-preserving target size with sides that are multiples of 48 and at most 2,520 patches; PyTorch's uint8 antialiased bicubic resize, ported bit for bit (`EmbeddingGemma2ImageProcessor`); `x float32(1/255)`; 16x16 patches with (x, y) positions; the vision graph (16 layers, 3x3 pooling, projection to 512); the soft tokens replace the token embeddings of `<|image>` + n x `<|image|>` + `<image|>` in the multimodal text graph; mean pooling over every token.
- **Interleaving.** `EmbeddingGemma2MultimodalInput.Create(parts)` takes strings and images in order; the text parts are concatenated with no separator and each image is expanded at its position. Text parts must not contain media placeholders. sentence-transformers' `{"text": "... <|image|>", "image": img}` becomes `Create("... ", img)`; text without a placeholder followed by images becomes `Create(text, img)`.
- **Graphs.** Inputs with images run on the vision graph and the multimodal text graph (release `embeddinggemma-2-vision-onnx-r1`); text-only inputs run on the released text graph and equal `EmbeddingGemma2TextEmbedder` bit for bit (it loads on the first text-only input). The multimodal graph also reproduces the released graph bit for bit on text-only input.
- **Limits.** Images above 50 megapixels (`MaxImagePixels`) throw `EmbeddingGemma2ImageTooLargeException`; inputs above 8,192 tokens after image expansion (about 29 images at 280 tokens) throw `EmbeddingGemma2InputTooLongException`. Both are checked before any inference.
- **Upstream conversion rules** apply to whatever decoder you use: alpha dropped (not composited), grayscale replicated, palettes looked up, EXIF orientation not applied, 8 bits per channel (16-bit images are rejected by the SkiaSharp adapter; upstream turns them white).
- **Budget.** 280 soft tokens per image (the checkpoint's default); `MaxImageSoftTokens` also accepts 70, 140, 560 and 1,120 (experimental: proposed rows from sentence-transformers, within the protected bounds).
- **Memory and batching.** One image per vision-graph call by default (`MaxVisionBatchSize = 1`): on ONNX Runtime CUDA one 2,520-patch image needs about 610 MB of attention buffer per layer, and two did not fit in 5.5 GB. On CPU, vision batches equal single images bit for bit; raise `MaxVisionBatchSize` and `MaxVisionBatchPatches` to trade memory for throughput. Each call is padded to its longest image (`PadImagesToMaxPatches = false`; upstream pads every image to 2,520 patches, which gives the same soft tokens on CPU and within 3.4e-5 on CUDA).
- **CUDA.** The vision graph needs cuDNN 9 on the loader path (its `ReduceMax` runs through cuDNN); the three graphs take about 2.9 GB of GPU memory.
- **Speed (indicative, i7-5820K 12 threads; RTX 4060 Ti).** Vision about 4.5 s per full-budget image on CPU, 0.23 s on CUDA; preprocessing about 1.5 to 2 ms for a photo and 5 ms for 4000x3000. The resize and the patch values run on AVX2 when the CPU has it and split the rows of large images over the thread pool; the bytes and values are the same on every path (integer resize, one float32 product per value).

## Videos

```csharp
var imageFiles = EmbeddingGemma2ModelAssets.OpenLocalImageAssets();        // same graphs as images
using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(imageFiles);

// Every decoded frame (RGB24) plus the average frame rate and duration: the package samples as upstream.
EmbeddingGemma2Video video = EmbeddingGemma2Video.FromDecodedFrames(frames, framesPerSecond: 30, durationSeconds: 5);
float[] e = embedder.Embed(video);                                          // a video alone, no prompt (the card)
float[] mixed = embedder.Embed(EmbeddingGemma2MultimodalInput.Create("The clip ", video, " shows moving shapes."));

// Or decode with the optional adapter on your own FFmpeg 8.x (Starlings.Models.EmbeddingGemma2.Video.FFmpeg):
// var clip = new EmbeddingGemma2VideoDecoder().DecodeFile("clip.mp4"); embedder.Embed(clip.Video);
```

- **Frames in, three forms.** `FromDecodedFrames(frames, fps, duration)`: every decoded frame with the timing torchcodec reports (packet count, `max(pts + duration) - min(pts)`, frames / duration); only the sampled frames are read, so `frames` may be a lazy list. `FromSampledFrames(frames)`: 1 to 32 frames you sampled yourself, used as given (equal to the file only when they are upstream's frames). `FromFramesWithoutTiming(frames)`: **rejected by default** with `EmbeddingGemma2VideoTimingRequiredException`; `AllowVideosWithoutTiming = true` opts in to upstream's behaviour for an array without metadata (every frame, thinned to 32), which gives a **different embedding than the same file** (cosine 0.965 on a 5 s fixture clip).
- **Sampling (upstream, exact).** `EmbeddingGemma2VideoSampling.SampleIndices`: `max(1, int(duration))` candidates at 1 frame per second, candidate `i` = frame `min(total - 1, int(i * fps))`, more than 32 thinned like `np.linspace(..., dtype=int)`. Quirks kept for parity: a clip under 2 s gives one frame; sampling is by frame index from the average rate (variable frame rate samples frames, not seconds).
- **Per frame.** The image preprocessing at 140 soft tokens (1,260 patches; same resize kernel; up to 8 frames preprocessed in parallel ahead of their vision calls, `EmbeddingGemma2ImageProcessor.Preprocess(frames, 140)` for your own pipeline), the vision graph in calls of at most `MaxVideoFramesPerVisionCall` frames (default 2, GPU-safe; on CPU frame batches equal single frames bit for bit), and `<|image>` + n x `<|video|>` + `<image|>` per frame with no separator and no timestamps. A video costs `frames x (n + 2)` tokens, at most 4,544; inputs above 8,192 tokens are rejected before inference (two maximal videos do not fit).
- **Decoding.** FFmpeg's default conversion is what upstream embeds (10-bit and odd sizes included); other converters move the embedding by up to 1.7e-2. Use the FFmpeg adapter (exact against torchcodec on the fixtures) or an equivalent FFmpeg call (`ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgb24 -` plus torchcodec-style metadata).
- **Speed (indicative).** Vision about 2.7 s per frame on CPU (i7-5820K, 12 threads: a 32-frame video takes about 70 to 100 s) and about 0.07 s on CUDA (RTX 4060 Ti); prefer CUDA for video.

## Memory on Linux

These apply to every profile on Linux.

- **Heap trim on release (on by default).** When a session's native memory is released (after `Dispose` and the last in-flight run), `Starlings.Runtime.Onnx` calls glibc's `malloc_trim(0)`, so a process that loads and unloads the model returns the freed memory instead of keeping it resident (text on the hosted runner: about 1,480 MiB after unloading without it, 400 to 500 MiB with it). It never runs on the request path and changes no allocator setting. Opt out per session with `OnnxSessionOptions.TrimNativeHeapOnRelease = false`, or for the whole process with the `AppContext` switch `Starlings.Runtime.Onnx.DisableNativeHeapTrim` (details in the `Starlings.Runtime.Onnx` README).
- **No CPU memory arena on CPU (default).** The package creates its CPU sessions with ONNX Runtime's CPU memory arena off (`OnnxSessionOptions.CpuMemoryArena`, `null` by default, means off here; set it to `true` to restore the arena). With the arena, the loaded video embedder's resident memory rose by about 2.2 GB in 15 minutes on the hosted runner with one session kept loaded, and the arena's freed regions stayed resident in glibc's per-thread heaps after unloading; without it, the loaded footprint is about a third smaller (video about 2.7 GB instead of 3.9 GB, text 1.8 GB instead of 2.1 GB) and the one-hour hosted soaks of text and video passed in three qualification runs out of three (they failed before). Embeddings are bit-identical with and without the arena (a test checks it); warm latency changed by 1 to 2% on text and not measurably on video (development runs). Evidence: [EmbeddingGemma 2 hosted soak](https://github.com/starlinghq/starling/blob/main/docs/evidence/eg2-hosted-soak/README.md).
- **Dedicated run threads on CPU (default).** The package's CPU sessions run on the runtime's dedicated run threads (`OnnxSessionOptions.DedicatedRunThreads`, `null` by default, means on here; set it to `false` to run on the calling thread). Without the arena, each run's buffers are freed into the calling thread's glibc heap, which keeps up to 64 MiB of them resident; callers from the .NET thread pool rotate, and the hosted one-hour soaks of text and images failed intermittently as more and more heaps kept that memory (text: 3 to 7 heaps of about 19 MiB; images: 2 to 8 heaps of up to 59 MiB). On a fixed set of threads (at most one per processor, started only when all are busy) the same heaps are reused. Embeddings are bit-identical either way (a test checks it). Evidence: [EmbeddingGemma 2 soak reliability](https://github.com/starlinghq/starling/blob/main/docs/evidence/eg2-soak-reliability/README.md).
- **Transparent huge pages.** The qualification soaks ran with `/sys/kernel/mm/transparent_hugepage/enabled` set to `madvise` (the Ubuntu default). On hosts set to `always`, the kernel's khugepaged fills in partly used 2 MiB heap ranges, so resident memory can look larger and rise over time without any leak. Evidence: [scheduled regression fix](https://github.com/starlinghq/starling/blob/main/docs/evidence/scheduled-regression-fix/README.md).
- The Stable CPU profiles (`onnx-fp32-cpu`, `image-fp32-cpu`) and the Stable CUDA text profile (`onnx-fp32-cuda`) need no `MALLOC_*` setting; their one-hour soaks ran with the defaults.

### Image and video on CUDA

**Deployment requirement for `embeddinggemma-2/video-fp32-cuda`: set `MALLOC_MMAP_THRESHOLD_=65536` in the environment of a Linux process that runs it.** It is recommended for `image-fp32-cuda` too.

With default glibc settings, a process that loads and unloads the multimodal embedder repeatedly keeps about 0.7 to 0.8 GiB of freed native memory resident in glibc's main heap between loads, and that high-water mark creeps upward with each new peak (allocator retention, not a leak: `malloc_trim` returns it and the trimmed series is flat). In 12-minute CUDA soaks with the thread fix below, the default video run failed the soak's growth rule on the loaded series (316 MiB/h fitted); with `MALLOC_MMAP_THRESHOLD_=65536` the image and video footprints were about half as large and flat. Device memory is returned at every unload, and the cuDNN, cuBLAS and driver mappings do not grow.

In a container whose CPU quota is below the host's CPU count, the session now uses .NET's container-aware processor count for its intra-op threads when none is set (ONNX Runtime's own default started one thread, and one malloc arena, per physical core of the host). Evidence: [CUDA soak growth](https://github.com/starlinghq/starling/blob/main/docs/evidence/cuda-soak-growth/README.md).

## Status

Two profiles are Stable on Linux x64 CPU. Text (`embeddinggemma-2/onnx-fp32-cpu`) has been Stable since 2026-10-15, and images (`image-fp32-cpu`) since 2026-10-16. Each passed every qualification gate, G0 to G7. The weekly qualification workflow reruns their G1 to G3 on the release sets and a one-hour soak. For text, G2 reads the reference embeddings committed under `model-pipeline/protected/`. For images, G2 reads the accepted reference tensors hosted at [`Starlinghq/starling-release-references`](https://huggingface.co/datasets/Starlinghq/starling-release-references). Those are pinned by commit and SHA-256 in the release set's protected `hosted-reference.json`, and every array is checked against the set's `tensors.json`. Videos (`video-fp32-cpu`) are not promoted, although every gate G0 to G7 is recorded passed and the weekly run passes its release-set G1 and G2 the same way: its hosted one-hour soak failed the growth rule in every run up to 2026-10-16 (about 280 MB/h), and its release-set G2 fails on AVX-512 runners (token embeddings 1.8e-3 against a bound of 1e-3; it passes on AVX2), so it stays Experimental. The text profile's hosted soak failed twice on 2026-10-16 (about 130 MB/h). With ONNX Runtime's CPU arena off (below), the text and video soaks passed in three runs out of three ([evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/eg2-hosted-soak/README.md)). The image soak still failed intermittently, and the text soak's margin was thin (35.7 MiB of loaded growth against a 36.0 MiB limit): runs from rotating thread-pool threads left freed buffers resident in more and more glibc heaps. With dedicated run threads (below), the text, image and video soaks passed in three full qualification runs out of three on 2026-10-16, each soak twice per run (largest text loaded growth 9.4 MiB; [evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/eg2-soak-reliability/README.md)). On CUDA, text (`onnx-fp32-cuda`) and images (`image-fp32-cuda`) are Stable since 2026-10-16 on an NVIDIA GeForce RTX 3090 ([Stable contract: CUDA](#stable-contract-cuda)): every gate is recorded passed (G4 and G5 in the binding CUDA round, #74), and every check of both passed in the first per-release GPU run (38857860186). Videos on CUDA (`video-fp32-cuda`) stay Experimental: that run landed on an RTX 4090, outside the contract, and its video G2 failed there (embedding max abs 1.009e-5 against 1e-5; token embeddings 3.34e-3 against 1e-3), while the RTX 3090 measured 2.77e-6. A confirming run on an RTX 3090 is pending. GPU regression runs for every release (an `ai-v*` tag) and on demand, on an RTX 3090 only, in the qualification workflow's `cuda (runpod)` job; CPU regression runs weekly. Gate evidence: the [catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

Text (`onnx-fp32-cpu`): release sets of 232 tokenizer rows (ids exact) and 696 embedding rows (no prompt, `query` and `document`; within 5.1e-7 of the PyTorch reference, binding bound 5e-6), MIRACL nDCG@10 0.8325; development fixtures of 138 rows up to 1,750 tokens (within 3.7e-7).

Images (`image-fp32-cpu` and `image-fp32-cuda` stable), Linux x64, from decoded RGB (42 protected rows: 28 images alone, 6 interleaved or prompted, 8 text queries): rendered ids exact; resized bytes, pixel values and positions bit-exact on the 28 supported images; embeddings within 2.3e-6 (CPU) and 1.0e-6 (CUDA) of the PyTorch reference, binding bound 1e-5, min cosine 0.9999995. Audio: unsupported. `EmbeddingGemma2Capabilities.All` lists every capability with its status.

Videos (`video-fp32-cpu`, `video-fp32-cuda`), Linux x64, experimental, from decoded RGB frames (35 protected rows: 22 videos alone, 6 interleaved or prompted, 2 input forms, 5 text queries; synthetic H.264, HEVC, VP9 and AV1 clips of 0.5 to 40 s): sampled indices exact (and on 20,849 upstream cases), rendered ids exact; decoded frames (FFmpeg adapter), resized frames, pixel values and positions bit-exact on all 32 video inputs; embeddings within 1.7e-6 (CPU) and 3.3e-6 (CUDA) of the PyTorch reference, binding bound 1e-5, min cosine 0.9999995. Frames without timing: rejected by default (upstream behaviour on opt-in reproduces its row within 1.2e-7). Timestamps, other sampling settings and audio tracks: unsupported.

## Stable contract

This is the contract of profile `embeddinggemma-2/onnx-fp32-cpu` (text embeddings). Its Stable label (since 2026-10-15) commits to it under the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md). The statuses are the registry manifest's and `EmbeddingGemma2Capabilities.Profiles`, which a test keeps equal key for key. Images are the profile of the contract that follows; videos are the experimental `video-fp32-*` profiles.

| Item | Contract |
| --- | --- |
| Profile | `embeddinggemma-2/onnx-fp32-cpu` (registry manifest `model-pipeline/registry/embeddinggemma-2/onnx-fp32-cpu/manifest.yaml`). |
| Model | `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b`. The text graph is Starling's float32 export (`onnx/embeddinggemma2-text-fp32.onnx`, digest in `EmbeddingGemma2ModelAssets`), downloaded with no token from the public repository `Starlinghq/embeddinggemma-2-onnx` at `92a02ae2`. |
| Tokenizer | `Starlings.Tokenizers` `0.3.0`, profile `gemma-bpe-embeddinggemma2`. `<bos>` and `<eos>` are added, and text is never truncated. |
| Runtime | ONNX Runtime 1.30.0 (the application adds `Microsoft.ML.OnnxRuntime` 1.30.0 CPU natives), fp32, .NET 10. |
| Supported platform | Linux x64, CPU. Gates G1 to G5 ran on an Intel Core i7-5820K (Haswell-E, 6 cores and 12 threads, AVX2 and FMA, no AVX-512), .NET 10.0.12. The weekly qualification reruns G1 to G3 on the release sets and a one-hour soak on GitHub-hosted `ubuntu-latest` x64 (4 logical CPUs; transparent huge pages `madvise`). No `MALLOC_*` environment setting is required for this profile. |
| Memory on Linux | The runtime calls `malloc_trim(0)` when a session is released, on by default. CPU sessions run without ONNX Runtime's CPU memory arena unless `OnnxSessionOptions.CpuMemoryArena` is set, and on the runtime's dedicated run threads unless `OnnxSessionOptions.DedicatedRunThreads` is set; outputs are the same. See [Memory on Linux](#memory-on-linux). |
| Input | Text or token ids. Prompts are explicit (`EmbeddingGemma2Prompt`: none, `query` or `document`) and never inserted. |
| Output | 768-dimensional float32 unit vectors (mean pooling over every attended token, then L2), or a Matryoshka size of 512, 256 or 128 (truncate, then normalize again). Other sizes are rejected. |
| Numerical bound | The protected tolerances of the profile (embedding max abs 5e-6, min cosine 0.9999999, against the PyTorch reference). Release set: 5.1e-7. The release set's reference embeddings are committed under `model-pipeline/protected/` with the SHA-256 its `reference.json` records, so the weekly run evaluates G2 against the same file on any CPU. |
| Task quality | MIRACL dev, 8 languages: nDCG@10 at least 0.826 and MRR@10 at least 0.808 (budget); measured 0.8325 and 0.8136. |
| Supported | Embedding from text and from token ids; the owned tokenizer; no prompt and the `query` and `document` prompts; Matryoshka 512, 256 and 128; inputs up to 2,049 tokens; batch; cancellation; concurrent calls; explicit acquisition and anonymous graph download; Native AOT for the typed API and the Extensions.AI generator; `Starlings.Models.EmbeddingGemma2.Extensions.AI` (text generator, Matryoshka dimensions); `Starlings.Models.EmbeddingGemma2.MLNet` (text, row-wise and look-ahead batching). |
| Limits | 8,192 tokens, special and prompt tokens included; above that, `EmbeddingGemma2InputTooLongException` before inference. Inputs of 2,050 to 8,192 tokens run but are outside the Stable contract (below); temporary memory grows with the square of the length (one 8,192-token row took 80 s and a 5.9 GB process peak on the workstation). The text graph (1.08 GB) and `tokenizer.json` are acquired explicitly; inference never downloads. |
| Outside the Stable contract | Inputs of 2,050 to 8,192 tokens and the 18 other named task prompts: experimental (proposed rows, not yet accepted fixtures). CUDA: the `onnx-fp32-cuda` contract below. Images: the `image-fp32-cpu` contract below. Videos: the experimental `video-fp32-*` profiles. Windows x64 CPU: experimental (CI builds, runs the weight-free tests and the clean-install drill without the model; no model run). macOS and ARM64: unsupported (no evidence). |
| Unsupported | inputs over 8,192 tokens, truncation, implicit prompt insertion, other Matryoshka sizes, audio, media placeholder tokens in text, pair input, ML.NET persistence and cancellation, the ML.NET adapter under Native AOT, streaming, TF32, bf16, fp16, quantized graphs, TensorRT, training. |
| Known issues | On hosts with transparent huge pages set to `always`, resident memory can grow over time with no leak behind it ([Memory on Linux](#memory-on-linux)). The development tolerance file's `export.token_embeddings` bound (Python ONNX Runtime against PyTorch) reads differently judged per value (1.7e-2) and per case (6.2e-6, the reading the bound was set from); which reading binds is an open maintainer decision. Every .NET check passes either way, and the release set passed at 4.22e-6 relative. |

## Stable contract: images

This is the contract of profile `embeddinggemma-2/image-fp32-cpu` (image and interleaved text-and-image embeddings). Its Stable label (since 2026-10-16) commits to it under the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md). The statuses are the registry manifest's and `EmbeddingGemma2Capabilities.Profiles`, which a test keeps equal key for key.

| Item | Contract |
| --- | --- |
| Profile | `embeddinggemma-2/image-fp32-cpu` (registry manifest `model-pipeline/registry/embeddinggemma-2/image-fp32-cpu/manifest.yaml`). |
| Model | `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b`. The vision graph and the multimodal text graph are Starling's float32 exports (digests in `EmbeddingGemma2ModelAssets.ImageLock`), downloaded with no token from the public repository `Starlinghq/embeddinggemma-2-vision-onnx`. Text-only inputs run on the text graph of the text contract. |
| Tokenizer | `Starlings.Tokenizers` `0.3.0`, profile `gemma-bpe-embeddinggemma2`; each image is expanded to `<|image>` + 280 x `<|image|>` + `<image|>` at its position. |
| Runtime | ONNX Runtime 1.30.0 CPU natives, fp32, .NET 10. |
| Supported platform | Linux x64, CPU. Gates G1 to G5 ran on an Intel Core i7-5820K (Haswell-E, 6 cores and 12 threads, AVX2 and FMA, no AVX-512), .NET 10.0.12. The weekly qualification reruns G1 to G3 on the release sets and a one-hour soak on GitHub-hosted `ubuntu-latest` x64 (4 logical CPUs; transparent huge pages `madvise`). No `MALLOC_*` environment setting is required for this profile. |
| Memory on Linux | The runtime calls `malloc_trim(0)` when a session is released, on by default. CPU sessions run without ONNX Runtime's CPU memory arena unless `OnnxSessionOptions.CpuMemoryArena` is set, and on the runtime's dedicated run threads unless `OnnxSessionOptions.DedicatedRunThreads` is set; outputs are the same. See [Memory on Linux](#memory-on-linux). One full-budget image uses about 610 MB of attention buffer per vision layer; `MaxVisionBatchSize` is 1 by default. |
| Input | Decoded RGB images (`EmbeddingGemma2Image.FromRgb24`, or the SkiaSharp adapter for PNG and JPEG), alone or interleaved with text (`EmbeddingGemma2MultimodalInput`), several per input; text-only inputs; explicit prompts (none, `query`, `document`). Upstream's conversion rules apply: alpha dropped, grayscale replicated, palettes looked up, EXIF orientation not applied. |
| Output | 768-dimensional float32 unit vectors (mean pooling over every token, then L2), or a Matryoshka size of 512, 256 or 128. |
| Numerical bound | The protected tolerances of the profile: rendered ids, resized pixels, pixel values and positions exact; decoded RGB from the SkiaSharp adapter exact against PIL; embedding max abs 1e-5 and min cosine 0.9999995; soft tokens and token embeddings within 1e-3 row-relative; batches within 1e-5 of single inputs. Release set: embedding max abs 8.55e-7, soft tokens 8.68e-5 row-relative. The release set's reference tensors regenerate byte-identical only on the reference CPU, so the accepted arrays are hosted at `Starlinghq/starling-release-references` (commit and SHA-256 in the protected `hosted-reference.json`), and the evaluator checks every array against the protected `tensors.json`. |
| Task quality | `image-retrieval-v1` (149 Commons images and their descriptions): nDCG@10 at least 0.916 and MRR@10 at least 0.893 (budget); measured 0.9222 and 0.9008, equal to the upstream reference. |
| Supported | Images alone and interleaved with text; several images per input; the `query` and `document` prompts; the default 280-token budget; batch; Matryoshka 512, 256 and 128; SkiaSharp decoding of PNG and JPEG; text-only inputs; cancellation; concurrent calls; anonymous graph download; Native AOT for the typed API from decoded RGB; `Starlings.Models.EmbeddingGemma2.Extensions.AI` (the multimodal generator). |
| Limits | 50,000,000 decoded pixels per image by default (`MaxImagePixels`, a memory bound; `EmbeddingGemma2ImageTooLargeException`) and 8,192 tokens after expansion (about 29 images; `EmbeddingGemma2InputTooLongException`), both checked before inference. The accepted release set has one 79 MP image; it passes G1 and G2 only with the limit raised, which is outside the contract (below). |
| Outside the Stable contract | Experimental: images above the default pixel limit with the limit raised; budgets of 70, 140, 560 and 1,120 soft tokens; GIF, BMP and WebP through the SkiaSharp adapter; EXIF orientation on opt-in; the SkiaSharp adapter under Native AOT. CUDA (`image-fp32-cuda`): the contract below. Windows x64: untested. macOS and ARM64: unsupported. |
| Unsupported | 16-bit-per-channel images (rejected; upstream renders them white), alpha compositing, GIFs with transparent pixels (rejected), other image formats in the adapter, inputs over 8,192 tokens, media placeholder tokens in text, audio, video (the experimental video profiles), the ML.NET adapter (no image column), persistence, streaming, training. |
| Known issues | On hosts with transparent huge pages set to `always`, resident memory can grow over time with no leak behind it ([Memory on Linux](#memory-on-linux)). One CPU architecture (AVX2) produced the reference; the C# resampler is integer arithmetic and machine-independent. |

## Stable contract: CUDA

These are the contracts of profiles `embeddinggemma-2/onnx-fp32-cuda` (text) and `embeddinggemma-2/image-fp32-cuda` (images), Stable since 2026-10-16. Each is its CPU contract above, with these differences. The statuses are the registry manifests' and `EmbeddingGemma2Capabilities.Profiles`, which a test keeps equal key for key. Videos on CUDA (`video-fp32-cuda`) are not part of it (see Status).

| Item | Contract |
| --- | --- |
| Profiles | `embeddinggemma-2/onnx-fp32-cuda` and `embeddinggemma-2/image-fp32-cuda` (registry manifests under `model-pipeline/registry/embeddinggemma-2/`). The same models, graphs and tokenizer as the CPU profiles. |
| Runtime | ONNX Runtime 1.30.0 with the CUDA execution provider (the application adds `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 instead of the CPU natives), `use_tf32=0`, fp32. The CUDA 13 runtime and cuDNN 9 must be on the loader path (the vision graph's ReduceMax runs through cuDNN); the package does not ship them. |
| Supported platform | Linux x64 with an **NVIDIA GeForce RTX 3090** (Ampere), driver 580 or later. G1 to G3 on the release sets, the binding G4 one-hour soaks and G5 ran on RTX 3090s. **Regression:** the per-release GPU run (the qualification workflow's `cuda (runpod)` job, for every `ai-v*` tag and on demand) reruns the CUDA tests, G1 to G3 on the release sets and a soak smoke test, on an RTX 3090 only; the job refuses any other GPU. Other GPUs, including the RTX 4090 (Ada), are untested and outside the contract. |
| Memory on Linux | Text: no `MALLOC_*` setting is required. Images: `MALLOC_MMAP_THRESHOLD_=65536` is recommended, not required ([Image and video on CUDA](#image-and-video-on-cuda)); the G4 soak and the per-release soak smoke test ran without it. One full-budget image needs about 610 MB of GPU attention buffer per vision layer; the three graphs take about 2.9 GB of GPU memory. |
| Cancellation | A CUDA run is never terminated mid-graph: it completes, then the call throws `OperationCanceledException` (#61). Latency is one chunk: **up to about 1.2 s** for text at the default 16 rows of 2,048 tokens (lower `MaxBatchSize` for faster cancellation), and one vision or text run for images. |
| Numerical bound | Each CUDA profile's protected tolerances. Text release set: embedding max abs 8.6e-7 (bound 5e-6), C# against Python ONNX Runtime CUDA 2.5e-7. Image release set: embedding max abs 2.16e-6 (bound 1e-5), token embeddings 6.8e-5 row-relative (1e-3). |
| Task quality | Text, MIRACL: nDCG@10 0.8325, MRR@10 0.8136 (budget 0.826 and 0.808). Images, `image-retrieval-v1`: nDCG@10 0.9222, MRR@10 0.9008 (budget 0.916 and 0.893). |
| Performance | G5 on an RTX 3090: text warm latency 0.414 to 1.095 of the Python ONNX Runtime CUDA run (budget 1.10); images 0.785 to 1.029, preprocessing 1.2 to 3.8% of the embedding time (budget 5%). |
| Supported | As in the CPU contracts, over CUDA sessions: text embedding from text and token ids, prompts none, `query` and `document`, Matryoshka, inputs up to 2,049 tokens, batch, cancellation, concurrent calls, `Starlings.Models.EmbeddingGemma2.Extensions.AI` (text generator) and `Starlings.Models.EmbeddingGemma2.MLNet` over CUDA; images alone and interleaved, several per input, prompts, the 280-token budget, batch, Matryoshka, PNG and JPEG decoding, text-only inputs, cancellation, concurrent calls. |
| Outside the Stable contract | Experimental: Native AOT with the CUDA natives (one local run, `tools/aot-check-ai.sh --cuda`, not in the per-release run); the multimodal `IEmbeddingGenerator` over CUDA (its test needs the video set, which the per-release GPU pod does not have, so it is skipped there); and the rows that are experimental on CPU. Windows x64 CUDA: untested. |

Evidence: [binding CUDA G4 and G5 round](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-16-cuda/README.md), [per-release GPU regression, run 38857860186](https://github.com/starlinghq/starling/blob/main/docs/evidence/gpu-regression-2026-10-16/README.md), release-set gates for [text](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/embeddinggemma2-onnx-fp32-cuda/README.md) and [images](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/embeddinggemma2-image-fp32-cuda/README.md), [CUDA cancellation](https://github.com/starlinghq/starling/blob/main/docs/evidence/gliner2-cuda-wrong-output/README.md).

## Qualification evidence

- G0 provenance: the [2026-10-14 provenance check](https://github.com/starlinghq/starling/blob/main/docs/evidence/g0-provenance-2026-10-14/README.md).
- G1, G2 and G3 on the P-030 release sets: [text](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/embeddinggemma2-onnx-fp32-cpu/README.md), [images](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/embeddinggemma2-image-fp32-cpu/README.md), [videos](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/embeddinggemma2-video-fp32-cpu/README.md).
- G4 one-hour soak: [text, 2026-10-14](https://github.com/starlinghq/starling/blob/main/docs/evidence/g4-soak-1h-2026-10-14/README.md); images and videos in the 2026-10-15 round below. G5 matched benchmark: [2026-10-15](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-15/README.md).
- G6 distribution and G7 integration: [G6](https://github.com/starlinghq/starling/blob/main/docs/evidence/g6-distribution-2026-10-14/README.md), [G7](https://github.com/starlinghq/starling/blob/main/docs/evidence/g7-integration-2026-10-14/README.md).
- Regression support: the weekly [qualification workflow](https://github.com/starlinghq/starling/blob/main/.github/workflows/qualification.yml) (jobs `release-gates` and `soak`): text green in [run 38822625245](https://github.com/starlinghq/starling/actions/runs/38822625245); images and text green (release gates and soak) in [run 38848321624](https://github.com/starlinghq/starling/actions/runs/38848321624); the video soak failed the growth rule there (see Status). CUDA: per-release GPU regression [run 38857860186](https://github.com/starlinghq/starling/blob/main/docs/evidence/gpu-regression-2026-10-16/README.md) (the qualification workflow's `cuda (runpod)` job).

## License and usage policy

The package is MIT OR Apache-2.0; it includes a port of PyTorch's resize kernel (BSD-3-Clause, `THIRD-PARTY-NOTICES.md`). The model weights, and Starling's ONNX graphs derived from them, are licensed under the Apache License 2.0 by Google DeepMind ([model card](https://huggingface.co/google/embeddinggemma-2)). The model card also states:

> Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

The text graph contains only the text path (embedding, per-layer embeddings, 24 layers, final norm, 512-to-768 projection); the vision graph only the vision tower, pooling and `embed_vision`; the multimodal text graph is the text graph with two inputs for soft tokens. All are converted to float32 and exported to ONNX; the release notices list the modifications.
