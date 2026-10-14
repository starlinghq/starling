# Starlings.Models.EmbeddingGemma2

**Preview.** **Experimental: text, images, videos, and text with images and videos.** Embeddings from [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, on ONNX Runtime 1.30.0, in-process and offline after an explicit download. The API and the supported platforms may still change. Every registry profile is Experimental; see [Path to Stable](#path-to-stable).

- **Text** (`EmbeddingGemma2TextEmbedder`, `EmbeddingGemma2Encoder`): owned tokenizer or token ids, on the released text graph. The text encoder rejects the ids of the media placeholder and boundary tokens (`<|image|>`, `<|audio|>`, `<|video|>`, `<|image>`, `<image|>`, `<|audio>`, `<audio|>`).
- **Images** (`EmbeddingGemma2MultimodalEmbedder`, experimental): an image alone, or text and images interleaved, from **already-decoded RGB** (`EmbeddingGemma2Image`), default budget of 280 soft tokens per image. See [Images](#images).
- **Videos** (`EmbeddingGemma2MultimodalEmbedder`, `EmbeddingGemma2Video`, experimental): a video alone, or text, images and videos interleaved, from **already-decoded RGB frames**, at the checkpoint's settings (1 frame per second, at most 32 frames, 140 soft tokens per frame, no timestamps). Audio is not supported. See [Videos](#videos).
- Right-padded batching, mean pooling over every attended token (prompt and image tokens included), L2 normalization: 768-dimensional unit vectors, or a Matryoshka size of 512, 256 or 128 (truncate, then normalize again, as sentence-transformers `truncate_dim`). Other sizes are rejected.
- Prompts are explicit and never added for you: `EmbeddingGemma2Prompt.Query` (`"task: search result | query: "`) for queries, `EmbeddingGemma2Prompt.Document` (`"title: none | text: "`) for documents, any of the 20 names in the model's `config_sentence_transformers.json` through `EmbeddingGemma2Prompt.FromName`, or `None` (the default). The model card: "Prefixes apply to text only. Pass images, video, and audio without any prefix."
- Inputs above 8,192 tokens (the card's context window; special tokens, prompt and image tokens included) are rejected with `EmbeddingGemma2InputTooLongException`, never truncated.
- CUDA runs with TF32 disabled (`use_tf32=0`); with TF32 the text embeddings drift by up to 1.9e-4 and the image embeddings by up to 1.4e-3.
- The package ships the asset locks (repository, commit, sizes and SHA-256), not the model files (1.1 GB for text, 2.9 GB with images). Acquisition is an explicit call; inference never downloads.

## Tokenizer

Text goes through the owned tokenizer `Starlings.Tokenizers` (profile `gemma-bpe-embeddinggemma2`, exact against huggingface/tokenizers on the published fixtures): `EmbeddingGemma2Tokenizer` adds `<bos>` and `<eos>` and never truncates. No Microsoft.ML.Tokenizers and no native tokenizer library. Any other `IEmbeddingGemma2Tokenizer` must reproduce the pinned `tokenizer.json` exactly.

## Install

```sh
dotnet add package Starlings.Models.EmbeddingGemma2 --prerelease
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
- **Speed (indicative, i7-5820K 12 threads; RTX 4060 Ti).** Vision about 4.5 s per full-budget image on CPU, 0.23 s on CUDA; preprocessing 20 to 60 ms (210 ms for 4000x3000).

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
- **Per frame.** The image preprocessing at 140 soft tokens (1,260 patches; same resize kernel), the vision graph in calls of at most `MaxVideoFramesPerVisionCall` frames (default 2, GPU-safe; on CPU frame batches equal single frames bit for bit), and `<|image>` + n x `<|video|>` + `<image|>` per frame with no separator and no timestamps. A video costs `frames x (n + 2)` tokens, at most 4,544; inputs above 8,192 tokens are rejected before inference (two maximal videos do not fit).
- **Decoding.** FFmpeg's default conversion is what upstream embeds (10-bit and odd sizes included); other converters move the embedding by up to 1.7e-2. Use the FFmpeg adapter (exact against torchcodec on the fixtures) or an equivalent FFmpeg call (`ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgb24 -` plus torchcodec-style metadata).
- **Speed (indicative).** Vision about 2.7 s per frame on CPU (i7-5820K, 12 threads: a 32-frame video takes about 70 to 100 s) and about 0.07 s on CUDA (RTX 4060 Ti); prefer CUDA for video.

## Status

Profiles `embeddinggemma-2/onnx-fp32-cpu` and `embeddinggemma-2/onnx-fp32-cuda`: Linux x64 experimental, from text and from token ids (138 protected rows, 46 texts with no prompt, `query` and `document`, up to 1,750 tokens: ids exact, embeddings within 3.7e-7 of the PyTorch reference, binding bound 5e-6).

Profiles `embeddinggemma-2/image-fp32-cpu` and `embeddinggemma-2/image-fp32-cuda`: Linux x64 experimental, from decoded RGB (42 protected rows: 28 images alone, 6 interleaved or prompted, 8 text queries): rendered ids exact; resized bytes, pixel values and positions bit-exact on the 28 supported images; embeddings within 2.3e-6 (CPU) and 1.0e-6 (CUDA) of the PyTorch reference, binding bound 1e-5, min cosine 0.9999995. Image budgets other than 280, inputs of 2,050 to 8,192 tokens, the other 18 task prompts, Windows, macOS, ARM64 and Native AOT: untested. Audio: unsupported. `EmbeddingGemma2Capabilities.All` lists every capability with its status.

Profiles `embeddinggemma-2/video-fp32-cpu` and `embeddinggemma-2/video-fp32-cuda`: Linux x64 experimental, from decoded RGB frames (35 protected rows: 22 videos alone, 6 interleaved or prompted, 2 input forms, 5 text queries; synthetic H.264, HEVC, VP9 and AV1 clips of 0.5 to 40 s): sampled indices exact (and on 20,849 upstream cases), rendered ids exact; decoded frames (FFmpeg adapter), resized frames, pixel values and positions bit-exact on all 32 video inputs; embeddings within 1.7e-6 (CPU) and 3.3e-6 (CUDA) of the PyTorch reference, binding bound 1e-5, min cosine 0.9999995. Frames without timing: rejected by default (upstream behaviour on opt-in reproduces its row within 1.2e-7). Timestamps, other sampling settings and audio tracks: unsupported; variable frame rate, interlaced and HDR video: untested.

## Stable contract

This is the contract a Stable profile would commit to. Today all six profiles (`onnx-`, `image-` and `video-fp32-{cpu,cuda}`) are **Experimental**. The statuses below are the current ones, from the registry manifests and `EmbeddingGemma2Capabilities.All`; the [Status](#status) section has the numbers. Support levels are defined in the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md).

| Item | Contract |
| --- | --- |
| Model | `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b`. Text graph (release `embeddinggemma-2-onnx-r1`), vision graph and multimodal text graph (release `embeddinggemma-2-vision-onnx-r1`) are Starling's float32 exports, verified by SHA-256 (`EmbeddingGemma2ModelAssets`). |
| Tokenizer | `Starlings.Tokenizers` `0.3.0-preview.1`, profile `gemma-bpe-embeddinggemma2`. `<bos>` and `<eos>` are added, and text is never truncated. |
| Runtime | ONNX Runtime 1.30.0, fp32. CPU; CUDA with TF32 off (cuDNN 9 for the vision graph). |
| Input | Text or token ids; images as decoded 8-bit RGB (`EmbeddingGemma2Image`); videos as decoded RGB frames with timing (`EmbeddingGemma2Video`); interleavings of these. Prompts are explicit (`EmbeddingGemma2Prompt`) and never inserted. |
| Limits | 8,192 tokens after image and video expansion, special and prompt tokens included. Above that, `EmbeddingGemma2InputTooLongException` before inference. Images above 50 megapixels throw `EmbeddingGemma2ImageTooLargeException`. Videos: 1 frame per second, at most 32 frames, 140 soft tokens per frame. Frames without timing are rejected by default. |
| Output | 768-dimensional float32 unit vectors (mean pooling over every attended token, then L2), or a Matryoshka size of 512, 256 or 128 (truncate, then normalize again). Other sizes are rejected. |
| Numerical bound | The protected tolerances of each profile (text 5e-6, image and video 1e-5, min cosine 0.9999995, against the PyTorch reference). |
| Supported | None yet. |
| Experimental | text and pre-tokenized embedding, `query` and `document` prompts, Matryoshka 512/256/128, inputs up to 2,049 tokens, batch, cancellation, concurrent calls; images at the 280-token budget, interleaved and multiple per input, PNG and JPEG decoding through the SkiaSharp adapter, EXIF orientation opt-in; videos from decoded frames with timing or pre-sampled frames, interleaved, FFmpeg 8 decoding through the optional adapter, 90-degree rotation; `Starlings.Models.EmbeddingGemma2.Extensions.AI` and `.MLNet` (text) on CPU and CUDA; all 20 named prompts; inputs of 2,050 to 8,192 tokens; image budgets 70, 140, 560 and 1,120; GIF, BMP and WebP decoding; variable frame rate, interlaced and HDR video; rotations -90 and 180; Native AOT (CPU and CUDA natives). Linux x64. The newer items rest on proposed rows judged with the protected bounds ([capability closure](https://github.com/starlinghq/starling/blob/main/docs/evidence/capability-closure-2026-10-14/README.md)). |
| Untested | video decoding on ARM64 or without AVX2, graph download (releases not published), Windows, macOS, ARM64. |
| Unsupported | inputs over 8,192 tokens, GIF images with transparent pixels, truncation, implicit prompt insertion, other Matryoshka sizes, audio and audio tracks, media placeholder tokens in text, pair input, alpha compositing, 16-bit images, other image formats, video timestamps, other frame rates or budgets, FFmpeg 9 in the FFmpeg 8 adapter, ML.NET image and video columns, ML.NET persistence and cancellation, streaming, TF32, bf16, fp16, quantized graphs, TensorRT, training. |

## Path to Stable

The order of promotion (`docs/plan/production-readiness.md`) takes `onnx-fp32-cpu` (text) to Preview first, then the CUDA profiles once the protected evaluator has run on a GPU. Gate evidence is in the [catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md) once draft PR #4 lands. Agents produce evidence; only the maintainer records a gate or changes maturity.

- [ ] G0 provenance: today a checklist ([Hub pins](https://github.com/starlinghq/starling/blob/main/docs/m0/hub-pins.md), [image spike](https://github.com/starlinghq/starling/blob/main/docs/m0/embeddinggemma2-image-spike.md), [video spike](https://github.com/starlinghq/starling/blob/main/docs/m0/embeddinggemma2-video-spike.md)). Needs the graphs published as release assets (`embeddinggemma-2-onnx-r1`, `-vision-onnx-r1`; M3), and the D2 legal review before the FFmpeg adapter ships (M4).
- [ ] G1 and G2: `not-run` in the manifests. By kickoff decision, they enter the manifests only once the P-030 release sets exist (M1). On development fixtures the evaluator passed ([text](https://github.com/starlinghq/starling/blob/main/docs/evidence/embeddinggemma2-dotnet/README.md), [images](https://github.com/starlinghq/starling/blob/main/docs/evidence/embeddinggemma2-images/README.md), [video](https://github.com/starlinghq/starling/blob/main/docs/evidence/embeddinggemma2-video/README.md)). The rerun at tokenizer `0.3.0-preview.1` passed for text (3.7e-7, bound 5e-6) and image (2.24e-6, bound 1e-5) on CPU (draft PR #6). CUDA needs the evaluator's CUDA mode (R7, W2) and GPU time (M6).
- [ ] Long-input evidence for 2,050 to 8,192 tokens (R7, W3 fixture proposal), or a narrowed limit.
- [ ] G3 task quality: the retrieval evaluator (nDCG@10, recall@k) is in draft PR #5. It needs licensed text, image and video retrieval sets and budgets (M1, M2).
- [ ] G4 resource correctness: one-hour soak and the plan 10.2 tests (R1); the CUDA frame-call measurement (R7, W1).
- [ ] G5 performance: the text and image harness is in draft PR #7 (an uncontrolled run, not G5 evidence; no video yet). Needs a controlled run against a budget (M2, M6).
- [ ] G6 distribution: the release drill on a seeded graph is a checklist (draft PR #6). Needs acquisition from the published release assets (R9 after M3), and Windows x64 (R6, R9).
- [ ] G7 integration: the Extensions.AI and ML.NET adapters are tested on CPU. Needs a representative downstream use (M7).
- [ ] Every `untested` capability above: tested, or recorded as `unsupported`. The validator refuses Stable with any `untested` capability.
- [ ] Qualification CI (P-018, R6), the compatibility matrix rows, the support policy, and a second release owner (M7).

## License and usage policy

The package is MIT OR Apache-2.0; it includes a port of PyTorch's resize kernel (BSD-3-Clause, `THIRD-PARTY-NOTICES.md`). The model weights, and Starling's ONNX graphs derived from them, are licensed under the Apache License 2.0 by Google DeepMind ([model card](https://huggingface.co/google/embeddinggemma-2)). The model card also states:

> Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

The text graph contains only the text path (embedding, per-layer embeddings, 24 layers, final norm, 512-to-768 projection); the vision graph only the vision tower, pooling and `embed_vision`; the multimodal text graph is the text graph with two inputs for soft tokens. All are converted to float32 and exported to ONNX; the release notices list the modifications.
