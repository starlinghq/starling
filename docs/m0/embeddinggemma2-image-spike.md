# EmbeddingGemma 2: image-path spike

Status: spike result, 2026-10-11. Brief: maintainer approval (Mark, 2026-10-11) to extend the EmbeddingGemma 2 contract to images first, then video; audio not requested. Template: the text spike (`docs/m0/embeddinggemma2-spike.md`) and the admission record (`model-pipeline/admissions/embeddinggemma-2.md`). Upstream code, the model card and repository files were read as data.

Pins (unchanged from the text spike): checkpoint `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b`; transformers 5.19.0, sentence-transformers 6.1.0, torch 2.14.1 (CPU build; torchvision 0.29.1), Pillow 12.3.0, ONNX Runtime 1.30.0, onnx 1.23.1, onnxscript 0.7.2; environments `model-pipeline/reference/embeddinggemma2/requirements.txt` and `requirements-cuda.txt` (already list Pillow and torchvision, so no new environment). Photos: scikit-image repository at commit `b700edc8005e4ee9e74fc83259f09e8df3213f15`, each file checked by SHA-256.

Contract under test: image-only and interleaved text-and-image embeddings (mean pooling over every token, L2 normalization, 768 dimensions), default vision budget (280 soft tokens per image).

## 1. Summary

| Question | Answer |
| --- | --- |
| How do images reach the embedding? | Processor (torchvision backend, not PIL): `convert("RGB")`, uint8 antialiased bicubic resize to the largest 48-multiple size within 2,520 patches, x 1/255, 16x16 patches; vision tower (16 layers); 3x3 average pooling to at most 280 soft tokens, x sqrt(768); RMSNorm and `Linear(768, 512)`; the soft tokens replace the token embedding at the `<|image|>` positions of `<|image>` + n x `<|image|>` + `<image|>`; the text model runs on the mixed sequence; mean pooling over every token. Section 2. |
| Graph route | Two graphs. A vision graph (pixels and patch positions to soft tokens), exported with the dynamo exporter with no repair (TorchScript also works). The released text graph cannot take soft tokens (ids only), so a variant adds two inputs and one `Where` node by rewriting the released graph (no re-export, same weights). Section 4. |
| ORT CPU parity (43 rows: 29 images, 6 interleaved or prompted, 8 text queries) | Embedding max abs 2.3e-6, min cosine 0.99999995 (1 and 12 threads identical; TorchScript graph the same). Soft tokens 1.6e-4 of the row's largest value; padded vs unpadded patches bit-identical; vision batches of 8 images of different sizes equal single images bit for bit; padded text batches 1.2e-7 from single rows. No NaN/Inf. |
| ORT CUDA (GPU 1, `use_tf32=0`) | Embedding 1.5e-6, min cosine 0.99999995. With ORT's TF32 default: 1.4e-3 and 0.99995 (seven times the text path's TF32 error). Pin `use_tf32=0`. |
| C# preprocessing | A 200-line C# port of PyTorch's uint8 antialias kernel gives the reference's resized pixels **bit for bit on all 29 images** (and a Pillow-quantized variant matches Pillow bit for bit). Pixel tensors equal the reference exactly; C# end to end from PIL-decoded RGB: embedding 2.3e-6 on CPU and 1.5e-6 on CUDA, the same worst values as Python ONNX Runtime (section 5). |
| Sensitivity | One or two levels of resampling difference on under 1% of pixels (Pillow vs torchvision, both "bicubic") moves the embedding by up to 2.6e-3 (cosine 0.99987). An approximate resampler would fail any embedding bound near the text path's. |
| Decoder | SkiaSharp (MIT, native) decodes every test image to the same RGB bytes as PIL, JPEG included (both use libjpeg-turbo). StbImageSharp (public domain/MIT, managed) differs by up to 3 levels on JPEG (embedding up to 1.6e-3); ImageSharp 4.1.3 differs by up to 52 levels on JPEG (embedding up to 7.1e-3) and needs a Six Labors license key to build in Release (Split License, not OSI). Recommend: core API on pre-decoded RGB (exact), SkiaSharp decoding in a separate adapter package. Section 6. |
| Upstream quirks to decide | EXIF orientation is ignored upstream; RGBA is not composited (alpha dropped); 16-bit images become white (PIL `convert("RGB")` saturates). Section 2.4. |
| Estimate | 14 to 21 working days for an image-embedding preview (section 9). |

## 2. How images flow upstream (transformers 5.19.0, sentence-transformers 6.1.0)

### 2.1 Input rendering (sentence-transformers)

`SentenceTransformer.encode` converts every input to the chat-message format (`modules.json` Transformer with `modality_config.message`, `format: structured`) and renders it with `chat_template.jinja`:

- A prompt (`prompt_name` or `prompt`) becomes a system message, rendered first with no separator, exactly as for text.
- An image alone renders as `<|image|>`. A dict `{"text": ..., "image": [...]}` renders the text; if the text contains `<|image|>` (or `<|video|>`, `<|audio|>`) those are the image positions, in order; otherwise each image is appended as `<|image|>` in the dict's key order (`{"text", "image"}` puts the image after the text, `{"image", "text"}` before). Observed: `{"text": "A cat on a blanket.", "image": img}` renders `A cat on a blanket.<|image|>` with no space.
- `EmbeddingGemma2Processor` expands each `<|image|>` to `<|image>` (255999) + n x `<|image|>` (258880) + `<image|>` (258882), where n is the image's soft-token count (section 2.2), then the tokenizer adds `<bos>` and `<eos>`. Example: an image of 451x300 is `<bos><|image><|image|>x260<image|><eos>`, 264 tokens. Text around a placeholder tokenizes as usual (`... blanket. <|image|>` gives a lone `▁` token before `<|image>`).
- The processor checks that the placeholder count equals the number of images per sample.
- Card guidance: "Prefixes apply to text only. Pass images, video, and audio without any prefix." A prompt with an image still works mechanically (system text before the image tokens); the fixtures include `query` and `document` prompts with images to pin the rendering, not to recommend it.

### 2.2 Image preprocessing (`Gemma4ImageProcessor`, torchvision backend)

`processor_config.json` names `Gemma4ImageProcessor`; in transformers 5 that class is the torchvision backend (`backend == "torchvision"`, checked at run time). The PIL backend (`Gemma4ImageProcessorPil`) exists but is not what upstream runs. Steps per image:

1. `convert_to_rgb`: PIL `image.convert("RGB")` unless the mode is already RGB, then `pil_to_tensor` (uint8, [3, H, W]). No EXIF transposition (section 2.4).
2. Target size (`get_aspect_ratio_preserving_size`, `patch_size` 16, `pooling_kernel_size` 3, `max_soft_tokens` 280, so `max_patches` 2,520 and a pixel budget of 645,120): `factor = sqrt(645120 / (H * W))`; each side `floor(factor * side / 48) * 48`. If one side rounds to 0 it becomes 48 and the other `min(floor(long / short) * 48, 13,440)`. Upscaling is allowed (a 1x1 image becomes 768x768). The image is left untouched when the target equals its size. Examples: 451x300 to 960x624 (260 soft tokens), 512x512 to 768x768 (256), 4000x3000 to 912x672 (266), 2000x40 to 5664x96 (236), 4000x8 to 13,440x48 (280).
3. Resize: `torchvision.transforms.v2.functional.resize(uint8 tensor, bicubic, antialias=True)`, which on CPU runs PyTorch's native uint8 antialias kernel (separable, horizontal then vertical, Keys cubic with a = -0.5, support 2 x downscale factor, int16 fixed-point weights with a per-pass precision, uint8 intermediate). This is not Pillow's resize: Pillow uses the same filter with 22-bit weights; the two differ by 1 or 2 levels on up to 0.8% of bytes.
4. Rescale: `uint8 * 0.00392156862745098` in float32 (bit-equal to `float32(v) * float32(1/255)`, not to `v / 255f`). No normalization (`do_normalize` false); the model applies `2 * (x - 0.5)` itself.
5. Patches: [H/16 * W/16, 768], row-major over the patch grid, each patch flattened as (row in patch, column in patch, channel). Positions (x, y) per patch. Padded with zero rows and (-1, -1) positions to 2,520 rows, always (also for one image).

### 2.3 Vision tower, pooling, projection, merge (modeling code)

- Patch embedder: `Linear(768, 768, bias=False)` on `2 * (pixel - 0.5)`, plus learned x and y position embeddings (two tables of 10,240 x 768), none for padding rows.
- Encoder (`gemma4_vision`): 16 bidirectional layers, hidden 768, 12 heads of 64, Q/K RMSNorm with scale and V RMSNorm without, 2-D axial RoPE (theta 100, x and y halves), attention scaling 1.0, gated GELU-tanh MLP 3,072; padding rows masked as keys. No clipped linears in this checkpoint.
- Pooler: padding rows zeroed; every 3x3 block of patches averaged (block index `x // 3 + (W/48) * (y // 3)`), times sqrt(768) in float32; `standardize` is false. Output 280 slots, of which the first `(H/48) * (W/48)` are valid (row-major block order); upstream strips the rest by boolean indexing.
- `embed_vision`: RMSNorm without scale (eps 1e-6) then `Linear(768, 512, bias=False)`: the soft tokens.
- Merge: `inputs_embeds = embed_tokens(ids) * sqrt(512)` with media ids replaced by pad, then `masked_scatter` puts the soft tokens (unscaled) at the `<|image|>` positions. The text model then computes the per-layer embeddings (PLE) from this mixed sequence, runs the 24 layers, final norm and the 512-to-768 projection.
- Pooling (sentence-transformers): mean over every token of the attention mask, `<bos>`, prompt, `<|image>`, the soft-token positions, `<image|>` and `<eos>` included (`include_prompt` true; the message format does not track prompt length), then L2 normalization. Matryoshka truncation is unchanged.

### 2.4 Upstream behaviour that a C# contract has to choose about

| Case | Upstream (observed) | Note |
| --- | --- | --- |
| EXIF orientation | Ignored. A JPEG stored rotated with Orientation 6 gives the same pixel tensor whether passed as a path or as a PIL image (sentence-transformers opens paths itself without `exif_transpose`). | Users may expect upright images. Matching upstream means ignoring the tag. |
| RGBA, LA, palette with transparency | Alpha dropped by `convert("RGB")`; fully transparent pixels keep their stored colour. No compositing on white or black. | |
| Grayscale, 1-bit, palette | Replicated or looked up by PIL. | Decoders agree exactly. |
| CMYK JPEG | PIL's CMYK-to-RGB conversion. | SkiaSharp agrees exactly on the test image. |
| 16-bit grayscale PNG (`I;16`) | `convert("RGB")` saturates: every value at or above 255 becomes 255, so a normal 16-bit image becomes white. | Every C# decoder scales 16 to 8 bits instead (cosine 0.68 against the upstream embedding). Propose: reject, or document as a deliberate deviation. |
| Very small or extreme aspect ratio | Accepted (1x1, 4000x8); the short side becomes 48 pixels. | |
| Image larger than the budget | Downscaled; no size limit upstream. | C# needs a decoded-pixel limit (memory). |

### 2.5 Reference stability (`image-reference.json`)

Padded batches of 8 images vs single images 2.3e-7; eager vs sdpa attention 2.4e-7; text-and-vision load (`audio_config` None, the card's "text and image" setting) vs full load 0; `encode()` end to end vs the module forward 0; restated vision path (`eg2_vision.VisionPath`) vs upstream: pooled soft tokens 0, projected soft tokens 1.5e-6 (padded) and 8.0e-5 (unpadded, in PyTorch); no NaN/Inf. Retrieval sanity: each of 8 text queries (`query` prompt) ranks its intended image first among 8 candidates (cosine 0.68 to 0.76 vs at most 0.66 for the runner-up).

## 3. Reference runner and image set

- `make_eg2_images.py`: 29 images. Six photos fetched from scikit-image at the pinned commit with SHA-256 checks (chelsea, coffee and camera CC0; astronaut, rocket and Hubble deep field public domain, as stated by scikit-image), 13 synthetic images from a seed (gradient, shapes, a text render with Pillow's built-in bitmap font, a 2-pixel checkerboard, noise, 1x1, 8x6, an image already at its target size, 4000x3000, 2000x40, 30x1500, 4000x8) and 10 derived images (grayscale, RGBA with a transparent band, LA, adaptive palette, palette with a transparent index, 1-bit, 16-bit grayscale, CMYK JPEG, progressive 4:2:0 JPEG, 4:4:4 JPEG, EXIF-rotated JPEG). Nothing is committed; `images.json` records source, license, file and decoded-pixel digests.
- `run_embeddinggemma2_image.py`: sentence-transformers fp32 CPU on PIL images (`Image.open`), one input per call: 29 image-only rows, 6 interleaved or prompted rows (placeholder caption, two images in one input, text without placeholder, `document` prompt with text and image, image with `query`, image with `document`), 8 text queries. It restates the processor step by step and stops if any pixel or position differs (none did), records per image the resized size, soft-token count and SHA-256 of every intermediate (resized uint8 image, pixel values, positions, encoder output, pooled soft tokens, projected soft tokens), writes the tensors to `.npz` files outside git (631 MB), and measures the PIL backend against the torchvision backend.
- `eg2_vision.py`: the exportable restatement and the text-graph rewrite.

## 4. Export and parity

Scripts: `export_embeddinggemma2_vision_onnx.py`, `parity_embeddinggemma2_image_ort.py`. Graphs stay under `/home/mark/Projects/AI-models/embeddinggemma2/image-spike/onnx/`; none is committed.

### 4.1 Graph boundaries

| Option | Verdict |
| --- | --- |
| Reuse the released text graph as is | Not possible: it takes `input_ids` and gathers the embedding inside; ONNX Runtime cannot feed an internal tensor. |
| One combined graph (pixels and ids in, token embeddings out) | Possible, but a fixed number of images per graph call or dynamic scatter inside the graph, 1.75 GB in one file, and text-only users pay for the vision weights. Not recommended. |
| **Vision graph + multimodal text graph (recommended)** | Vision graph per image or batch of images; C# scatters the soft tokens; the text graph takes them as an extra input. Same split as upstream (`get_image_features`, then the language model). Video and audio reuse the same text inputs. |

The multimodal text graph is the released graph (`embeddinggemma2-text-fp32.onnx`, SHA-256 `02cc7ecc...`) rewritten by `eg2_vision.add_multimodal_inputs`: inputs `multimodal_embeds` (float32, [batch, sequence, 512]) and `multimodal_mask` (bool, [batch, sequence]); one `Unsqueeze` and one `Where` after `Mul(embedding, embed_scale)`; every consumer of the scaled embedding (first layer norm, PLE projection, residual) rewired to the `Where` output. 2,603 nodes (2,601 + 2), 1,084,327,666 bytes, SHA-256 `dcbc95e9...`. A text-only call passes an all-false mask: it equals the released graph to 1e-3 abs on the 8 text queries and 7e-3 abs on random ids (token embeddings of about 2,000; ORT fuses the two graphs slightly differently, so not bit for bit); both graphs are within 2e-4 of PyTorch element-wise (relative to `max(1, |ref|)`) on the text queries. Because the text graph's release asset is still a placeholder (`EmbeddingGemma2ModelAssets.ModelSourceIsPlaceholder`), the multimodal graph could replace it as the one text asset (decision 3).

### 4.2 Vision export

| Graph | Exporter | Result | Nodes | Size |
| --- | --- | --- | --- | --- |
| `embeddinggemma2-vision-dynamo-fp32.onnx` | dynamo, opset 20; dynamic batch and patches (`9 * slots`, slots 1 to 1,120) | exported, no repair, 53 s; stripped of metadata and saved as one file | 1,760 | 671,256,072 bytes, SHA-256 `6c97f477...` |
| `embeddinggemma2-vision-legacy-fp32.onnx` | TorchScript, opset 18 | exported, no repair (tracer warnings on Python booleans that are constant here), 14 s | 5,675 | 671,886,837 bytes |

Outputs: `soft_tokens` ([batch, patches/9, 512]), `pooled` ([batch, patches/9, 768], diagnostic), `soft_token_mask` (bool). The restatement replaces upstream's `int(sqrt(...))` kernel size and `one_hot` with the fixed kernel 3 and an equality against `arange`, keeps fixed-size outputs, and uses an additive float mask (no NaN for padding rows). Neither failure category of the stop rule occurred.

### 4.3 Parity (PyTorch fp32 CPU reference; 43 rows)

Token embeddings reach about 2,000 to 3,900 and pooled soft tokens about 23,000, so element-wise errors are given relative to the row's largest value ("row-relative"). The element-wise measure relative to `max(1, |ref|)` used for the text path is large here (up to 0.12) because small elements of large rows carry the row's absolute error; it is reported but not proposed as a gate.

| Candidate | Pooled soft tokens, row-relative | Soft tokens (512) max abs / row-relative | Token embeddings row-relative | Embedding max abs | Min cosine | Batches | NaN/Inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ORT CPU, dynamo, 12 threads | 3.3e-5 | 9.1e-4 / 1.6e-4 | 4.5e-5 | 2.3e-6 | 0.99999995 | vision batch of 8 = single (0); text batch of 6 vs single 1.2e-7 | none |
| ORT CPU, dynamo, 1 thread | identical to 12 threads (every worst value equal) | | | | | | none |
| ORT CPU, TorchScript graph, 12 threads | 3.2e-5 | 8.6e-4 / 1.5e-4 | 3.6e-5 | 2.3e-6 | 0.99999993 | same | none |
| ORT CUDA, dynamo, `use_tf32=0` | 2.8e-5 | 1.5e-4 / 2.3e-5 | 1.2e-4 | 1.5e-6 | 0.99999995 | text batch 2.4e-7; vision batch not run (GPU memory) | none |
| ORT CUDA, dynamo, `use_tf32=1` | 4.0e-2 | 1.71 / 0.29 | 2.6e-2 | 1.4e-3 | 0.99995 | text batch 5.9e-5 | none |

- Worst rows are photos with fine texture (Hubble deep field 2.3e-6, rocket 1.3e-6); synthetic and small images stay under 4e-7; text queries under 1.2e-7.
- Unpadded vision input (`9 * count` patches instead of 2,520) gives bit-identical soft tokens on ORT CPU (3.4e-5 on CUDA): C# can skip the padding, which cuts vision cost for wide or tall images (236 soft tokens instead of 280 slots) and allows batching images of similar size without 2,520-row padding.
- GPU memory: one 2,520-patch image needs a 610 MB attention buffer per layer on ORT CUDA; a batch of two failed with about 5.5 GB free (another process holds 10.8 GB of GPU 1). Batching on GPU needs a memory-aware batch size.
- Timing (indicative; shared machine): vision 4.5 to 5 s per image on ORT CPU (12 threads, i7-5820K), 0.23 s on CUDA (RTX 4060 Ti); text graph 0.4 s (CPU) and 0.03 s (CUDA) per 264-token row. Session load 2.2 s (CPU) and 4.9 s (CUDA) for both graphs.

## 5. Image preprocessing in C# (prototype)

Throwaway console project outside the repository (`/home/mark/Projects/AI-models/embeddinggemma2/csharp-image`, .NET 10, ONNX Runtime 1.30.0): target size, resampler, rescale, patches and positions, the two graphs, scatter, pooling.

### 5.1 Resampler

Two fixed-point quantizations of one separable antialiased bicubic resampler (about 200 lines): PyTorch's (int16 weights, precision `p` = the largest value with `round(max_weight * 2^(p+1)) < 2^15`, computed per pass; rounding bias `2^(p-1)`; arithmetic shift; saturate to 0..255) and Pillow's (22-bit weights). Inputs are PIL's decoded RGB bytes.

| Comparison (29 images) | Result |
| --- | --- |
| C# PyTorch-quantized vs the reference's resized uint8 image (torchvision backend) | **exact on 29 of 29** (upscale, downscale by up to 4.4x, one-axis resize, extreme aspect, no-resize) |
| C# Pillow-quantized vs Pillow `resize(BICUBIC)` | exact on 29 of 29 |
| Pillow vs torchvision (the two transformers backends) | up to 2 levels on up to 0.8% of bytes; embedding up to 2.6e-3, cosine down to 0.99987 (8x6 noise upscaled); median about 3e-4 |
| C# pixel tensors (rescale by `float32(1/255)`, patch order) vs the reference | exact on every image of every row |

The kernel choice matters more than its precision: the reference's arithmetic is reproducible, but only by porting it, not by using a library "bicubic". The port depends on PyTorch's CPU uint8 path; on an AVX2 machine the AVX kernel and on others the generic kernel use the same int16 weights, but the reference was produced only on AVX2 (`torch_cpu_capability` in `image-environment.json`).

### 5.2 End to end in C#

| C# run (43 rows) | Pixel values | Embedding max abs | Min cosine |
| --- | --- | --- | --- |
| CUDA (`use_tf32=0`), PIL-decoded RGB input | exact | 1.5e-6 | 0.99999996 |
| CPU (12 threads), PIL-decoded RGB input | exact | 2.3e-6 | 0.99999997 |
| CPU, PIL-decoded RGB input, unpadded patches | exact | 2.3e-6 | 0.99999997 |
| CPU, SkiaSharp decoding | exact except the 16-bit PNG | 2.3e-6 on the other 42 rows | 0.99999997 (0.68 on `mode-gray16`) |
| CPU, ImageSharp 4.1.3 decoding (Debug build) | differs on every JPEG | 7.1e-3 | 0.9984 (0.68 on `mode-gray16`) |
| CUDA, SkiaSharp decoding | exact except the 16-bit PNG (section 2.4) | 1.5e-6 on the other 42 rows | 0.99999996 (0.68 on `mode-gray16`) |
| CUDA, StbImageSharp decoding | differs on every JPEG | 1.6e-3 | 0.99993 (0.68 on `mode-gray16`) |

## 6. Decoder options and licenses

| Library | License | Native code | Fidelity vs PIL (29 images) | Notes |
| --- | --- | --- | --- | --- |
| **SkiaSharp 4.153.1** (+ `SkiaSharp.NativeAssets.Linux.NoDependencies` on Linux) | MIT (package metadata). The native library's `THIRD-PARTY-NOTICES.txt` lists Skia, ANGLE, libjpeg-turbo, libwebp, sfntly (BSD-style), libpng, zlib, expat, HarfBuzz (old MIT), FreeType (FTL), ICU (ICU license), jsoncpp, imgui, SDL, SPIR-V (permissive), the Adobe DNG SDK and piex (vendor licenses that permit redistribution with notice), and a GIF decoder tri-licensed MPL 1.1 / GPL 2.0 / LGPL 2.1 (usable under MPL 1.1, file-level copyleft on unmodified upstream files). No GPL-only part. The notices file must ship with any package that brings the natives; the DNG and MPL entries deserve a maintainer read | yes, per RID (win, osx, linux x64/arm/arm64, musl, bionic); `libSkiaSharp.so` 12.2 MB on linux-x64 | exact on every image except 16-bit (where PIL is the outlier); JPEG baseline, progressive, 4:2:0, 4:4:4, CMYK, EXIF image all exact | Decodes to unpremultiplied RGBA with no colour-space conversion when asked; exposes the EXIF origin (`EncodedOrigin`) if orientation is ever wanted. Decode time 3 to 10 ms for small images, 215 ms for 4000x3000. |
| StbImageSharp 2.30.16 | Unlicense OR MIT | none (managed port of stb_image) | PNG exact; JPEG differs by up to 3 levels on 1 to 5% of bytes (own IDCT and chroma upsampling) | Embedding error up to 1.6e-3 on JPEG. Fine for a "best effort" decoder, not for parity. |
| SixLabors.ImageSharp 4.1.3 | Six Labors Split License 1.0 (Apache-2.0 only for open-source, transitive, small-company or non-profit use; otherwise a commercial license); not OSI-approved | none | PNG exact; JPEG differs by up to 52 levels (progressive 4:2:0) | Its build targets refuse a Release build without a license key (`SixLaborsLicenseKey`); a Debug build only warns. Not acceptable as a dependency of a Starlings package. |
| Magick.NET 15.0.0 | Apache-2.0 (ImageMagick license for the native part) | yes, large (tens of MB per platform) | not tested | Too heavy for a decode step. |
| System.Drawing.Common | MIT | GDI+ | not tested | Windows-only since .NET 7; excluded by the brief. |

Recommendation: the model package takes already-decoded RGB (width, height, bytes, row stride): exact parity with the reference, no native dependency, and callers who already decode (ASP.NET, MAUI, WPF, Skia apps) pay nothing. Decoding from a file or stream lives in a separate package (`Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` or a neutral `Starlings.Imaging.SkiaSharp`), which reproduces PIL's `convert("RGB")` rules (alpha dropped, palette looked up, CMYK as Skia does, 16-bit rejected) and has its own looser tolerance profile (section 8). StbImageSharp is a possible managed fallback with a documented JPEG tolerance, not recommended for the first release.

## 7. Proposed contract

| Feature | Status proposed | Evidence |
| --- | --- | --- |
| Image embedding from pre-decoded RGB (8-bit, 3 channels), default budget 280 soft tokens | experimental | sections 4 and 5: pixel tensors exact, embedding 2.3e-6 (CPU), 1.5e-6 (CUDA) |
| Interleaved text and images (`<\|image\|>` placeholders in text, or images appended after the text) | experimental | 4 interleaved rows, two-image row; the C# side must expand placeholders before tokenizing (owned tokenizer on the expanded string untested) |
| Prompts with images | experimental for the rendering (system text first); card recommends no prompt for images | `img-query-prompt`, `img-document-prompt`, `mm-document-prompt` |
| Image decoding (PNG, JPEG baseline/progressive, grayscale, palette, RGBA/LA alpha dropped, CMYK) through the SkiaSharp adapter | experimental with the decoder profile | section 6 |
| EXIF orientation | upstream ignores it; propose ignored by default (parity), opt-in `ApplyExifOrientation` marked as a deviation | section 2.4 |
| 16-bit-per-channel images | unsupported (rejected): upstream turns them white | section 2.4 |
| Vision budgets 70, 140, 560, 1,120 soft tokens | untested (the card allows them; same graph, other patch budgets) | |
| Images per input | limited by the 8,192-token context (about 29 images at 280 tokens); inputs over 8,192 tokens rejected | card |
| Decoded size limit | propose rejecting images above 50 megapixels before resizing (memory), configurable | none upstream |
| Batch | experimental (vision batch independence bit-exact on CPU; GPU batch size limited by memory) | section 4.3 |
| CUDA | experimental with `use_tf32: 0` pinned; TF32 unsupported | section 4.3 |
| Video, audio | unsupported in this step (video is the next spike) | |

## 8. Proposed tolerances (values only; no tolerances file)

For the maintainer to set (plan 10.1). Two profiles: the binding one starts from pre-decoded RGB, the decoder one is a convenience layer.

| Key | Proposed bound | Observed worst (CPU / CUDA TF32 off / C#) |
| --- | --- | --- |
| Rendered ids (placeholder expansion, soft-token count) | exact | exact |
| Resized uint8 image and pixel values from pre-decoded RGB | exact (SHA-256 of the fixture digests) | exact / exact / exact |
| Patch positions | exact | exact |
| Soft tokens (512), row-relative `max|d| / max|ref|` | 1e-3 | 1.6e-4 / 2.3e-5 / not separately |
| Pooled soft tokens (768) | diagnostic only | 3.3e-5 row-relative |
| Token embeddings (768), row-relative | 1e-3 | 4.5e-5 / 1.2e-4 / not compared |
| Embedding max abs, image and interleaved rows | 1e-5 | 2.3e-6 / 1.5e-6 / 1.5e-6 |
| Embedding min cosine | >= 0.9999995 | 0.99999995 / 0.99999995 / 0.99999996 |
| Batched vs single (vision batch, padded text batch), embedding max abs | 1e-5 | 1.2e-7 / 2.4e-7 / not run |
| Unpadded vs padded vision input, soft tokens | same as soft tokens | 0 / 3.4e-5 |
| NaN/Inf | none | none |
| Provider options | CUDA `use_tf32: 0` pinned | TF32 on: 1.4e-3, fails |
| Decoder profile (SkiaSharp): decoded RGB vs PIL | exact for PNG and JPEG in the fixture set; embedding bound as above | exact on 28 of 28 supported images |
| Decoder profile, other decoders (if ever offered) | embedding max abs 5e-3, min cosine 0.9995, per decoder, reported not gated | stb 1.6e-3 / 0.99993 |

The embedding bound is twice the text profile's (5e-6) because the vision tower adds error that the worst photos carry to 2.3e-6; it keeps four times headroom over the observed worst.

## 9. Implementation estimate

| Work item | Days |
| --- | --- |
| Release step for the multimodal text graph (rewrite script, stripped single file, digest; replace or add to the text asset) and the vision graph (export, strip, digest, reproducibility check of two exports, hosting 0.67 GB) | 2 to 3 |
| Preprocessing in the model package: target size, PyTorch-quantized resampler (vectorized), rescale, patches, positions; RGB input type with stride; size limits; unit tests on the fixture digests | 2 to 3 |
| Encoder: vision session, unpadded or padded batching with a memory-aware GPU batch size, soft-token scatter, placeholder expansion and tokenization of interleaved input, 8,192-token budget, CUDA profile | 3 to 4 |
| Decoder adapter package (SkiaSharp): PIL conversion rules, 16-bit rejection, EXIF option, notices for the bundled natives, tests on the fixture images | 2 to 3 |
| Fixture and evaluation plumbing: image rows and pixel digests in `model-pipeline evaluate` (embedding task with image inputs), C# stage results | 2 to 3 |
| Extensions.AI (`DataContent` images in `IEmbeddingGenerator`) and ML.NET (image column) adapters | 2 to 3 |
| Capability statuses, README, sample, acquisition lock for two assets | 1 to 2 |
| **Total** | **14 to 21** |

## 10. Timing of the C# prototype (indicative)

Scalar C# preprocessing from decoded RGB (target size, resize, rescale, patches; Release, one thread): 19 to 58 ms for images up to 1000x872, 210 ms for 4000x3000. Decoding with SkiaSharp adds 3 to 10 ms (small images) and 215 ms (4000x3000). Vision graph per image: 4.3 to 4.8 s on ORT CPU (12 threads), 3.5 to 3.9 s unpadded for the 236-token wide and tall images, 0.23 s on CUDA. Text graph: 0.3 to 0.6 s (CPU) and 0.03 to 0.08 s (CUDA) per row of 240 to 284 tokens. The vision tower dominates on CPU; preprocessing is not a bottleneck, and the resampler can be vectorized later without changing results (integer arithmetic).

## 11. What carries over to video

- Same vision graph: the video processor (`EmbeddingGemma2VideoProcessor`) uses the same aspect-preserving size rule with `max_soft_tokens` 140 (1,260 patches), the same torchvision uint8 antialiased bicubic resize, rescale and patching; `get_video_features` runs the same vision tower and `embed_vision`. The exported graph already accepts any multiple of 9 patches.
- Same C# resampler, rescale and patch code; same multimodal text graph (`multimodal_mask` is set at `<|video|>` positions, id 258884).
- New for video: frame sampling (1 fps from the video's frame rate and duration, at most 32 frames, `overflow_strategy` uniform), placeholder expansion per frame (`<|image>` + 140 x `<|video|>` + `<image|>` per frame, no timestamps by default), and above all video decoding in .NET (FFmpeg bindings bring LGPL/GPL questions; an API on pre-decoded frames avoids them, as for images).
- Budget: 140 tokens per frame, so 32 frames take 4,480 + frame markers of the 8,192-token context.

## 12. Decisions for the maintainer

1. Contract scope: image-only and interleaved text and images at Experimental, default budget only (section 7).
2. API boundary: pre-decoded RGB in the model package (exact) with decoding in a separate SkiaSharp adapter package, or decoding in the model package.
3. Text graph asset: replace the (not yet released) text graph with the multimodal variant as the single text asset (text-only calls pass an all-false mask), or ship both (2.2 GB for multimodal users).
4. Upstream quirks: EXIF orientation ignored by default (parity) with an opt-in, RGBA alpha dropped as upstream, 16-bit images rejected.
5. Decoder license: accept SkiaSharp (MIT, native, bundled permissive natives); exclude ImageSharp 4 (Split License, build-time license key).
6. Tolerances: set or change the values in section 8; accept the proposed fixtures (stacked pull request, `protected-change` label).
7. Vision hosting: Starling hosts the 0.67 GB vision graph next to the text graph (Apache-2.0 permits it with notice).

## 13. Not done

- No product code: no `ai-dotnet` project change, no evaluator change (another change is editing `evaluate.py`).
- Vision budgets other than 280 were not run; no video or audio.
- No PyTorch CUDA reference; no bfloat16 profile; no MTEB/MIEB task-quality run.
- The owned tokenizer was not run on the expanded placeholder strings (the C# runs fed the reference ids).
- One CPU (AVX2) for the reference; a non-AVX2 machine might run PyTorch's generic uint8 kernel (same weights expected, not verified).
- Two exports of the vision graph were not compared for a reproducible digest.
- Timing is indicative only (P-008 protocol not run).
