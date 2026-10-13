# EmbeddingGemma 2 image embeddings: implementation evidence

Contract of 2026-10-11 (Mark; `model-pipeline/admissions/embeddinggemma-2.md`, Decision update): image-only and interleaved text-and-image embeddings, default 280-token vision budget, Experimental; decoded RGB in the model package, decoding in a separate SkiaSharp adapter; text-only input keeps the released text graph. Binding tolerances: `model-pipeline/protected/embeddinggemma-2/image-fp32-{cpu,cuda}/tolerances.yaml`; accepted fixtures: `image-fp32-cpu/` (PR #42). Spike: `docs/m0/embeddinggemma2-image-spike.md`.

One workstation: Intel i7-5820K (12 threads, AVX2), 94 GB RAM, Linux x64, .NET 10, ONNX Runtime 1.30.0; CUDA on GPU 1 only (RTX 4060 Ti, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, CUDA 13 and cuDNN 9 from the nvidia wheels; GPU 0 belongs to another workload and was not used). Development evidence; the manifests' gate entries are left to the maintainer.

## Files

| File | Contents |
| --- | --- |
| `release-graphs.json` | Two runs of `release_embeddinggemma2_vision_onnx.py`: sizes, SHA-256, node counts, ORT-vs-PyTorch check of the vision graph, text-only identity of the multimodal graph; `byte_identical: true` |
| `text-only-investigation-cpu.json` | `investigate_embeddinggemma2_mm_text_only.py`: the spike's multimodal graph and the release layout against the released text graph at every ORT optimization level, and the fused nodes that differ |
| `csharp-tests-cpu.txt`, `csharp-tests-cuda.txt` | Output of the C# image test classes (CPU: `ImageInputTests`, `ImageParityTests`; CUDA: `CudaImageParityTests`), from the TRX reports |
| `report.md`, `report.json` | `model-pipeline evaluate` on the CPU stage results, unedited |
| `release-drill-excerpt.txt` | The EmbeddingGemma 2 steps of `tools/release-drill-ai.sh --no-model --seed-embeddinggemma2-from <dir>` |

## Graphs

| Graph | Bytes | SHA-256 | Made by |
| --- | ---: | --- | --- |
| `onnx/embeddinggemma2-vision-fp32.onnx` | 671,256,072 | `6c97f477a82f6863d22d800476de6d0a618b23e514aa9b65a46b220c4b106b92` | dynamo export of `eg2_vision.VisionPath`, opset 20, normalized, weights inline (the same bytes as the spike's graph) |
| `onnx/embeddinggemma2-text-mm-fp32.onnx` | 1,084,327,898 | `a2c22e072ec230b105b35c2787dcfccba186254e6a921e729c45b40e50dbbb74` | the released text graph (`02cc7ecc...`) rewritten by `eg2_vision.add_multimodal_inputs_release` |

Two independent runs gave byte-identical files. Release tag `embeddinggemma-2-vision-onnx-r1` (one tag for both), notice `model-pipeline/reference/embeddinggemma2/NOTICE-embeddinggemma2-vision-onnx.txt` (Apache-2.0, the card's prohibited-use sentence verbatim, the modifications); `EmbeddingGemma2ModelAssets.ImageModelSourceIsPlaceholder` is true until the maintainer creates the release.

## The ~1e-3 text-only difference of the spike's multimodal graph

| Graph vs released text graph, text-only input | ORT_DISABLE_ALL | ORT_ENABLE_BASIC | ORT_ENABLE_EXTENDED | ORT_ENABLE_ALL | CUDA (ALL, `use_tf32=0`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spike layout (one `Where` after `Mul(embedding, embed_scale)`) | 0 | 0 | 3.0e-3 | 3.0e-3 | 4.6e-3 |
| Release layout (`add_multimodal_inputs_release`) | 0 | 0 | 0 | 0 | 0 |

Token embeddings of the 8 text queries and a random 2x300 batch; token embeddings reach about 2,000, so 3e-3 is 1.5e-6 relative. Cause: ONNX Runtime's `MatMulScaleFusion` (extended level). In the released graph the per-layer-embedding (PLE) projection is `MatMul(embedding * sqrt(512), W) * (1/sqrt(512))`, which ORT turns into one `FusedMatMul(embedding, W, alpha = 0.99999994)`. With the spike's `Where` between `Mul(embedding, sqrt(512))` and the `MatMul`, ORT can fold only the second scalar, `FusedMatMul(where_output, W, alpha = 0.0442)`, so the PLE input is rounded differently. It is not a dtype or mask issue: with optimizations off or at the basic level the two graphs are identical, and the only fused-node attribute difference after optimization is that `alpha`. The release layout keeps every released node's inputs (so ORT fuses the released nodes exactly as before) and adds the soft tokens' own PLE projection with the same weight, mixed in by a second `Where` after the projection; text-only input is now bit-identical to the released graph on CPU and CUDA, and image rows stay within 2.2e-6 of the reference in Python ORT (the spike graph: 2.3e-6).

## Parity of the C# path (42 in-contract rows)

From SkiaSharp-decoded fixture images (decoded RGB equal to PIL's, checked by digest) through the C# preprocessing, the vision graph and the multimodal text graph; text-only rows on the released text graph. `mode-gray16` is rejected by contract (upstream renders the 16-bit image white).

| Check | CPU (12 threads) | CUDA (`use_tf32=0`) | Bound |
| --- | --- | --- | --- |
| Rendered ids (owned tokenizer on the expanded placeholders), 43 rows | exact | exact (same code) | exact |
| Decoded RGB vs PIL, 28 images | exact | | exact |
| Resized bytes, pixel values, positions, 28 images | exact | | exact |
| Soft tokens, row-relative (worst) | 1.57e-4 (photo-camera) | 2.28e-5 (photo-coffee) | 1e-3 |
| Token embeddings, row-relative (worst) | 4.54e-5 (photo-rocket) | 7.48e-5 | 1e-3 |
| Embedding max abs | 2.24e-6 (photo-hubble-deep-field) | 9.88e-7 (photo-rocket) | 1e-5 |
| Embedding min cosine (float64) | 0.99999999989 | 0.99999999997 | 0.9999995 |
| Text-only rows vs reference | 1.2e-7 | 8.9e-8 | 1e-5 |
| Text-only rows: multimodal embedder vs `EmbeddingGemma2TextEmbedder` | identical | identical | |
| Batches vs single rows: vision batch 8 / padded to 2,520 | 0 / 0 | 1.3e-7 (vision batch 1, text batch 4) | 1e-5 |
| Matryoshka 256 vs truncated 768 | within 1e-5 | | 1e-5 |
| Retrieval sanity: each of 8 queries ranks its image first | 8/8 | 8/8 | |
| NaN/Inf | none | none | none |

Weight-free checks in CI: the resampler equals torchvision on 15 cases (downscaling up to 9x, upscaling, one-axis, thin, 1x1), the target size equals transformers on 120 size/budget cases, patches and positions equal the processor, the multimodal embedder's plumbing on fake graphs (soft-token order, batching, padding, limits, routing), the Extensions.AI image generator contract, and the SkiaSharp adapter's rules on hand-made PNGs (16-bit rejection, alpha dropped, gray replicated, header-checked pixel limit, truncated and unsupported data, all 8 EXIF orientations against Pillow).

## Staged evaluator

`model-pipeline evaluate model-pipeline/registry/embeddinggemma-2/image-fp32-cpu/manifest.yaml --results <dir> --out <dir> --reference-tensors <ref-clean-tensors> --require-gates G1,G2`, stage results from `ai-dotnet/tools/Starlings.Fidelity.Tool -- embeddinggemma2-images`, both at `712e5cc` with the protected evidence verified against Git: **exit 0, first divergence none, G1 passed, G2 passed** (`report.md`). Stage 1: 42 rows of ids and 35 images of each digest exact. Stage 3: embedding max abs 2.24e-6, min cosine 0.999999999893, soft tokens 1.57e-4 and token embeddings 4.54e-5 row-relative, both batch layouts 0 against single rows, provider options `{}`. The reference tensors are the spike's `.npz` files (631 MB, outside Git), every array checked against the protected `tensors.json`. CUDA has no evaluator run (the fidelity tool references the CPU natives); the CUDA numbers above come from the test project.

## Release drill

`tools/release-drill-ai.sh --no-model --seed-embeddinggemma2-from <dir>` (the three graphs and the six photos): DRILL PASS. The imaging adapter packs with exactly `Starlings.Models.EmbeddingGemma2`, `SkiaSharp [4.153.1]` and `SkiaSharp.NativeAssets.Linux.NoDependencies [4.153.1]`; no restore graph holds ImageSharp; SkiaSharp appears only in the EmbeddingGemma2Search sample. Offline, from packages only: the text search as before, then the image search ("a photo of a cat" ranks `chelsea.png` first, coffee, rocket, astronaut and galaxies likewise), the Extensions.AI image generator equal to the typed API (max abs 0), and an interleaved caption-plus-image input.

## Timing (indicative; shared workstation)

Vision about 4.5 s per full-budget image on CPU (12 threads), 0.2 s on CUDA; preprocessing 20 to 60 ms per image (210 ms for 4000x3000); the 42-row CPU parity run takes about 3 minutes, the CUDA one 40 s including session loads.

## Not done

- Onboarding pipeline: no recipe. `model-pipeline/ONBOARDING.md` has one execution plugin (`e5-onnx-cpu-v1`), which deliberately rejects other profiles; EmbeddingGemma 2 images need a second plugin (assets against `ImageLock`, the image reference runner, the vision release step, the fidelity tool's image dump, and the image evaluator task added here with `--reference-tensors`). The evaluator task is the prerequisite; the plugin is follow-up work.
- ML.NET image column: not done (`mlnet.image_column` unsupported); the transform stays text-only.
- Budgets other than 280, GIF/BMP/WebP decoding, Windows/macOS/ARM64: untested. Video: next item; the preprocessing is frame-ready (140-token budget, same resize).
- The CUDA evaluator stage, and an Extensions.AI image run over a CUDA session.
