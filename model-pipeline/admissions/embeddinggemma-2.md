# Admission: EmbeddingGemma 2

Plan section 7 (priority P1, added 2026-10-10 at the maintainer's request) and pipeline stage 1. Facts below were read on 2026-10-10 from the Hugging Face API, from every file at the pinned revision (downloaded and hashed, `docs/m0/hub-pins.md`) and from the transformers 5.19.0 source. The admission spike ran the same day: `docs/m0/embeddinggemma2-spike.md`.

| Field | Value |
| --- | --- |
| Candidate | `google/embeddinggemma-2` (https://huggingface.co/google/embeddinggemma-2) |
| Code revision | transformers 5.19.0 (first release that defines `embedding_gemma2`; wheel SHA-256 `afcd2dd5...`), sentence-transformers 6.1.0; hashed in `model-pipeline/reference/embeddinggemma2/requirements.txt` |
| Weight revision | `914f7f89142e33e77833254d9c9b90c3cef7303b` (Hub `main` on 2026-10-10; created 2026-09-14, last modified 2026-10-06) |
| Requested by / demand evidence | Maintainer request, 2026-10-10. Hub: about 45.6K downloads and 1,505 likes in its first month (2026-10-10). Successor to `google/embeddinggemma-300m` (3.6M downloads). |
| Proposed owner | Unassigned |
| Proposed maturity at first release | Experimental, text embeddings only |
| Decision | Admitted at maturity Experimental, text embeddings only. Starling exports and hosts the text-path ONNX graph (GitHub release asset, as for GLiNER2) with the Apache-2.0 license and the card's Gemma Prohibited Use Policy sentence repeated in the package README and the asset notice. Inputs above 8,192 tokens (the card's validated context) are rejected. Slot: the next new architecture after P-016. Tolerances: `model-pipeline/protected/embeddinggemma-2/*/tolerances.yaml`. Update 2026-10-11 (Mark): contract extended to images (image-only and interleaved text and images, default 280-token vision budget, Experimental), then video. The model package takes already-decoded RGB pixels; file decoding is a separate SkiaSharp adapter (SkiaSharp approved; ImageSharp excluded: not OSI-approved, less accurate). Text-only input keeps the released text graph; the multimodal text graph is used only when images are present, after the 1e-3 text-only difference is explained. Upstream quirks: EXIF ignored by default with an opt-in, alpha dropped as upstream, 16-bit images rejected. The vision graph is hosted as a GitHub release asset. Image tolerances: `model-pipeline/protected/embeddinggemma-2/image-fp32-{cpu,cuda}/tolerances.yaml`. Update 2026-10-11 (Mark), video: video-only and interleaved text/image/video at Experimental with the checkpoint settings (1 fps, at most 32 frames, 140 tokens per frame, no timestamps); upstream quirks kept for parity (one frame under 2 s, index-based sampling, FFmpeg's default 10-bit conversion); frame arrays without timing metadata rejected by default with upstream behaviour as an explicit option; the model package takes decoded frames; an optional adapter on FFmpeg.AutoGen loads the user's own LGPL FFmpeg 8.x and ships no FFmpeg binaries (decoded RGB gated exact against torchcodec on FFmpeg 8 x86-64; other decoders reported); requirements-video.txt is the video reference environment. Video tolerances: `model-pipeline/protected/embeddinggemma-2/video-fp32-{cpu,cuda}/tolerances.yaml`. Update 2026-10-12 (Mark): a second optional video adapter for FFmpeg 9.x (separate package on FFmpeg.AutoGen 9, user's own LGPL FFmpeg, no binaries shipped), to be gated exact against torchcodec on FFmpeg 9; every adapter stays a voluntary package that no core package references (`docs/kickoff.md`, Optional adapters); FFmpeg adapters are held out of publishing until a legal review. |
| Decided by and date | Mark, 2026-10-10 (priority and admission) |

## Rights (hard blockers)

| Item | License or terms | Redistribution |
| --- | --- | --- |
| Code | transformers 5.19.0 and sentence-transformers 6.1.0 (Apache-2.0) | Yes |
| Weights | Apache-2.0: Hub `license` metadata, card front matter, and the card's license link (`ai.google.dev/gemma/docs/gemma_4_license`, the Apache License 2.0 text). No LICENSE file in the repository. Not gated. | Yes, with license and attribution. The card also says "Deployments must adhere to the Gemma Prohibited Use Policy"; how to pass that on is a maintainer decision (spike report, decision 2). |
| Tokenizer assets | Same repository, Apache-2.0 | Yes |
| Evaluation data | Not chosen. MTEB tasks are candidates; each dataset keeps its own license. | Per dataset |

Note: the earlier `google/embeddinggemma-300m` is under the Gemma license, not Apache-2.0. EmbeddingGemma 2's card and linked license text confirm Apache-2.0, so that blocker does not apply; the prohibited-use sentence is the remaining rights question.

## Score (provisional, before the spike; rescored after the spike: 4.35, see `docs/m0/embeddinggemma2-spike.md` section 9)

| Criterion | Weight | Score (0 to 5) | Notes |
| --- | --- | --- | --- |
| User demand | 25% | 5 | Maintainer priority; fast Hub uptake. |
| Backend feasibility | 20% | 3 | New architecture (`EmbeddingGemma2Model`); ONNX export of the text tower unproven. |
| Reusable architecture coverage | 20% | 4 | Reuses the e5 path (Runtime.Onnx, mean pooling, Normalize, adapters, acquisition); adds a Gemma-family backbone and tokenizer. |
| Access and licensing clarity | 15% | 4 | Apache-2.0, ungated; card text to confirm. |
| Testability | 10% | 4 | sentence-transformers reference; MTEB for task evaluation. |
| Maintenance burden (5 = low) | 10% | 3 | Multimodal towers and a new tokenizer family add surface. |
| **Weighted total** | | **3.95** | |

## Model facts (from `config.json` and modules at `914f7f89`)

- Architecture `EmbeddingGemma2Model`, `model_type` `embedding_gemma2`, dtype bfloat16. `model.safetensors` is 1,488,915,288 bytes with 744,371,992 parameters (all towers; the text path is 271,002,624 parameters, 1.08 GB in fp32). An earlier version of this record gave 744,371,512 as a byte count.
- Text tower (`embedding_gemma2_text`): 24 layers, hidden 512, intermediate 2048, 4 attention heads, 2 KV heads (1 on layers 5, 11, 17 and 23), head dim 256 (512 on the full-attention layers), sliding window 512 (`abs(q - k) <= 512`, up to 1,025 positions), `layer_types` sliding x5 then full, repeated four times (full attention on layers 5, 11, 17, 23), vocabulary 262,144, `max_position_embeddings` 262,144. Bidirectional attention on every layer; projection-only per-layer embeddings (PLE).
- Vision tower (`gemma4_vision`): 16 layers, hidden 768, patch 16. Audio tower (`gemma4_audio`): 12 layers, hidden 1024.
- sentence-transformers modules: Transformer, mean Pooling (`embedding_dimension` 768, `include_prompt` true), Normalize. The 512-to-768 projection is `EmbeddingGemma2TextModel.embedding_projection` (`Linear(512, 768, bias=False)`), applied per token after the final norm inside the text backbone, so the module's `token_embeddings` are already 768 wide and the projection belongs to the exported graph.
- Prompts (`config_sentence_transformers.json`, `default_prompt_name` null): `query` = `task: search result | query: `, `document` = `title: none | text: `, plus task prompts for question answering, fact checking, code retrieval, classification, clustering and sentence similarity. sentence-transformers renders text through `chat_template.jinja`, which yields the same ids as tokenizing `prompt + text`.
- Card: 8,192-token context window; Matryoshka dimensions 768, 512, 256, 128 (truncate and renormalize); bfloat16 or float32 only, never float16. `sentence_bert_config.json` sets no `max_seq_length`, so sentence-transformers does not truncate.

## Execution route

Planned: text tower plus pooling exported to ONNX and run on ONNX Runtime through the existing `Starlings.Runtime.Onnx` component, as for e5, CPU fp32 first, CUDA with `use_tf32=0`. Risks for the spike: export of the new attention layout (sliding window with per-layer KV-head counts), the 262k embedding table (about 134M parameters in fp32 for the text tower alone), and whether the 512-to-768 projection is part of the text path. Vision and audio inputs are out of the first contract (plan section 7 puts multimodal work later).

## Tokenizer profile

`tokenizer.json` model BPE with `byte_fallback` true, 262,144 vocabulary, 24 added tokens. Normalizer `Replace(" " -> "▁")`; pre-tokenizer `Split(" ", MergedWithPrevious)`; post-processor `TemplateProcessing` with `<bos>` and `<eos>`; decoder `Sequence[Replace("▁" -> " "), ByteFallback, Fuse]`. This is a SentencePiece-style BPE (Gemma family), not byte-level: a third owned profile after Unigram (e5, GLiNER2) and byte-level BPE (Qwen, in progress). It can reuse the BPE model being built for the Qwen profiles but needs `byte_fallback`, the string-pattern Split, and the ByteFallback and Fuse decoders.

## Feature contract (proposed)

| Upstream feature | In scope | Evidence source |
| --- | --- | --- |
| Text embedding (mean pooling, normalized, 768 dimensions) | Yes | `modules.json`, `1_Pooling/config.json` |
| Query and document prompts | Yes, explicit and never silent (as e5 prefixes) | `config_sentence_transformers.json`; fixtures with none, `query`, `document` |
| Matryoshka truncation to 512, 256, 128 | Experimental (card confirms; slice and renormalize equals sentence-transformers `truncate_dim`) | Card; spike probe |
| Image embeddings (image-only and interleaved with text) | Yes, Experimental, default 280-token budget (decided 2026-10-11; implemented in `EmbeddingGemma2MultimodalEmbedder`, profiles `image-fp32-cpu` and `image-fp32-cuda`) | `docs/m0/embeddinggemma2-image-spike.md`, `docs/evidence/embeddinggemma2-images/` |
| Video embeddings (video-only and interleaved with text and images) | Yes, Experimental, checkpoint settings: 1 fps, at most 32 frames, 140 tokens per frame, no timestamps (decided 2026-10-11; implemented in `EmbeddingGemma2MultimodalEmbedder` with `EmbeddingGemma2Video` and the optional adapter `Starlings.Models.EmbeddingGemma2.Video.FFmpeg`, profiles `video-fp32-cpu` and `video-fp32-cuda`) | `docs/m0/embeddinggemma2-video-spike.md`, `docs/evidence/embeddinggemma2-video/` |
| Audio embeddings | Not requested | `config.json` audio tower |
| Input up to 8,192 tokens | Experimental to 2,049 tokens (tested); 2,049 to 8,192 untested; longer input rejected | Card |

## Hardware and compute estimate

Text-tower spike on the workstation CPU (94 GB RAM) and RTX 4060 Ti. A few GB of disk for the snapshot and exported graphs.

## Spike plan (P-014 style)

1. Pin the transformers version that defines `embedding_gemma2`; download at `914f7f89` and record SHA-256 of every file in `docs/m0/hub-pins.md`.
2. Read the card for license text, prompts, Matryoshka dimensions and the supported context length.
3. Reference runner: sentence-transformers fp32 CPU on the M0 corpus plus prompts; fixtures for tokenizer (ids, offsets) and embeddings.
4. Export the text tower plus pooling to ONNX; check ORT CPU and CUDA parity, batch independence and NaN/Inf.
5. Owned Gemma BPE profile: conformance on the corpus.
6. Effort estimate and a go or no-go.

## Open risks

- New architecture; transformers and ONNX exporter support may lag the release.
- Third tokenizer family: added maintenance until the BPE work lands.
- Multimodal pressure: requests for image or audio embeddings need separate processor work and admission.

## Spike results (2026-10-10)

Full report: `docs/m0/embeddinggemma2-spike.md`. In short:

- Route: the text path (embedding, PLE, 24 layers, final norm, 512-to-768 projection) exports to one ONNX graph with the dynamo exporter (opset 20) with no repair; the TorchScript exporter also works with no repair. Mean pooling and normalization stay outside the graph, as for e5.
- Parity on 138 rows (46 corpus texts x no prompt, `query`, `document`; up to 1,750 tokens): final embedding max abs 4.2e-7 on ORT CPU (1 and 12 threads identical), 4.8e-7 on ORT CUDA with `use_tf32=0` (1.9e-4 with ORT's TF32 default), 3.7e-7 in C# (CPU) and 3.6e-7 (CUDA); min cosine 0.99999999999473. Padded batches and lengths across the sliding window (511 to 2,049 tokens) match single rows within 3e-7. No NaN/Inf.
- Tokenizer: transformers' live tokenizer equals `tokenizer.json`. The owned library at 0.1.0-preview.3 needs byte fallback and an unknown token in its BPE model, a string-pattern `Split` (`MergedWithPrevious`), and the `Sequence`, `Replace`, `ByteFallback` and `Fuse` decoders (report section 5).
- Estimate: 14 to 21 working days for a text-embedding preview with adapters. Rescored total 4.35. Recommendation: admit at Experimental, text only. Decisions for the maintainer: report section 10.

## Image-path spike (2026-10-11)

Requested by Mark on 2026-10-11 (extend the contract to images first, then video; audio not requested). Full report: `docs/m0/embeddinggemma2-image-spike.md`. In short:

- Flow: the processor that runs upstream is the torchvision backend (`convert("RGB")`, uint8 antialiased bicubic resize to the largest 48-multiple size within 2,520 patches, x 1/255, 16x16 patches); the vision tower (16 layers), 3x3 pooling to at most 280 soft tokens, `embed_vision` (RMSNorm, 768 to 512); the soft tokens replace the token embedding at `<|image|>` positions of `<|image>` + n x `<|image|>` + `<image|>`; mean pooling over every token. EXIF orientation is ignored and alpha dropped upstream; 16-bit images become white.
- Route: a vision graph (dynamo, no repair; TorchScript also works; 0.67 GB) and a multimodal variant of the released text graph made by rewriting it (two inputs and one `Where`, same weights; 1.08 GB). ORT CPU on 43 rows: embedding max abs 2.3e-6, min cosine 0.99999995; CUDA `use_tf32=0` 1.5e-6 (TF32 on: 1.4e-3). Vision batches and unpadded patches equal single padded images bit for bit on CPU.
- C# preprocessing: a port of PyTorch's uint8 antialias kernel reproduces the reference's resized pixels bit for bit on all 29 test images; pixel tensors exact; C# end to end 1.5e-6. One or two levels of resampling difference (Pillow vs torchvision) move the embedding by up to 2.6e-3, so the port is required.
- Decoding: SkiaSharp (MIT; bundled permissive natives, one MPL 1.1 tri-licensed GIF decoder) matches PIL's decoded RGB exactly on every supported test image, JPEG included; StbImageSharp differs on JPEG (embedding 1.6e-3); ImageSharp 4 is Split-licensed (not OSI) and needs a license key for Release builds. Proposed: model package on pre-decoded RGB, decoding in a separate SkiaSharp adapter package.
- Estimate: 14 to 21 working days for an image-embedding preview. Proposed fixtures: stacked pull request (`model-pipeline/protected/embeddinggemma-2/image-fp32-cpu/`, `PROPOSED.md`). Decisions for the maintainer: report section 12.

## Video-path spike (2026-10-11)

Requested by Mark on 2026-10-11 (images first, then video). Full report: `docs/m0/embeddinggemma2-video-spike.md`. In short:

- Flow: upstream decodes with torchcodec (FFmpeg); metadata from a packet scan; sampling by frame index at 1 fps from the average frame rate, `max(1, int(duration))` candidates thinned to 32 with `np.linspace(..., dtype=int)`; each frame resized with the image rule at 140 tokens (1,260 patches) by the same torchvision kernel; per frame `<|image>` + n x `<|video|>` + `<image|>`, no timestamps; soft tokens at `<|video|>` (258884). Quirks: one frame for clips under 2 s; decoded arrays without metadata keep all frames (different embedding); `.webm` path strings are text in sentence-transformers.
- Graphs: the released vision and multimodal text graphs take frames unchanged. ORT CPU embedding max abs 1.7e-6, CUDA `use_tf32=0` 1.1e-6 (TF32 on 4.9e-4), 36 rows on 22 synthetic videos (H.264, HEVC, VP9, AV1; 0.5 to 40 s; rotation, 10-bit, odd sizes).
- C#: sampling port exact on 20,849 cases; the existing image resampler gives the reference's frame pixels exactly; owned tokenizer exact; end to end 1.7e-6 (CPU) and 4.5e-7 (CUDA).
- Decoding: a C# decoder on FFmpeg.AutoGen 8.1.0 (MIT bindings) with user-supplied LGPL FFmpeg reproduces torchcodec's RGB bit for bit on all 22 videos (so do the `ffmpeg` CLI and torchcodec on FFmpeg 9). Other conversions (PyAV, OpenCV, a managed converter) move the embedding by 5e-3 to 1.7e-2. No managed .NET decoder exists for these codecs. Proposed: model package on decoded frames; optional FFmpeg adapter package shipping no FFmpeg binaries.
- Estimate: 11 to 17 working days. Proposed fixtures: stacked pull request (`model-pipeline/protected/embeddinggemma-2/video-fp32-cpu/`, `PROPOSED.md`). Decisions for the maintainer: report section 11.
