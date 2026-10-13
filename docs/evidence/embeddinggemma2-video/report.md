# Fidelity report: embeddinggemma-2/video-fp32-cpu (video-embedding)

Date 2026-10-12. Evaluator model-pipeline evaluate 0.1.0 at `dd1ed7f3727aef5cb91c72b9954123f3f053e8f6`. Exit code 0.

First divergence: **none**.

## Gates

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every check passed or is not applicable by the manifest |
| G2 | passed | every check passed or is not applicable by the manifest |

## Stages

| Stage | Status |
| --- | --- |
| Stage 1, decoding, sampling, rendering and preprocessing (indices, frames, ids, pixels, positions) | passed |
| Stage 2, export (not applicable: Python ORT parity is protected evidence of its own) | passed |
| Stage 3, .NET from decoded frames (embeddings, soft tokens, token embeddings, batches) | passed |
| Stage 4 (not applicable: text-only rows run in stage 3 on the released text graph) | passed |

## Checks

Bounds come only from the protected tolerances file. Row-relative: per video (soft tokens) or row (token embeddings), max |d| / max |ref|.

| Check | Tolerance key | Bound | Max abs | Row-relative | Min cosine | Cases | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `tokenizer.assets` |  |  |  |  |  |  | passed |
| `dotnet-ids.assets` |  |  |  |  |  |  | passed |
| `tokenizer.token_ids` | discrete.token_ids | exact |  |  |  | 36 | passed |
| `tokenizer.frame_indices` | discrete.frame_indices | exact |  |  |  | 32 | passed |
| `tokenizer.frames_sha256` | discrete.decoded_frames | exact |  |  |  | 32 | passed |
| `tokenizer.resized_sha256` | discrete.resized_pixels | exact |  |  |  | 32 | passed |
| `tokenizer.pixel_values_sha256` | discrete.pixel_values | exact |  |  |  | 32 | passed |
| `tokenizer.positions_sha256` | discrete.video_positions | exact |  |  |  | 32 | passed |
| `tokenizer.image_rgb_sha256` | discrete.decoded_frames | exact |  |  |  | 1 | passed |
| `tokenizer.decoder_frame_sha256` | decoder_ffmpeg.decoded_rgb_vs_torchcodec | exact |  |  |  | 32 | passed |
| `dotnet-ids.embedding` | float.embedding_max_abs | 1e-05 | 1.69e-06 |  |  | 36 | passed |
| `dotnet-ids.min_cosine` | float.min_cosine | 0.9999995 |  |  | 0.999999999918 | 36 | passed |
| `dotnet-ids.soft_tokens` | float.soft_tokens_row_relative | 0.001 |  | 5.81e-05 |  | 32 | passed |
| `dotnet-ids.token_embeddings` | float.token_embeddings_row_relative | 0.001 |  | 0.00021 |  | 31 | passed |
| `dotnet-ids.finite` | float.nan_or_inf | fail |  |  |  |  | passed |
| `dotnet-ids.batch.frame-calls-2-text-batch-4` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 |  |  | 6 | passed |
| `dotnet-ids.batch.padded-1260-frame-calls-3` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 |  |  | 6 | passed |
| `dotnet-ids.provider_options` | provider_options | {} |  |  |  |  | passed |
| `export.not_applicable` |  |  |  |  |  |  | not-applicable |
| `dotnet-text.not_applicable` |  |  |  |  |  |  | not-applicable |

Details:

- `export.not_applicable`: the video tolerances bind the C# path; Python ORT parity: protected parity-cpu.json
- `dotnet-text.not_applicable`: text-only rows are stage 3 rows (released text graph)

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `e4562846b8f9bdec8f894724a1b5cfe4f9c2f6a6`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/video-fp32-cpu/tolerances.yaml` | `6b9de23d8d15703094ad93205066e4b05fa55b73ad2b21f1f699235cc0558172` |
| `model-pipeline/protected/embeddinggemma-2/video-fp32-cpu/video-embeddings.jsonl` | `1104697a73b3a25af9ff9a15df436e779ab78bddd3993567910d82c4bf22e01d` |
| `model-pipeline/protected/embeddinggemma-2/video-fp32-cpu/videos.json` | `36bcacbc4f88249cb7520e77826af4bba5e00e3d72f5fe3afafc9f16ed27e87c` |
| `model-pipeline/protected/embeddinggemma-2/video-fp32-cpu/tensors.json` | `715e5ed73e303de6039f66ba2f878c61c65d14264597f9ef0047caaf2709a61f` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | FFmpeg adapter on FFmpeg n8.1.3-17-gc389840894-20261011; EmbeddingGemma2VideoSampling; EmbeddingGemma2ImageProcessor at 140 soft tokens; EmbeddingGemma2MultimodalEmbedder rendering with EmbeddingGemma2Tokenizer | ai-dotnet/tools/Starlings.Fidelity.Tool `dd1ed7f3727aef5cb91c72b9954123f3f053e8f6` | omarchy | `2b6ff176719890f5` |
| dotnet | C# EmbeddingGemma2MultimodalEmbedder, ONNX Runtime 1.30.0 CPU (managed binding 1.30.0), frames from the FFmpeg adapter | ai-dotnet/tools/Starlings.Fidelity.Tool `dd1ed7f3727aef5cb91c72b9954123f3f053e8f6` | omarchy | `b85f771d45b9227f` |

Reference tensors: `/home/mark/Projects/AI-models/embeddinggemma2/video-spike/ref-clean-tensors` (each array checked against the protected tensors.json).

Excluded by contract (evaluated only when present in the results): `vid-array-no-metadata` (decoded frames without timing metadata (rejected by contract unless opted in))
