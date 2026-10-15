# Fidelity report: embeddinggemma-2/video-fp32-cuda (video-embedding)

Date 2026-10-15. Evaluator model-pipeline evaluate 0.1.0 at `a8717f6203b50b7539f326bca6c2834bd1419959`. Exit code 0.

Fixtures: `model-pipeline/protected/embeddinggemma-2/release/video-fp32-release` (from the --fixtures option).

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
| `tokenizer.token_ids` | discrete.token_ids | exact |  |  |  | 11 | passed |
| `tokenizer.frame_indices` | discrete.frame_indices | exact |  |  |  | 12 | passed |
| `tokenizer.frames_sha256` | discrete.decoded_frames | exact |  |  |  | 12 | passed |
| `tokenizer.resized_sha256` | discrete.resized_pixels | exact |  |  |  | 12 | passed |
| `tokenizer.pixel_values_sha256` | discrete.pixel_values | exact |  |  |  | 12 | passed |
| `tokenizer.positions_sha256` | discrete.video_positions | exact |  |  |  | 12 | passed |
| `tokenizer.image_rgb_sha256` | discrete.decoded_frames | exact |  |  |  | 1 | passed |
| `tokenizer.decoder_frame_sha256` | decoder_ffmpeg.decoded_rgb_vs_torchcodec | exact |  |  |  | 12 | passed |
| `dotnet-ids.embedding` | float.embedding_max_abs | 1e-05 | 2.77e-06 |  |  | 11 | passed |
| `dotnet-ids.min_cosine` | float.min_cosine | 0.9999995 |  |  | 0.999999999818 | 11 | passed |
| `dotnet-ids.soft_tokens` | float.soft_tokens_row_relative | 0.001 |  | 0.00025 |  | 12 | passed |
| `dotnet-ids.token_embeddings` | float.token_embeddings_row_relative | 0.001 |  | 0.000797 |  | 11 | passed |
| `dotnet-ids.finite` | float.nan_or_inf | fail |  |  |  |  | passed |
| `dotnet-ids.batch.frame-calls-2-text-batch-4` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 |  |  | 1 | passed |
| `dotnet-ids.batch.padded-1260-frame-calls-3` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 4.64e-07 |  |  | 1 | passed |
| `dotnet-ids.provider_options` | provider_options | {'use_tf32': 0} |  |  |  |  | passed |
| `export.not_applicable` |  |  |  |  |  |  | not-applicable |
| `dotnet-text.not_applicable` |  |  |  |  |  |  | not-applicable |

Details:

- `export.not_applicable`: the video tolerances bind the C# path; Python ORT parity: protected parity-cpu.json
- `dotnet-text.not_applicable`: text-only rows are stage 3 rows (released text graph)

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `fd229fa69cb19847f41d410ecdb9e3e253a550b2`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/video-fp32-cuda/tolerances.yaml` | `afcddf0bc61c55e671c465d406378ade8f1362eb7df6f579367d0d8731c9a208` |
| `model-pipeline/protected/embeddinggemma-2/release/video-fp32-release/video-embeddings.jsonl` | `a9c69aaa2bf91df5497ba2572bb43cef93c1b20415a22186eb39756a8e49d1ab` |
| `model-pipeline/protected/embeddinggemma-2/release/video-fp32-release/videos.json` | `71f059885ccde317b7c27731b596243fd5c4597531f6c116e64f7a6ae4caac0c` |
| `model-pipeline/protected/embeddinggemma-2/release/video-fp32-release/tensors.json` | `9d2ec884fbb34a0213e14b8fd12ee661703af5f29da62367d15e7b34d36082fe` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | FFmpeg adapter on FFmpeg n8.1.3-17-gc389840894-20261011; EmbeddingGemma2VideoSampling; EmbeddingGemma2ImageProcessor at 140 soft tokens; EmbeddingGemma2MultimodalEmbedder rendering with EmbeddingGemma2Tokenizer | ai-dotnet/tools/Starlings.Fidelity.Tool `a8717f6203b50b7539f326bca6c2834bd1419959` | omarchy | `e0b0c29c58c669dc` |
| dotnet | C# EmbeddingGemma2MultimodalEmbedder, ONNX Runtime 1.30.0 CUDA (managed binding 1.30.0), frames from the FFmpeg adapter | ai-dotnet/tools/Starlings.Fidelity.Tool `a8717f6203b50b7539f326bca6c2834bd1419959` | omarchy | `321a3ae6e0d294e5` |

Reference tensors: `/home/mark/.cache/starling-release-refs/eg2-video-t6/tensors` (each array checked against the protected tensors.json).

Excluded by contract (evaluated only when present in the results): `vid-array-no-metadata` (decoded frames without timing metadata (rejected by contract unless opted in))
