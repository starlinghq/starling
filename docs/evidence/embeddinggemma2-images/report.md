# Fidelity report: embeddinggemma-2/image-fp32-cpu (image-embedding)

Date 2026-10-12. Evaluator model-pipeline evaluate 0.1.0 at `712e5cc1c7d673cc481238da47d0f05ed1f2f8b8`. Exit code 0.

First divergence: **none**.

## Gates

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every check passed or is not applicable by the manifest |
| G2 | passed | every check passed or is not applicable by the manifest |

## Stages

| Stage | Status |
| --- | --- |
| Stage 1, rendering and preprocessing from decoded RGB (ids, resized bytes, pixel values, positions) | passed |
| Stage 2, export (not applicable: Python ORT parity is protected evidence of its own) | passed |
| Stage 3, .NET from decoded RGB (embeddings, soft tokens, token embeddings, batches) | passed |
| Stage 4 (not applicable: text-only rows run in stage 3 on the released text graph) | passed |

## Checks

Bounds come only from the protected tolerances file. Row-relative: per image (soft tokens) or row (token embeddings), max |d| / max |ref|.

| Check | Tolerance key | Bound | Max abs | Row-relative | Min cosine | Cases | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `tokenizer.assets` |  |  |  |  |  |  | passed |
| `dotnet-ids.assets` |  |  |  |  |  |  | passed |
| `tokenizer.token_ids` | discrete.token_ids | exact |  |  |  | 42 | passed |
| `tokenizer.rgb_sha256` | decoder_skiasharp.decoded_rgb_vs_pil | exact |  |  |  | 35 | passed |
| `tokenizer.resized_sha256` | discrete.resized_pixels | exact |  |  |  | 35 | passed |
| `tokenizer.pixel_values_sha256` | discrete.pixel_values | exact |  |  |  | 35 | passed |
| `tokenizer.positions_sha256` | discrete.image_positions | exact |  |  |  | 35 | passed |
| `dotnet-ids.embedding` | float.embedding_max_abs | 1e-05 | 2.24e-06 |  |  | 42 | passed |
| `dotnet-ids.min_cosine` | float.min_cosine | 0.9999995 |  |  | 0.999999999893 | 42 | passed |
| `dotnet-ids.soft_tokens` | float.soft_tokens_row_relative | 0.001 |  | 0.000157 |  | 35 | passed |
| `dotnet-ids.token_embeddings` | float.token_embeddings_row_relative | 0.001 |  | 4.54e-05 |  | 34 | passed |
| `dotnet-ids.finite` | float.nan_or_inf | fail |  |  |  |  | passed |
| `dotnet-ids.batch.vision-batch-8` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 |  |  | 6 | passed |
| `dotnet-ids.batch.padded-2520-vision-batch-3` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 |  |  | 6 | passed |
| `dotnet-ids.provider_options` | provider_options | {} |  |  |  |  | passed |
| `export.not_applicable` |  |  |  |  |  |  | not-applicable |
| `dotnet-text.not_applicable` |  |  |  |  |  |  | not-applicable |

Details:

- `export.not_applicable`: the image tolerances bind the C# path; Python ORT parity: protected parity-cpu.json
- `dotnet-text.not_applicable`: text-only rows are stage 3 rows (released text graph)

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `230f09eba7d79507e0783bdde688c78160faae1f`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/image-fp32-cpu/tolerances.yaml` | `40aefceb6d97b896d880f9b09d6238fcf0261377a85bdee80fd3808258a69b05` |
| `model-pipeline/protected/embeddinggemma-2/image-fp32-cpu/image-embeddings.jsonl` | `8bdea7fce279128414b5223204619a2de5fbace2b307cd7570503f197ab1c113` |
| `model-pipeline/protected/embeddinggemma-2/image-fp32-cpu/images.json` | `17c4ff205e9a14c26cb75a073b15474780600a3cf9776b7c47c973fff7ec616a` |
| `model-pipeline/protected/embeddinggemma-2/image-fp32-cpu/tensors.json` | `87ac904c7f8a977fd3716042876ecb2c0d41ce757be6ff1453a683f61e878a0e` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | EmbeddingGemma2MultimodalEmbedder rendering with EmbeddingGemma2Tokenizer; EmbeddingGemma2ImageProcessor on SkiaSharp-decoded RGB | ai-dotnet/tools/Starlings.Fidelity.Tool `712e5cc1c7d673cc481238da47d0f05ed1f2f8b8` | omarchy | `2f80d9513ac6359c` |
| dotnet | C# EmbeddingGemma2MultimodalEmbedder, ONNX Runtime 1.30.0 CPU (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `712e5cc1c7d673cc481238da47d0f05ed1f2f8b8` | omarchy | `50435e0f7754cdc2` |

Reference tensors: `/home/mark/Projects/AI-models/embeddinggemma2/image-spike/ref-clean-tensors` (each array checked against the protected tensors.json).
