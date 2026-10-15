# Fidelity report: embeddinggemma-2/image-fp32-cpu (image-embedding)

Date 2026-10-15. Evaluator model-pipeline evaluate 0.1.0 at `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023`. Exit code 0.

Fixtures: `model-pipeline/protected/embeddinggemma-2/release/image-fp32-release` (from the --fixtures option).

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
| `tokenizer.token_ids` | discrete.token_ids | exact |  |  |  | 19 | passed |
| `tokenizer.rgb_sha256` | decoder_skiasharp.decoded_rgb_vs_pil | exact |  |  |  | 20 | passed |
| `tokenizer.resized_sha256` | discrete.resized_pixels | exact |  |  |  | 20 | passed |
| `tokenizer.pixel_values_sha256` | discrete.pixel_values | exact |  |  |  | 20 | passed |
| `tokenizer.positions_sha256` | discrete.image_positions | exact |  |  |  | 20 | passed |
| `dotnet-ids.embedding` | float.embedding_max_abs | 1e-05 | 8.55e-07 |  |  | 19 | passed |
| `dotnet-ids.min_cosine` | float.min_cosine | 0.9999995 |  |  | 0.999999999981 | 19 | passed |
| `dotnet-ids.soft_tokens` | float.soft_tokens_row_relative | 0.001 |  | 8.68e-05 |  | 20 | passed |
| `dotnet-ids.token_embeddings` | float.token_embeddings_row_relative | 0.001 |  | 1.79e-05 |  | 19 | passed |
| `dotnet-ids.finite` | float.nan_or_inf | fail |  |  |  |  | passed |
| `dotnet-ids.batch.vision-batch-8` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 |  |  | 1 | passed |
| `dotnet-ids.batch.padded-2520-vision-batch-3` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 |  |  | 1 | passed |
| `dotnet-ids.provider_options` | provider_options | {} |  |  |  |  | passed |
| `export.not_applicable` |  |  |  |  |  |  | not-applicable |
| `dotnet-text.not_applicable` |  |  |  |  |  |  | not-applicable |

Details:

- `export.not_applicable`: the image tolerances bind the C# path; Python ORT parity: protected parity-cpu.json
- `dotnet-text.not_applicable`: text-only rows are stage 3 rows (released text graph)

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `75574190e77f5528a416e87c6e12ea17e3b85d33`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/image-fp32-cpu/tolerances.yaml` | `40aefceb6d97b896d880f9b09d6238fcf0261377a85bdee80fd3808258a69b05` |
| `model-pipeline/protected/embeddinggemma-2/release/image-fp32-release/image-embeddings.jsonl` | `13bf5780df73de4e88097bfcb7dcf648d32a789eab1683e2bf12713437769b9d` |
| `model-pipeline/protected/embeddinggemma-2/release/image-fp32-release/images.json` | `888b56cb1e0d49f7b1e94f65eaf4d97e0cba2921d6eea9b763e7f3c97dcb4f29` |
| `model-pipeline/protected/embeddinggemma-2/release/image-fp32-release/tensors.json` | `18c8c1ea1c04d8d69a751ae93403edb5dff09675be16951335e927d508e33ea1` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | EmbeddingGemma2MultimodalEmbedder rendering with EmbeddingGemma2Tokenizer; EmbeddingGemma2ImageProcessor on SkiaSharp-decoded RGB | ai-dotnet/tools/Starlings.Fidelity.Tool `e0e29b9677c1c4b3e99d1dbe5161c794b426dff4` | omarchy | `53974ea3084ad685` |
| dotnet | C# EmbeddingGemma2MultimodalEmbedder, ONNX Runtime 1.30.0 CPU (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `e0e29b9677c1c4b3e99d1dbe5161c794b426dff4` | omarchy | `522c52bb8d6d3c60` |

Reference tensors: `/home/mark/.cache/starling-release-gates/runs/eg2-image-ref/tensors` (each array checked against the protected tensors.json).
