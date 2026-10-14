# Fidelity report: embeddinggemma-2/onnx-fp32-cuda

Date 2026-10-14. Evaluator model-pipeline evaluate 0.1.0 at `2db3dce7ca30269ab9c00eb4b21985d09523969d`. Exit code 0.

First divergence: **none**.

## Gates

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every check passed or is not applicable by the manifest |
| G2 | passed | every check passed or is not applicable by the manifest |

## Stages

| Stage | Status |
| --- | --- |
| Stage 1, tokenizer (owned tokenizer vs protected tokenizer fixtures) | passed |
| Stage 2, export (Python ONNX Runtime vs PyTorch reference) | passed |
| Stage 3a, .NET from fixture ids | passed |
| Stage 3b, .NET from text (owned tokenizer + encoder) | passed |

## Checks

Absolute errors. Bounds come only from the protected tolerances file; report-only metrics: mean_abs, p99_abs.

| Check | Tolerance key | Bound | Max abs | Mean abs | p99 abs | Min cosine | Cases | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dotnet.identity` |  |  |  |  |  |  |  | passed |
| `export.identity` |  |  |  |  |  |  |  | passed |
| `tokenizer.identity` |  |  |  |  |  |  |  | passed |
| `tokenizer.input_ids` | discrete.token_ids | exact |  |  |  |  | 138 | passed |
| `tokenizer.attention_mask` | discrete.attention_mask | exact |  |  |  |  | 138 | passed |
| `tokenizer.offsets` | discrete.offsets | exact |  |  |  |  |  | not-applicable |
| `export.input_ids` |  |  |  |  |  |  |  | passed |
| `export.provider_options` |  | {'use_tf32': 0} |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs |  | 0.00205 | 2.91e-06 | 3.16e-05 |  | 138 | not-applicable |
| `export.token_embeddings` | float.token_embeddings_relative | 0.0001 | 0.0258 |  |  |  | 138 | passed |
| `export.embedding` | float.embedding_max_abs | 5e-06 | 5.1e-07 | 2.14e-08 | 9.3e-08 | 0.999999999993 | 138 | passed |
| `export.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999993 | 138 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.92e-07 | 3.22e-08 | 1.58e-07 | 0.999999999998 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 3.4e-07 | 3.18e-08 | 1.49e-07 | 0.999999999998 | 6 | passed |
| `export.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999998 | 6 | passed |
| `export.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 3.37e-07 | 2.07e-08 | 9.1e-08 | 0.999999999997 | 138 | passed |
| `export.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 5.32e-07 | 2.34e-08 | 1e-07 | 0.999999999992 | 138 | passed |
| `export.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999992 | 138 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  | {'use_tf32': 0} |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 5e-06 | 4.71e-07 | 2.11e-08 | 9e-08 | 0.999999999993 | 138 | passed |
| `dotnet.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999993 | 138 | passed |
| `dotnet.embedding.vs_python` | csharp_vs_python_same_machine_and_provider.embedding_max_abs | 1e-06 | 4.6e-07 | 2e-09 | 2.2e-08 | 1.000000000000 | 138 | passed |
| `dotnet.text_embedding` | float.embedding_max_abs | 5e-06 | 4.71e-07 | 2.11e-08 | 9e-08 | 0.999999999993 | 138 | passed |
| `dotnet.text_embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999993 | 138 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.7e-07 | 3.11e-08 | 1.53e-07 | 0.999999999998 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 2.05e-07 | 2.89e-08 | 1.37e-07 | 0.999999999998 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999998 | 6 | passed |
| `dotnet.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 3.37e-07 | 2.05e-08 | 9e-08 | 0.999999999997 | 138 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 5.31e-07 | 2.33e-08 | 9.86e-08 | 0.999999999992 | 138 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999992 | 138 | passed |
| `dotnet.batch.text-batch-none.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.7e-07 | 2.3e-08 | 1e-07 | 0.999999999998 | 46 | passed |
| `dotnet.batch.text-batch-none.vs_reference` | float.embedding_max_abs | 5e-06 | 2.2e-07 | 2.35e-08 | 9.5e-08 | 0.999999999998 | 46 | passed |
| `dotnet.batch.text-batch-none.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999998 | 46 | passed |
| `dotnet.batch.text-batch-query.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.24e-07 | 2.15e-08 | 8.5e-08 | 0.999999999999 | 46 | passed |
| `dotnet.batch.text-batch-query.vs_reference` | float.embedding_max_abs | 5e-06 | 2.56e-07 | 2.3e-08 | 9.4e-08 | 0.999999999998 | 46 | passed |
| `dotnet.batch.text-batch-query.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999998 | 46 | passed |
| `dotnet.batch.text-batch-document.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.6e-07 | 2.34e-08 | 9.9e-08 | 0.999999999997 | 46 | passed |
| `dotnet.batch.text-batch-document.vs_reference` | float.embedding_max_abs | 5e-06 | 4.32e-07 | 2.6e-08 | 1.19e-07 | 0.999999999994 | 46 | passed |
| `dotnet.batch.text-batch-document.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999994 | 46 | passed |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `export.last_hidden_state`: report only: float.last_hidden_state_max_abs is not in the protected tolerances
- `export.token_embeddings`: per case max |d| / max(1, max |ref|) 7.18e-06 <= 0.0001 (worst case long-mixed); report only, per value |d| / max(1, |ref|) 0.012 (worst case long-mixed)

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `8feb5069ac6f830f161366d8d2b6fc61cd42a0b3`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cuda/tolerances.yaml` | `1bd9c65a257d3ff992fabce9c820ff634dc88f2506bd68bb5c50d3017d7beba0` |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/tokenizer.jsonl` | `f8e090f09297c319ae54886cacea84975a9157aff4f90b766ca13a54b1895be6` |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/embeddings.jsonl` | `326664e8c3fb074eb7ad7a640d2c8afffa49ebabc0f9318be39ab23b93fc49db` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Tokenizers via EmbeddingGemma2Tokenizer (profile gemma-bpe-embeddinggemma2, no truncation), on prompt text + text | ai-dotnet/tools/Starlings.Fidelity.Tool `2db3dce7ca30269ab9c00eb4b21985d09523969d` | omarchy | `06b7a0897d45a95e` |
| export | Python onnxruntime 1.30.0 CUDAExecutionProvider on the release graph; reference token embeddings from eg2_text.TextPath on torch 2.14.1+cu130 fp32 CPU, same ids | model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py `2db3dce7ca30269ab9c00eb4b21985d09523969d` | omarchy | `cbff6a6487f2f225` |
| dotnet | C# EmbeddingGemma2Encoder + EmbeddingGemma2TextEmbedder, ONNX Runtime 1.30.0 CUDA (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `2db3dce7ca30269ab9c00eb4b21985d09523969d` | omarchy | `381bc0d7e60713b9` |
