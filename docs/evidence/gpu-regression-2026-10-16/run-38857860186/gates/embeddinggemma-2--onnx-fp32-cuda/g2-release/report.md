# Fidelity report: embeddinggemma-2/onnx-fp32-cuda

Date 2026-10-16. Evaluator model-pipeline evaluate 0.1.0 at `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8`. Exit code 0.

Fixtures: `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release` (from the --fixtures option).

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
| `tokenizer.input_ids` | discrete.token_ids | exact |  |  |  |  | 696 | passed |
| `tokenizer.attention_mask` | discrete.attention_mask | exact |  |  |  |  | 696 | passed |
| `tokenizer.offsets` | discrete.offsets | exact |  |  |  |  |  | not-applicable |
| `export.input_ids` |  |  |  |  |  |  |  | passed |
| `export.provider_options` |  | {'use_tf32': 0} |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs |  | 0.000299 | 2.19e-06 | 1.55e-05 |  | 696 | not-applicable |
| `export.token_embeddings` | float.token_embeddings_relative | 0.0001 | 0.00776 |  |  |  | 696 | passed |
| `export.embedding` | float.embedding_max_abs | 5e-06 | 8.6e-07 | 2.2e-08 | 9e-08 | 0.999999999996 | 696 | passed |
| `export.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 696 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.1e-07 | 2.63e-08 | 1.02e-07 | 0.999999999999 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 3e-07 | 2.71e-08 | 1.04e-07 | 0.999999999999 | 6 | passed |
| `export.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999999 | 6 | passed |
| `export.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.9e-07 | 2.05e-08 | 8.4e-08 | 0.999999999998 | 696 | passed |
| `export.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 5.7e-07 | 2.36e-08 | 9.7e-08 | 0.999999999997 | 696 | passed |
| `export.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999997 | 696 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  | {'use_tf32': 0} |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 5e-06 | 8.6e-07 | 2.18e-08 | 8.97e-08 | 0.999999999996 | 696 | passed |
| `dotnet.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 696 | passed |
| `dotnet.embedding.vs_python` | csharp_vs_python_same_machine_and_provider.embedding_max_abs | 1e-06 | 2.5e-07 | 1.81e-09 | 1.6e-08 | 1.000000000000 | 696 | passed |
| `dotnet.text_embedding` | float.embedding_max_abs | 5e-06 | 8.6e-07 | 2.18e-08 | 8.97e-08 | 0.999999999996 | 696 | passed |
| `dotnet.text_embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 696 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.1e-07 | 2.59e-08 | 1.02e-07 | 0.999999999999 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 1.6e-07 | 2.68e-08 | 1.01e-07 | 0.999999999999 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999999 | 6 | passed |
| `dotnet.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.7e-07 | 2.02e-08 | 8.3e-08 | 0.999999999998 | 696 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 5.9e-07 | 2.35e-08 | 9.6e-08 | 0.999999999997 | 696 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999997 | 696 | passed |
| `dotnet.batch.text-batch-none.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.83e-07 | 2.11e-08 | 8.8e-08 | 0.999999999997 | 232 | passed |
| `dotnet.batch.text-batch-none.vs_reference` | float.embedding_max_abs | 5e-06 | 2.84e-07 | 2.36e-08 | 9.8e-08 | 0.999999999996 | 232 | passed |
| `dotnet.batch.text-batch-none.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 232 | passed |
| `dotnet.batch.text-batch-query.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.67e-07 | 2.36e-08 | 9.8e-08 | 0.999999999997 | 232 | passed |
| `dotnet.batch.text-batch-query.vs_reference` | float.embedding_max_abs | 5e-06 | 3.48e-07 | 2.6e-08 | 1.05e-07 | 0.999999999998 | 232 | passed |
| `dotnet.batch.text-batch-query.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999998 | 232 | passed |
| `dotnet.batch.text-batch-document.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 3.87e-07 | 2.55e-08 | 1.01e-07 | 0.999999999995 | 232 | passed |
| `dotnet.batch.text-batch-document.vs_reference` | float.embedding_max_abs | 5e-06 | 9.8e-07 | 2.88e-08 | 1.18e-07 | 0.999999999994 | 232 | passed |
| `dotnet.batch.text-batch-document.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999994 | 232 | passed |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `export.last_hidden_state`: report only: float.last_hidden_state_max_abs is not in the protected tolerances
- `export.token_embeddings`: per case max |d| / max(1, max |ref|) 2.43e-06 <= 0.0001 (worst case oasst1-06f21312-d9c1-4970-9d7a-b129e8069aba@document); report only, per value |d| / max(1, |ref|) 0.00394 (worst case humaneval-32@query)

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cuda/tolerances.yaml` | `1bd9c65a257d3ff992fabce9c820ff634dc88f2506bd68bb5c50d3017d7beba0` |
| `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release/tokenizer.jsonl` | `25a991f55885ce1fc686e7daf23bb62e727f6cad0a0f3b2a7546de0b4abe2954` |
| `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release/embeddings.jsonl` | `c2ffe5e45be11904b7ad0277758c0d8bafc48c6c65dcc87ff932ccbaf574285c` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Tokenizers via EmbeddingGemma2Tokenizer (profile gemma-bpe-embeddinggemma2, no truncation), on prompt text + text | ai-dotnet/tools/Starlings.Fidelity.Tool `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8` | 65fb18f5f39a | `8866664ce3de15ec` |
| export | Python onnxruntime 1.30.0 CUDAExecutionProvider on the release graph; reference token embeddings from eg2_text.TextPath on torch 2.14.1+cu130 fp32 CPU, same ids | model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8` | 65fb18f5f39a | `3bf7f179bd929f8f` |
| dotnet | C# EmbeddingGemma2Encoder + EmbeddingGemma2TextEmbedder, ONNX Runtime 1.30.0 CUDA (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8` | 65fb18f5f39a | `22e4b0d1612bc0b9` |
