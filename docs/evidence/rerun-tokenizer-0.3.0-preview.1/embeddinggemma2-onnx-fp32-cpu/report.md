# Fidelity report: embeddinggemma-2/onnx-fp32-cpu

Date 2026-10-13. Evaluator model-pipeline evaluate 0.1.0 at `2f53409610a9f697db1ad226d6670d04a25d69e4`. Exit code 0.

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
| `export.provider_options` |  | {} |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs |  | 0.00104 | 2.69e-06 | 2.93e-05 |  | 138 | not-applicable |
| `export.token_embeddings` | float.token_embeddings_relative | 0.0001 | 0.0225 |  |  |  | 138 | passed |
| `export.embedding` | float.embedding_max_abs | 5e-06 | 4.2e-07 | 2.14e-08 | 8.8e-08 | 0.999999999995 | 138 | passed |
| `export.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999995 | 138 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.8e-07 | 1.81e-08 | 1.17e-07 | 0.999999999999 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 3.4e-07 | 2.42e-08 | 1.3e-07 | 0.999999999999 | 6 | passed |
| `export.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999999 | 6 | passed |
| `export.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.8e-07 | 5.07e-09 | 4.9e-08 | 0.999999999999 | 138 | passed |
| `export.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 4.2e-07 | 2.12e-08 | 8.76e-08 | 0.999999999996 | 138 | passed |
| `export.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 138 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  | {} |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 5e-06 | 3.7e-07 | 2.11e-08 | 8.5e-08 | 0.999999999995 | 138 | passed |
| `dotnet.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999995 | 138 | passed |
| `dotnet.embedding.vs_python` | csharp_vs_python_same_machine_and_provider.embedding_max_abs | 1e-06 | 4.2e-07 | 2e-09 | 2.3e-08 | 1.000000000000 | 138 | passed |
| `dotnet.text_embedding` | float.embedding_max_abs | 5e-06 | 3.7e-07 | 2.11e-08 | 8.5e-08 | 0.999999999995 | 138 | passed |
| `dotnet.text_embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999995 | 138 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.76e-07 | 1.8e-08 | 1.12e-07 | 0.999999999999 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 2.04e-07 | 2.35e-08 | 1.23e-07 | 0.999999999999 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999999 | 6 | passed |
| `dotnet.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.78e-07 | 5e-09 | 4.8e-08 | 0.999999999999 | 138 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 3.68e-07 | 2.08e-08 | 8.49e-08 | 0.999999999996 | 138 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 138 | passed |
| `dotnet.batch.text-batch-none.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.5e-07 | 1.32e-08 | 5.82e-08 | 0.999999999999 | 46 | passed |
| `dotnet.batch.text-batch-none.vs_reference` | float.embedding_max_abs | 5e-06 | 1.92e-07 | 1.97e-08 | 7.6e-08 | 0.999999999999 | 46 | passed |
| `dotnet.batch.text-batch-none.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999999 | 46 | passed |
| `dotnet.batch.text-batch-query.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.1e-07 | 1.19e-08 | 5.2e-08 | 1.000000000000 | 46 | passed |
| `dotnet.batch.text-batch-query.vs_reference` | float.embedding_max_abs | 5e-06 | 1.94e-07 | 1.93e-08 | 7.4e-08 | 0.999999999999 | 46 | passed |
| `dotnet.batch.text-batch-query.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999999 | 46 | passed |
| `dotnet.batch.text-batch-document.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.78e-07 | 1.37e-08 | 6e-08 | 0.999999999999 | 46 | passed |
| `dotnet.batch.text-batch-document.vs_reference` | float.embedding_max_abs | 5e-06 | 3.68e-07 | 2.18e-08 | 9.6e-08 | 0.999999999996 | 46 | passed |
| `dotnet.batch.text-batch-document.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 46 | passed |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `export.last_hidden_state`: report only: float.last_hidden_state_max_abs is not in the protected tolerances
- `export.token_embeddings`: per case max |d| / max(1, max |ref|) 6.17e-06 <= 0.0001 (worst case long-english@document); report only, per value |d| / max(1, |ref|) 0.0166 (worst case long-english@document)

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `fb276585f2cd5fc6a848d902280c0002a49c2fc6`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/tolerances.yaml` | `cadfb7a767968f9bdbee8501a6386a872e7dd72919b688edc486105472d69bb3` |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/tokenizer.jsonl` | `f8e090f09297c319ae54886cacea84975a9157aff4f90b766ca13a54b1895be6` |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/embeddings.jsonl` | `326664e8c3fb074eb7ad7a640d2c8afffa49ebabc0f9318be39ab23b93fc49db` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Tokenizers via EmbeddingGemma2Tokenizer (profile gemma-bpe-embeddinggemma2, no truncation), on prompt text + text | ai-dotnet/tools/Starlings.Fidelity.Tool `2f53409610a9f697db1ad226d6670d04a25d69e4` | omarchy | `77f4be9b5d8ddf73` |
| export | Python onnxruntime 1.30.0 CPUExecutionProvider on the release graph; reference token embeddings from eg2_text.TextPath on torch 2.14.1+cpu fp32 CPU, same ids | model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py `2f53409610a9f697db1ad226d6670d04a25d69e4` | omarchy | `c936efd0be418af7` |
| dotnet | C# EmbeddingGemma2Encoder + EmbeddingGemma2TextEmbedder, ONNX Runtime 1.30.0 CPU (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `2f53409610a9f697db1ad226d6670d04a25d69e4` | omarchy | `41d6d9e850ae31fb` |
