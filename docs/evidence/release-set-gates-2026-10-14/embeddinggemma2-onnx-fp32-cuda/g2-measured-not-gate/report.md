# Fidelity report: embeddinggemma-2/onnx-fp32-cuda

Date 2026-10-15. Evaluator model-pipeline evaluate 0.1.0 at `None`. Exit code 0.

Fixtures: `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release` (from the --fixtures option).

First divergence: **none**.

## Gates

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | not-run | protected evidence not verified against Git (verification disabled by --protected-base none) |
| G2 | not-run | protected evidence not verified against Git (verification disabled by --protected-base none) |

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
| `export.last_hidden_state` | float.last_hidden_state_max_abs |  | 0.000877 | 2.26e-06 | 1.6e-05 |  | 696 | not-applicable |
| `export.token_embeddings` | float.token_embeddings_relative | 0.0001 | 0.0203 |  |  |  | 696 | passed |
| `export.embedding` | float.embedding_max_abs | 5e-06 | 9.8e-07 | 2.25e-08 | 9.3e-08 | 0.999999999995 | 696 | passed |
| `export.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999995 | 696 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.44e-07 | 3.55e-08 | 1.45e-07 | 0.999999999998 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 2.16e-07 | 3.64e-08 | 1.45e-07 | 0.999999999998 | 6 | passed |
| `export.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999998 | 6 | passed |
| `export.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 7.1e-07 | 2.17e-08 | 9e-08 | 0.999999999996 | 696 | passed |
| `export.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 4.45e-07 | 2.47e-08 | 1.02e-07 | 0.999999999994 | 696 | passed |
| `export.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999994 | 696 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  | {'use_tf32': 0} |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 5e-06 | 9.8e-07 | 2.24e-08 | 9.3e-08 | 0.999999999995 | 696 | passed |
| `dotnet.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999995 | 696 | passed |
| `dotnet.embedding.vs_python` | csharp_vs_python_same_machine_and_provider.embedding_max_abs | 1e-06 | 2.6e-07 | 1.79e-09 | 1.6e-08 | 1.000000000000 | 696 | passed |
| `dotnet.text_embedding` | float.embedding_max_abs | 5e-06 | 9.8e-07 | 2.24e-08 | 9.3e-08 | 0.999999999995 | 696 | passed |
| `dotnet.text_embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999995 | 696 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.48e-07 | 3.51e-08 | 1.45e-07 | 0.999999999998 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 2.16e-07 | 3.6e-08 | 1.45e-07 | 0.999999999998 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999998 | 6 | passed |
| `dotnet.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 7.4e-07 | 2.14e-08 | 9e-08 | 0.999999999996 | 696 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 4.45e-07 | 2.46e-08 | 1.01e-07 | 0.999999999994 | 696 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999994 | 696 | passed |
| `dotnet.batch.text-batch-none.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 2.44e-07 | 2.2e-08 | 9.05e-08 | 0.999999999998 | 232 | passed |
| `dotnet.batch.text-batch-none.vs_reference` | float.embedding_max_abs | 5e-06 | 2.84e-07 | 2.41e-08 | 1.02e-07 | 0.999999999998 | 232 | passed |
| `dotnet.batch.text-batch-none.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999998 | 232 | passed |
| `dotnet.batch.text-batch-query.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 3.02e-07 | 2.45e-08 | 1.01e-07 | 0.999999999998 | 232 | passed |
| `dotnet.batch.text-batch-query.vs_reference` | float.embedding_max_abs | 5e-06 | 4.25e-07 | 2.65e-08 | 1.08e-07 | 0.999999999996 | 232 | passed |
| `dotnet.batch.text-batch-query.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 232 | passed |
| `dotnet.batch.text-batch-document.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 5.6e-07 | 2.61e-08 | 1.06e-07 | 0.999999999998 | 232 | passed |
| `dotnet.batch.text-batch-document.vs_reference` | float.embedding_max_abs | 5e-06 | 4.2e-07 | 2.82e-08 | 1.14e-07 | 0.999999999997 | 232 | passed |
| `dotnet.batch.text-batch-document.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999997 | 232 | passed |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `export.last_hidden_state`: report only: float.last_hidden_state_max_abs is not in the protected tolerances
- `export.token_embeddings`: per case max |d| / max(1, max |ref|) 7.97e-06 <= 0.0001 (worst case massive-nl-NL-10563@document); report only, per value |d| / max(1, |ref|) 0.00448 (worst case oasst1-06f21312-d9c1-4970-9d7a-b129e8069aba@document)

## Protected evidence

Status **unverified** (verification disabled by --protected-base none; base `None`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cuda/tolerances.yaml` | `1bd9c65a257d3ff992fabce9c820ff634dc88f2506bd68bb5c50d3017d7beba0` |
| `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release/tokenizer.jsonl` | `25a991f55885ce1fc686e7daf23bb62e727f6cad0a0f3b2a7546de0b4abe2954` |
| `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release/embeddings.jsonl` | `c2ffe5e45be11904b7ad0277758c0d8bafc48c6c65dcc87ff932ccbaf574285c` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Tokenizers via EmbeddingGemma2Tokenizer (profile gemma-bpe-embeddinggemma2, no truncation), on prompt text + text | ai-dotnet/tools/Starlings.Fidelity.Tool `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023` | omarchy | `d88dc0795fa683d2` |
| export | Python onnxruntime 1.30.0 CUDAExecutionProvider on the release graph; reference token embeddings from eg2_text.TextPath on torch 2.14.1+cu130 fp32 CPU, same ids | model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023` | omarchy | `395c5e6f3451447a` |
| dotnet | C# EmbeddingGemma2Encoder + EmbeddingGemma2TextEmbedder, ONNX Runtime 1.30.0 CUDA (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023` | omarchy | `75abdae1375e2a89` |
