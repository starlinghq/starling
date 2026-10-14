# Fidelity report: embeddinggemma-2/onnx-fp32-cpu

Date 2026-10-14. Evaluator model-pipeline evaluate 0.1.0 at `None`. Exit code 0.

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
| `export.provider_options` |  | {} |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs |  | 0.000454 | 2.27e-06 | 1.62e-05 |  | 696 | not-applicable |
| `export.token_embeddings` | float.token_embeddings_relative | 0.0001 | 0.0107 |  |  |  | 696 | passed |
| `export.embedding` | float.embedding_max_abs | 5e-06 | 5.1e-07 | 2.34e-08 | 9.6e-08 | 0.999999999996 | 696 | passed |
| `export.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 696 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.2e-07 | 1.42e-08 | 7.1e-08 | 1.000000000000 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 1.5e-07 | 2.14e-08 | 8.39e-08 | 0.999999999999 | 6 | passed |
| `export.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999999 | 6 | passed |
| `export.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.99e-07 | 1.97e-09 | 3.7e-08 | 0.999999999999 | 696 | passed |
| `export.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 5.1e-07 | 2.34e-08 | 9.6e-08 | 0.999999999996 | 696 | passed |
| `export.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 696 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  | {} |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 5e-06 | 5.1e-07 | 2.33e-08 | 9.5e-08 | 0.999999999996 | 696 | passed |
| `dotnet.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 696 | passed |
| `dotnet.embedding.vs_python` | csharp_vs_python_same_machine_and_provider.embedding_max_abs | 1e-06 | 1.7e-07 | 1.8e-09 | 1.6e-08 | 1.000000000000 | 696 | passed |
| `dotnet.text_embedding` | float.embedding_max_abs | 5e-06 | 5.1e-07 | 2.33e-08 | 9.5e-08 | 0.999999999996 | 696 | passed |
| `dotnet.text_embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 696 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 |  |  |  |  |  | passed |
| `dotnet.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.09e-07 | 1.41e-08 | 7.1e-08 | 1.000000000000 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 5e-06 | 1.22e-07 | 2.08e-08 | 8.3e-08 | 0.999999999999 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999999 | 6 | passed |
| `dotnet.batch.all-cases-batch8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.98e-07 | 1.94e-09 | 3.68e-08 | 0.999999999999 | 696 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference` | float.embedding_max_abs | 5e-06 | 5.1e-07 | 2.32e-08 | 9.55e-08 | 0.999999999996 | 696 | passed |
| `dotnet.batch.all-cases-batch8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 696 | passed |
| `dotnet.batch.text-batch-none.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.36e-07 | 4.88e-09 | 4.24e-08 | 1.000000000000 | 232 | passed |
| `dotnet.batch.text-batch-none.vs_reference` | float.embedding_max_abs | 5e-06 | 3.35e-07 | 2.08e-08 | 8.5e-08 | 0.999999999997 | 232 | passed |
| `dotnet.batch.text-batch-none.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999997 | 232 | passed |
| `dotnet.batch.text-batch-query.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.74e-07 | 9.32e-09 | 5.18e-08 | 0.999999999999 | 232 | passed |
| `dotnet.batch.text-batch-query.vs_reference` | float.embedding_max_abs | 5e-06 | 3e-07 | 2.27e-08 | 9.2e-08 | 0.999999999998 | 232 | passed |
| `dotnet.batch.text-batch-query.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999998 | 232 | passed |
| `dotnet.batch.text-batch-document.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 5e-06 | 1.98e-07 | 1.05e-08 | 5.98e-08 | 0.999999999999 | 232 | passed |
| `dotnet.batch.text-batch-document.vs_reference` | float.embedding_max_abs | 5e-06 | 3.65e-07 | 2.52e-08 | 1.04e-07 | 0.999999999996 | 232 | passed |
| `dotnet.batch.text-batch-document.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999999999996 | 232 | passed |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `export.last_hidden_state`: report only: float.last_hidden_state_max_abs is not in the protected tolerances
- `export.token_embeddings`: per case max |d| / max(1, max |ref|) 4.22e-06 <= 0.0001 (worst case massive-nl-NL-10563@document); report only, per value |d| / max(1, |ref|) 0.00282 (worst case humaneval-10@document)

## Protected evidence

Status **unverified** (verification disabled by --protected-base none; base `None`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/tolerances.yaml` | `cadfb7a767968f9bdbee8501a6386a872e7dd72919b688edc486105472d69bb3` |
| `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release/tokenizer.jsonl` | `25a991f55885ce1fc686e7daf23bb62e727f6cad0a0f3b2a7546de0b4abe2954` |
| `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release/embeddings.jsonl` | `ac0a3a9b64c6c241df721727ac48e7abe9acd1f7b926185c8d487e79def4b47c` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Tokenizers via EmbeddingGemma2Tokenizer (profile gemma-bpe-embeddinggemma2, no truncation), on prompt text + text | ai-dotnet/tools/Starlings.Fidelity.Tool `0eaaf9f943412c96f738c9e9c7e59192f2910845` | omarchy | `4ebf954c115a07a5` |
| export | Python onnxruntime 1.30.0 CPUExecutionProvider on the release graph; reference token embeddings from eg2_text.TextPath on torch 2.14.1+cpu fp32 CPU, same ids | model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py `10c016c1e8724983306d79a5db02b84326109c32` | omarchy | `754be49608bd06f1` |
| dotnet | C# EmbeddingGemma2Encoder + EmbeddingGemma2TextEmbedder, ONNX Runtime 1.30.0 CPU (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `0eaaf9f943412c96f738c9e9c7e59192f2910845` | omarchy | `73ce4b842cbd8b32` |
