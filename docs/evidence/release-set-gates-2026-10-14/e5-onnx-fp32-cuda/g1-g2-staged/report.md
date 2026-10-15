# Fidelity report: multilingual-e5-small/onnx-fp32-cuda

Date 2026-10-15. Evaluator model-pipeline evaluate 0.1.0 at `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023`. Exit code 0.

Fixtures: `model-pipeline/protected/multilingual-e5-small/release/onnx-fp32-release` (from the --fixtures option).

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

Absolute errors. Bounds come only from the protected tolerances file; report-only metrics: mean_abs, p99_abs, min_cosine.

| Check | Tolerance key | Bound | Max abs | Mean abs | p99 abs | Min cosine | Cases | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dotnet.identity` |  |  |  |  |  |  |  | passed |
| `export.identity` |  |  |  |  |  |  |  | passed |
| `tokenizer.identity` |  |  |  |  |  |  |  | passed |
| `tokenizer.input_ids` | discrete.token_ids | exact |  |  |  |  | 232 | passed |
| `tokenizer.attention_mask` | discrete.attention_mask | exact |  |  |  |  | 232 | passed |
| `tokenizer.offsets` | discrete.offsets | exact |  |  |  |  |  | not-applicable |
| `export.input_ids` |  |  |  |  |  |  |  | passed |
| `export.provider_options` |  | {'use_tf32': 0} |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs | 0.0001 | 2.25e-06 | 1.66e-07 | 6.26e-07 |  | 232 | passed |
| `export.embedding` | float.embedding_max_abs | 1e-05 | 1.3e-07 | 1.84e-08 | 6.3e-08 | 1.000000000000 | 232 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 8e-08 | 1.57e-08 | 5.4e-08 | 1.000000000000 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 1e-05 | 1.5e-07 | 2.38e-08 | 7.5e-08 | 1.000000000000 | 6 | passed |
| `export.batch.all-cases-batch16.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 1.07e-07 | 1.23e-08 | 4.4e-08 | 1.000000000000 | 232 | passed |
| `export.batch.all-cases-batch16.vs_reference` | float.embedding_max_abs | 1e-05 | 1.38e-07 | 1.96e-08 | 6.7e-08 | 1.000000000000 | 232 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  | {'use_tf32': 0} |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 1e-05 | 1.2e-07 | 1.81e-08 | 6e-08 | 1.000000000000 | 232 | passed |
| `dotnet.embedding.vs_python` | csharp_vs_python_same_machine_and_provider.embedding_max_abs | 1e-06 | 8.5e-08 | 2.94e-09 | 2e-08 | 1.000000000000 | 232 | passed |
| `dotnet.last_hidden_state.vs_python` |  |  | 0 | 0 | 0 |  | 232 | not-applicable |
| `dotnet.text_embedding` | float.embedding_max_abs | 1e-05 | 1.2e-07 | 1.81e-08 | 6e-08 | 1.000000000000 | 232 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 7e-08 | 1.5e-08 | 5e-08 | 1.000000000000 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 1e-05 | 1.16e-07 | 2.31e-08 | 7e-08 | 1.000000000000 | 6 | passed |
| `dotnet.batch.all-cases-batch16.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 1.05e-07 | 1.17e-08 | 4.1e-08 | 1.000000000000 | 232 | passed |
| `dotnet.batch.all-cases-batch16.vs_reference` | float.embedding_max_abs | 1e-05 | 1.42e-07 | 1.93e-08 | 6.5e-08 | 1.000000000000 | 232 | passed |
| `dotnet.batch.text-batch.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 8.7e-08 | 1.32e-08 | 4.5e-08 | 1.000000000000 | 232 | passed |
| `dotnet.batch.text-batch.vs_reference` | float.embedding_max_abs | 1e-05 | 1.42e-07 | 2e-08 | 6.8e-08 | 1.000000000000 | 232 | passed |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `dotnet.last_hidden_state.vs_python`: report only: no protected bound for C# hidden states

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `75574190e77f5528a416e87c6e12ea17e3b85d33`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cuda/tolerances.yaml` | `fdb34fa7ca8c6e9d8601e6592300e3733c3523ccef38337e392b326c592b5d1e` |
| `model-pipeline/protected/multilingual-e5-small/release/onnx-fp32-release/tokenizer.jsonl` | `0469daa84f5225c4b11b63f78ba85ba0b5f7d2f3b48f96c124fafe24a868af1b` |
| `model-pipeline/protected/multilingual-e5-small/release/onnx-fp32-release/embeddings.jsonl` | `49202413016a64b33892e0c066dfe880cb6c97fd740edb2c3f005d1ba57cd2d5` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Tokenizers via E5Tokenizer (profile xlmr-unigram-e5, right truncation to 512) | ai-dotnet/tools/Starlings.Fidelity.Tool `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023` | omarchy | `1b3ab2939760b330` |
| export | Python onnxruntime 1.30.0 CUDAExecutionProvider on onnx/model.onnx; reference hidden states from sentence-transformers modules on torch 2.14.1+cu130 fp32 CPU, same ids | model-pipeline/reference/encoder/stage_e5_export.py `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023` | omarchy | `8ac6a4060bf9faa9` |
| dotnet | C# E5Encoder + E5TextEmbedder, ONNX Runtime 1.30.0 CUDA (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023` | omarchy | `a3d3e80cd6b51dfe` |
