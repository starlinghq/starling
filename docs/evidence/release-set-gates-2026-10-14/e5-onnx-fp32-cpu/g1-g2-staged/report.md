# Fidelity report: multilingual-e5-small/onnx-fp32-cpu

Date 2026-10-14. Evaluator model-pipeline evaluate 0.1.0 at `b469654b17262920457e810af410063b13cce8ea`. Exit code 0.

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
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs | 0.0001 | 2.56e-06 | 1.82e-07 | 6.85e-07 |  | 232 | passed |
| `export.embedding` | float.embedding_max_abs | 1e-05 | 1.51e-07 | 2.07e-08 | 7.1e-08 | 1.000000000000 | 232 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 1e-05 | 1.45e-07 | 2.21e-08 | 8e-08 | 1.000000000000 | 6 | passed |
| `export.batch.all-cases-batch16.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 232 | passed |
| `export.batch.all-cases-batch16.vs_reference` | float.embedding_max_abs | 1e-05 | 1.51e-07 | 2.07e-08 | 7.1e-08 | 1.000000000000 | 232 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 1e-05 | 1.46e-07 | 2.04e-08 | 7e-08 | 1.000000000000 | 232 | passed |
| `dotnet.embedding.vs_python` | csharp_vs_python_same_machine_and_provider.embedding_max_abs | 1e-06 | 8e-08 | 2.96e-09 | 2e-08 | 1.000000000000 | 232 | passed |
| `dotnet.last_hidden_state.vs_python` |  |  | 0 | 0 | 0 |  | 232 | not-applicable |
| `dotnet.text_embedding` | float.embedding_max_abs | 1e-05 | 1.46e-07 | 2.04e-08 | 7e-08 | 1.000000000000 | 232 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 1e-05 | 1.19e-07 | 2.18e-08 | 7.5e-08 | 1.000000000000 | 6 | passed |
| `dotnet.batch.all-cases-batch16.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 232 | passed |
| `dotnet.batch.all-cases-batch16.vs_reference` | float.embedding_max_abs | 1e-05 | 1.46e-07 | 2.04e-08 | 7e-08 | 1.000000000000 | 232 | passed |
| `dotnet.batch.text-batch.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 232 | passed |
| `dotnet.batch.text-batch.vs_reference` | float.embedding_max_abs | 1e-05 | 1.46e-07 | 2.04e-08 | 7e-08 | 1.000000000000 | 232 | passed |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `dotnet.last_hidden_state.vs_python`: report only: no protected bound for C# hidden states

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `b469654b17262920457e810af410063b13cce8ea`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/tolerances.yaml` | `d6a18d85059f0401b2449d7a01d91b9f8b93ce5098639aee211e03ddd6358e12` |
| `model-pipeline/protected/multilingual-e5-small/release/onnx-fp32-release/tokenizer.jsonl` | `0469daa84f5225c4b11b63f78ba85ba0b5f7d2f3b48f96c124fafe24a868af1b` |
| `model-pipeline/protected/multilingual-e5-small/release/onnx-fp32-release/embeddings.jsonl` | `49202413016a64b33892e0c066dfe880cb6c97fd740edb2c3f005d1ba57cd2d5` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Tokenizers via E5Tokenizer (profile xlmr-unigram-e5, right truncation to 512) | ai-dotnet/tools/Starlings.Fidelity.Tool `b469654b17262920457e810af410063b13cce8ea` | omarchy | `86eb82de2e15fde9` |
| export | Python onnxruntime 1.30.0 CPUExecutionProvider on onnx/model.onnx; reference hidden states from sentence-transformers modules on torch 2.14.1+cpu fp32 CPU, same ids | model-pipeline/reference/encoder/stage_e5_export.py `b469654b17262920457e810af410063b13cce8ea` | omarchy | `85afa22348b573fd` |
| dotnet | C# E5Encoder + E5TextEmbedder, ONNX Runtime 1.30.0 CPU (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `b469654b17262920457e810af410063b13cce8ea` | omarchy | `9f045adaf3c9204b` |
