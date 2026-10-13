# Fidelity report: multilingual-e5-small/onnx-fp32-cpu

Date 2026-10-09. Evaluator model-pipeline evaluate 0.1.0 at `faeb9693b0f6c7be9a91af72c1fb5c8df1f4e343`. Exit code 0.

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
| `tokenizer.input_ids` | discrete.token_ids | exact |  |  |  |  | 46 | passed |
| `tokenizer.attention_mask` | discrete.attention_mask | exact |  |  |  |  | 46 | passed |
| `tokenizer.offsets` | discrete.offsets | exact |  |  |  |  |  | not-applicable |
| `export.input_ids` |  |  |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs | 0.0001 | 2.89e-06 | 2e-07 | 7.6e-07 |  | 46 | passed |
| `export.embedding` | float.embedding_max_abs | 1e-05 | 2.3e-07 | 1.99e-08 | 7.4e-08 | 0.999999999999 | 46 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 1e-05 | 2.06e-07 | 2.15e-08 | 7.79e-08 | 1.000000000000 | 6 | passed |
| `export.batch.all-cases-batch16.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 46 | passed |
| `export.batch.all-cases-batch16.vs_reference` | float.embedding_max_abs | 1e-05 | 2.3e-07 | 1.99e-08 | 7.4e-08 | 0.999999999999 | 46 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 1e-05 | 2.32e-07 | 1.95e-08 | 7.14e-08 | 0.999999999999 | 46 | passed |
| `dotnet.embedding.vs_python` | csharp_vs_python_same_machine_and_provider.embedding_max_abs | 1e-06 | 1e-07 | 2.97e-09 | 2e-08 | 1.000000000000 | 46 | passed |
| `dotnet.last_hidden_state.vs_python` |  |  | 0 | 0 | 0 |  | 46 | not-applicable |
| `dotnet.text_embedding` | float.embedding_max_abs | 1e-05 | 2.32e-07 | 1.95e-08 | 7.14e-08 | 0.999999999999 | 46 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 6 | passed |
| `dotnet.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 1e-05 | 2.06e-07 | 2.13e-08 | 7.6e-08 | 1.000000000000 | 6 | passed |
| `dotnet.batch.all-cases-batch16.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 46 | passed |
| `dotnet.batch.all-cases-batch16.vs_reference` | float.embedding_max_abs | 1e-05 | 2.32e-07 | 1.95e-08 | 7.14e-08 | 0.999999999999 | 46 | passed |
| `dotnet.batch.text-batch.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 46 | passed |
| `dotnet.batch.text-batch.vs_reference` | float.embedding_max_abs | 1e-05 | 2.32e-07 | 1.95e-08 | 7.14e-08 | 0.999999999999 | 46 | passed |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `dotnet.last_hidden_state.vs_python`: report only: no protected bound for C# hidden states

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `ff552651d4ee94fa525c15e26b4440fa5e0fd755`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/tolerances.yaml` | `d6a18d85059f0401b2449d7a01d91b9f8b93ce5098639aee211e03ddd6358e12` |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/tokenizer.jsonl` | `cb0e61fe1de97956fef1a847a102f2aadd675a2ed7efbfc1a48ac9db2f4b61a1` |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl` | `cb7c1fcd618bcddeb664dc05fb14f896c253b1151c2969ab87a1b819a6415ae1` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Tokenizers via E5Tokenizer (profile xlmr-unigram-e5, right truncation to 512) | ai-dotnet/tools/Starlings.Fidelity.Tool `faeb9693b0f6c7be9a91af72c1fb5c8df1f4e343` | omarchy | `63104204e840fed7` |
| export | Python onnxruntime 1.30.0 CPUExecutionProvider on onnx/model.onnx; reference hidden states from sentence-transformers modules on torch 2.14.1+cpu fp32 CPU, same ids | model-pipeline/reference/encoder/stage_e5_export.py `faeb9693b0f6c7be9a91af72c1fb5c8df1f4e343` | omarchy | `99ffc64b3380b6f0` |
| dotnet | C# E5Encoder + E5TextEmbedder, ONNX Runtime 1.30.0 CPU (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `faeb9693b0f6c7be9a91af72c1fb5c8df1f4e343` | omarchy | `046a8b84e4bf6167` |
