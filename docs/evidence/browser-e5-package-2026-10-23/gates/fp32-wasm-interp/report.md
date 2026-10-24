# Fidelity report: multilingual-e5-small/onnx-fp32-wasm

Date 2026-10-24. Evaluator model-pipeline evaluate 0.1.0 at `d8fa2b168e531910a5da01b9d803c05a04f140e6`. Exit code 0.

Fixtures: `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu` (from the manifest).

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
| `export.last_hidden_state` | float.last_hidden_state_max_abs |  |  |  |  |  |  | not-applicable |
| `export.embedding` | float.embedding_max_abs | 1e-05 | 2.32e-07 | 1.95e-08 | 7.14e-08 | 0.999999999999 | 46 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 1e-05 | 2.06e-07 | 2.13e-08 | 7.6e-08 | 1.000000000000 | 6 | passed |
| `export.batch.short-b8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 8 | passed |
| `export.batch.short-b8.vs_reference` | float.embedding_max_abs | 1e-05 | 2.06e-07 | 2.42e-08 | 8.53e-08 | 1.000000000000 | 8 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 1e-05 | 1.4e-07 | 2.14e-08 | 7.4e-08 | 1.000000000000 | 46 | passed |
| `dotnet.embedding.vs_native_same_graph` | browser_vs_native_same_graph.embedding_max_abs |  | 2.01e-07 | 1.67e-08 | 6.3e-08 | 1.000000000000 | 46 | not-applicable |
| `dotnet.text_embedding` | float.embedding_max_abs | 1e-05 | 1.35e-07 | 2.15e-08 | 7.4e-08 | 1.000000000000 | 46 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch.short-b8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 1.02e-07 | 1.08e-08 | 5.6e-08 | 1.000000000000 | 8 | passed |
| `dotnet.batch.short-b8.vs_reference` | float.embedding_max_abs | 1e-05 | 1.3e-07 | 2.61e-08 | 8.63e-08 | 1.000000000000 | 8 | passed |
| `dotnet.batch.generator-all-cases.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 1.02e-07 | 9.18e-09 | 4.4e-08 | 1.000000000000 | 46 | passed |
| `dotnet.batch.generator-all-cases.vs_reference` | float.embedding_max_abs | 1e-05 | 1.35e-07 | 2.15e-08 | 7.4e-08 | 1.000000000000 | 46 | passed |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `export.last_hidden_state`: report only: float.last_hidden_state_max_abs is not in the protected tolerances (export results need tensors 'last_hidden_state' and 'reference_last_hidden_state')
- `dotnet.embedding.vs_native_same_graph`: report only: browser_vs_native_same_graph.embedding_max_abs is not in the protected tolerances

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `634fb790b1d1d2780fad55148c4546c31b1fffa0`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-wasm/tolerances.yaml` | `a4bb1ee25d5fdfbc9c66360bc2b65ab4d9c84c664d75db05a9f9eb339e8add3b` |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/tokenizer.jsonl` | `cb0e61fe1de97956fef1a847a102f2aadd675a2ed7efbfc1a48ac9db2f4b61a1` |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl` | `cb7c1fcd618bcddeb664dc05fb14f896c253b1151c2969ab87a1b819a6415ae1` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Models.E5.Browser on onnxruntime-web 1.30.0 (wasm, bundle wasm, 4 threads), build interp: E5 tokenization (owned tokenizer, truncation to 512) | browser/conformance/run_conformance.py (E5Conformance) via model-pipeline browser-results `d8fa2b168e531910a5da01b9d803c05a04f140e6` | omarchy | `42443ebf132e29e1` |
| export | Python onnxruntime 1.30.0 CPUExecutionProvider on onnx/model.onnx (the browser profile's graph), float64 pooling; no hidden-state reference (not bound for browser profiles) | model-pipeline/reference/encoder/stage_e5_browser_native.py `d8fa2b168e531910a5da01b9d803c05a04f140e6` | omarchy | `5db786134c2fded9` |
| dotnet | Starlings.Models.E5.Browser on onnxruntime-web 1.30.0 (wasm, bundle wasm, 4 threads), build interp | browser/conformance/run_conformance.py (E5Conformance) via model-pipeline browser-results `d8fa2b168e531910a5da01b9d803c05a04f140e6` | omarchy | `7a08364c7b668eb3` |
