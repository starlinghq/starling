# Fidelity report: multilingual-e5-small/onnx-int8embed-webgpu

Date 2026-10-23. Evaluator model-pipeline evaluate 0.1.0 at `ccc20820389955e70789f7250d40d18b88a3f90e`. Exit code 0.

Fixtures: `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu` (from the manifest).

First divergence: **none**.

## Gates

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every check passed or is not applicable by the manifest |
| G2 | passed | every check passed or is not applicable by the manifest |
| Preview requirement `task_metric_before_preview.retrieval_top10_overlap_vs_fp32` (not G1 or G2) | not-run | gates Preview, not G1 or G2. Not evaluated: it needs this profile's embeddings of a P-030 held-out query set (the profile's budgeted G3 retrieval set) and the fp32 baseline's, through `model-pipeline evaluate --g3 retrieval --baseline <fp32 embeddings>`; a staged fidelity run has no retrieval outputs |

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
| `tokenizer.input_ids` | discrete.token_ids | exact |  |  |  |  | 46 | passed |
| `tokenizer.attention_mask` | discrete.attention_mask | exact |  |  |  |  | 46 | passed |
| `tokenizer.offsets` | discrete.offsets | exact |  |  |  |  |  | not-applicable |
| `export.input_ids` |  |  |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs |  |  |  |  |  |  | not-applicable |
| `export.embedding` | float.embedding_max_abs | 0.01 | 0.00351 | 0.000383 | 0.00155 | 0.999803821388 | 46 | passed |
| `export.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999803821388 | 46 | passed |
| `export.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `export.batch.fixture-padded.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 6 | passed |
| `export.batch.fixture-padded.vs_reference` | float.embedding_max_abs | 0.01 | 0.00138 | 0.000306 | 0.00114 | 0.999954892563 | 6 | passed |
| `export.batch.fixture-padded.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999954892563 | 6 | passed |
| `export.batch.short-b8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 0 | 0 | 0 | 1.000000000000 | 8 | passed |
| `export.batch.short-b8.vs_reference` | float.embedding_max_abs | 0.01 | 0.00212 | 0.000332 | 0.00129 | 0.999928932908 | 8 | passed |
| `export.batch.short-b8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999928932908 | 8 | passed |
| `dotnet.input_ids` |  |  |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.embedding` | float.embedding_max_abs | 0.01 | 0.00351 | 0.000383 | 0.00155 | 0.999803824347 | 46 | passed |
| `dotnet.embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999803824347 | 46 | passed |
| `dotnet.embedding.vs_native_same_graph` | browser_vs_native_same_graph.embedding_max_abs | 1e-05 | 1.73e-07 | 2.5e-08 | 9e-08 | 0.999999999999 | 46 | passed |
| `dotnet.text_embedding` | float.embedding_max_abs | 0.01 | 0.00351 | 0.000383 | 0.00155 | 0.999803825000 | 46 | passed |
| `dotnet.text_embedding.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999803825000 | 46 | passed |
| `dotnet.batch_evidence.embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_embedding` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 |  |  |  |  |  | passed |
| `dotnet.batch.short-b8.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 1.6e-07 | 1.32e-08 | 7e-08 | 1.000000000000 | 8 | passed |
| `dotnet.batch.short-b8.vs_reference` | float.embedding_max_abs | 0.01 | 0.00212 | 0.000332 | 0.00129 | 0.999928934281 | 8 | passed |
| `dotnet.batch.short-b8.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999928934281 | 8 | passed |
| `dotnet.batch.generator-all-cases.vs_single` | float.padded_batch_vs_single_embedding_max_abs | 1e-05 | 1.6e-07 | 1.12e-08 | 5.4e-08 | 1.000000000000 | 46 | passed |
| `dotnet.batch.generator-all-cases.vs_reference` | float.embedding_max_abs | 0.01 | 0.00351 | 0.000383 | 0.00155 | 0.999803825000 | 46 | passed |
| `dotnet.batch.generator-all-cases.vs_reference.min_cosine` | float.min_cosine | 1 |  |  |  | 0.999803825000 | 46 | passed |
| `preview.retrieval_top10_overlap_vs_fp32` | task_metric_before_preview.retrieval_top10_overlap_vs_fp32 | 0.95 |  |  |  |  |  | not-run |

Details:

- `tokenizer.offsets`: manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets
- `export.last_hidden_state`: report only: float.last_hidden_state_max_abs is not in the protected tolerances (export results need tensors 'last_hidden_state' and 'reference_last_hidden_state')
- `preview.retrieval_top10_overlap_vs_fp32`: gates Preview, not G1 or G2. Not evaluated: it needs this profile's embeddings of a P-030 held-out query set (the profile's budgeted G3 retrieval set) and the fp32 baseline's, through `model-pipeline evaluate --g3 retrieval --baseline <fp32 embeddings>`; a staged fidelity run has no retrieval outputs

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `634fb790b1d1d2780fad55148c4546c31b1fffa0`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/multilingual-e5-small/onnx-int8embed-webgpu/tolerances.yaml` | `1b9f979874c522759d61e1cf7f3dd5055a146f2f18117ce598cc302e7bc93e75` |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/tokenizer.jsonl` | `cb0e61fe1de97956fef1a847a102f2aadd675a2ed7efbfc1a48ac9db2f4b61a1` |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl` | `cb7c1fcd618bcddeb664dc05fb14f896c253b1151c2969ab87a1b819a6415ae1` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Models.E5.Browser on onnxruntime-web 1.30.0 (webgpu, bundle webgpu, 4 threads), build aot: E5 tokenization (owned tokenizer, truncation to 512) | browser/conformance/run_conformance.py (E5Conformance) via model-pipeline browser-results `ccc20820389955e70789f7250d40d18b88a3f90e` | omarchy | `494c5f8e93e198c2` |
| export | Python onnxruntime 1.30.0 CPUExecutionProvider on onnx/model-int8-embed.onnx (the browser profile's graph), float64 pooling; no hidden-state reference (not bound for browser profiles) | model-pipeline/reference/encoder/stage_e5_browser_native.py `ccc20820389955e70789f7250d40d18b88a3f90e` | omarchy | `43cc554a0a268ff4` |
| dotnet | Starlings.Models.E5.Browser on onnxruntime-web 1.30.0 (webgpu, bundle webgpu, 4 threads), build aot | browser/conformance/run_conformance.py (E5Conformance) via model-pipeline browser-results `ccc20820389955e70789f7250d40d18b88a3f90e` | omarchy | `129f76b9d42d246a` |
