# Fidelity report: gliner2-base-v1/onnx-fp32-cpu (entity-extraction)

Date 2026-10-10. Evaluator model-pipeline evaluate 0.1.0 at `626bbbc8d5ac933009aadbfc3652e5127a132c06`. Exit code 4.

**No protected tolerances file** at `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/tolerances.yaml`: every bound is unset, so no gate can pass. Metrics below are measurements only. Expected keys are listed under `expected_tolerance_keys` in report.json.

First divergence: **none**.

## Gates

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | not-run | the profile has no protected tolerances file; no bound is set, so no gate can pass |
| G2 | not-run | the profile has no protected tolerances file; no bound is set, so no gate can pass |

## Stages

| Stage | Status |
| --- | --- |
| Stage 1, processor (normalized text, words, maps, ids, positions vs protected fixtures) | incomplete |
| Stage 2, export (Python ONNX Runtime vs PyTorch reference) | incomplete |
| Stage 3, .NET from fixture inputs | incomplete |
| Stage 4, .NET end to end from text | incomplete |

## Checks

Absolute errors; `Max rel` is |d| / max(1, |ref|). Bounds come only from the protected tolerances file.

| Check | Tolerance key | Bound | Max abs | Mean abs | p99 abs | Max rel | Cases | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `export.identity` |  |  |  |  |  |  |  | passed |
| `tokenizer.identity` |  |  |  |  |  |  |  | passed |
| `tokenizer.normalized_text` | discrete.normalized_text |  |  |  |  |  |  | not-run |
| `tokenizer.words` | discrete.words |  |  |  |  |  |  | not-run |
| `tokenizer.start_map` | discrete.character_maps |  |  |  |  |  |  | not-run |
| `tokenizer.end_map` | discrete.character_maps |  |  |  |  |  |  | not-run |
| `tokenizer.input_ids` | discrete.token_ids |  |  |  |  |  |  | not-run |
| `tokenizer.attention_mask` | discrete.attention_mask |  |  |  |  |  |  | not-run |
| `tokenizer.word_index` | discrete.word_positions |  |  |  |  |  |  | not-run |
| `tokenizer.schema_index` | discrete.schema_positions |  |  |  |  |  |  | not-run |
| `tokenizer.character_maps.well_formed` |  |  |  |  |  |  |  | passed |
| `export.inputs` |  |  |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf |  |  |  |  |  |  | not-run |
| `export.last_hidden_state` | float.last_hidden_state_max_abs |  | 2.29e-05 | 3.75e-07 | 2.56e-06 |  | 58 | not-run |
| `export.count_logits` | float.count_logits_max_abs |  | 4.8e-05 | 1.14e-05 | 3.8e-05 |  | 58 | not-run |
| `export.gate_decision` | discrete.gate_decision |  |  |  |  |  |  | not-run |
| `export.span_logits` | float.span_logits_relative |  | 0.00776 | 9.29e-05 | 0.000696 | 2.4e-05 | 58 | not-run |
| `export.scores` | float.scores_max_abs |  | 3.64e-06 | 7.66e-10 | 2.5e-11 |  | 58 | not-run |
| `export.entities` | entities.identity | identity , order_swap_max_confidence_delta , threshold_flip_max_score_margin  |  |  |  |  |  | not-run |
| `export.entities.confidence` | float.confidence_max_abs |  |  |  |  |  |  | not-run |
| `export.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative |  |  |  |  |  |  | passed |
| `export.batch.by-label-count-batch8.vs_single` | float.padded_batch_vs_single_span_logits_relative |  | 0 | 0 | 0 | 0 | 56 | not-run |
| `export.batch.by-label-count-batch8.vs_reference` | float.span_logits_relative |  | 0.00776 | 8.84e-05 | 0.000649 | 2.07e-05 | 56 | not-run |
| `export.batch.by-label-count-batch8-count.vs_single` | float.count_logits_max_abs |  | 0 | 0 | 0 |  | 56 | not-run |
| `export.batch.by-label-count-batch8-count.vs_reference` | float.count_logits_max_abs |  | 4.8e-05 | 1.16e-05 | 3.8e-05 |  | 56 | not-run |
| `dotnet.count_logits` | float.count_logits_max_abs |  |  |  |  |  |  | not-run |
| `dotnet.gate_decision` | discrete.gate_decision |  |  |  |  |  |  | not-run |
| `dotnet.span_logits` | float.span_logits_relative |  |  |  |  |  |  | not-run |
| `dotnet.scores` | float.scores_max_abs |  |  |  |  |  |  | not-run |
| `dotnet.entities` | entities.identity |  |  |  |  |  |  | not-run |
| `dotnet.finite` | float.nan_or_inf |  |  |  |  |  |  | not-run |
| `dotnet.text_entities` | entities.identity |  |  |  |  |  |  | not-run |

Details:

- `tokenizer.normalized_text`: discrete.normalized_text is not in the protected tolerances
- `tokenizer.words`: discrete.words is not in the protected tolerances
- `tokenizer.start_map`: discrete.character_maps is not in the protected tolerances
- `tokenizer.end_map`: discrete.character_maps is not in the protected tolerances
- `tokenizer.input_ids`: discrete.token_ids is not in the protected tolerances
- `tokenizer.attention_mask`: discrete.attention_mask is not in the protected tolerances
- `tokenizer.word_index`: discrete.word_positions is not in the protected tolerances
- `tokenizer.schema_index`: discrete.schema_positions is not in the protected tolerances
- `export.finite`: float.nan_or_inf: None has no implemented meaning (only 'fail')
- `export.last_hidden_state`: no numeric bound for float.last_hidden_state_max_abs in the protected tolerances
- `export.count_logits`: no numeric bound for float.count_logits_max_abs in the protected tolerances
- `export.gate_decision`: discrete.gate_decision is not in the protected tolerances
- `export.span_logits`: no numeric bound for float.span_logits_relative in the protected tolerances
- `export.scores`: no numeric bound for float.scores_max_abs in the protected tolerances
- `export.entities`: entities.identity, entities.order_swap_max_confidence_delta, entities.threshold_flip_max_score_margin not in the protected tolerances
- `export.entities.confidence`: entities were not compared: entities.identity, entities.order_swap_max_confidence_delta, entities.threshold_flip_max_score_margin not in the protected tolerances
- `export.batch.by-label-count-batch8.vs_single`: no numeric bound for float.padded_batch_vs_single_span_logits_relative in the protected tolerances
- `export.batch.by-label-count-batch8.vs_reference`: no numeric bound for float.span_logits_relative in the protected tolerances
- `export.batch.by-label-count-batch8-count.vs_single`: no numeric bound for float.count_logits_max_abs in the protected tolerances
- `export.batch.by-label-count-batch8-count.vs_reference`: no numeric bound for float.count_logits_max_abs in the protected tolerances
- `dotnet.count_logits`: no results for stage 'dotnet'
- `dotnet.gate_decision`: no results for stage 'dotnet'
- `dotnet.span_logits`: no results for stage 'dotnet'
- `dotnet.scores`: no results for stage 'dotnet'
- `dotnet.entities`: no results for stage 'dotnet'
- `dotnet.finite`: no results for stage 'dotnet'
- `dotnet.text_entities`: no results for stage 'dotnet'

## Allowed entity exceptions

None: every compared entity list matched exactly, in order.

## Protected evidence

Status **verified** (base HEAD; base `626bbbc8d5ac933009aadbfc3652e5127a132c06`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/extraction.jsonl` | `5b381443248fac8a454ec2b6abda166f5b7488a6d05ef954db7a65384a3a5ae0` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | GLiNER2 Python processor (ExtractorCollator, inference mode, fast tokenizer): reproducibility of the fixture processor path, not the owned C# processor | model-pipeline/reference/gliner2/stage_gliner2.py `29df85a12b6c77e4eecaa3cd6b1bf6d79a299153-dirty` | omarchy | `b7b963f400feb1d6` |
| export | Python onnxruntime 1.30.0 CPUExecutionProvider on gliner2-A-dynamo.onnx (route A, fixture inputs); entities decoded with gliner2_entity.decode_entities; reference hidden states from the GLiNER2 encoder on torch 2.14.1+cpu fp32 CPU, same ids | model-pipeline/reference/gliner2/stage_gliner2.py `29df85a12b6c77e4eecaa3cd6b1bf6d79a299153-dirty` | omarchy | `6eecfa615465ace4` |
