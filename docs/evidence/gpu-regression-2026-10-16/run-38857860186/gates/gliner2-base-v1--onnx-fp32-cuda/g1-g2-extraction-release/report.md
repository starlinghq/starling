# Fidelity report: gliner2-base-v1/onnx-fp32-cuda (entity-extraction)

Date 2026-10-16. Evaluator model-pipeline evaluate 0.1.0 at `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8`. Exit code 0.

Fixtures: `model-pipeline/protected/gliner2-base-v1/release/extraction-release` (from the --fixtures option).

First divergence: **none**.

## Gates

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every check passed or is not applicable by the manifest |
| G2 | passed | every check passed or is not applicable by the manifest |

## Stages

| Stage | Status |
| --- | --- |
| Stage 1, processor (normalized text, words, maps, ids, positions vs protected fixtures) | passed |
| Stage 2, export (Python ONNX Runtime vs PyTorch reference) | passed |
| Stage 3, .NET from fixture inputs | passed |
| Stage 4, .NET end to end from text | passed |

## Checks

Absolute errors; `Max rel` is |d| / max(1, |ref|). Bounds come only from the protected tolerances file.

| Check | Tolerance key | Bound | Max abs | Mean abs | p99 abs | Max rel | Cases | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dotnet.identity` |  |  |  |  |  |  |  | passed |
| `export.identity` |  |  |  |  |  |  |  | passed |
| `tokenizer.identity` |  |  |  |  |  |  |  | passed |
| `tokenizer.normalized_text` | discrete.normalized_text | exact |  |  |  |  | 30 | passed |
| `tokenizer.words` | discrete.words | exact |  |  |  |  | 30 | passed |
| `tokenizer.start_map` | discrete.character_maps | exact |  |  |  |  | 30 | passed |
| `tokenizer.end_map` | discrete.character_maps | exact |  |  |  |  | 30 | passed |
| `tokenizer.input_ids` | discrete.token_ids | exact |  |  |  |  | 30 | passed |
| `tokenizer.attention_mask` | discrete.attention_mask | exact |  |  |  |  | 30 | passed |
| `tokenizer.word_index` | discrete.word_positions | exact |  |  |  |  | 30 | passed |
| `tokenizer.schema_index` | discrete.schema_positions | exact |  |  |  |  | 30 | passed |
| `tokenizer.character_maps.well_formed` |  |  |  |  |  |  |  | passed |
| `export.inputs` |  |  |  |  |  |  |  | passed |
| `export.provider_options` |  | use_tf32 0 |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs | 0.0001 | 1.72e-05 | 3.11e-07 | 2.12e-06 |  | 30 | passed |
| `export.count_logits` | float.count_logits_max_abs | 0.0005 | 3.8e-05 | 1.17e-05 | 3.1e-05 |  | 30 | passed |
| `export.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 30 | passed |
| `export.span_logits` | float.span_logits_relative | 0.0001 | 0.00076 | 2.75e-05 | 0.000256 | 1.4e-05 | 30 | passed |
| `export.scores` | float.scores_max_abs | 2e-05 | 3.37e-06 | 3.32e-09 | 2.79e-08 |  | 30 | passed |
| `export.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 30 | passed |
| `export.entities.confidence` | float.confidence_max_abs | 2e-05 | 1.91e-06 | 4.19e-07 | 1.79e-06 |  | 28 | passed |
| `export.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `export.batch.by-label-count-batch8.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00075 | 2.37e-05 | 0.000221 | 8.67e-06 | 30 | passed |
| `export.batch.by-label-count-batch8.vs_reference` | float.span_logits_relative | 0.0001 | 0.00137 | 3.32e-05 | 0.000276 | 1.49e-05 | 30 | passed |
| `export.batch.by-label-count-batch8-count.vs_single` | float.count_logits_max_abs | 0.0005 | 3e-05 | 8.43e-06 | 2.7e-05 |  | 30 | passed |
| `export.batch.by-label-count-batch8-count.vs_reference` | float.count_logits_max_abs | 0.0005 | 3.6e-05 | 1.15e-05 | 3e-05 |  | 30 | passed |
| `dotnet.inputs` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  | use_tf32 0 |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.count_logits` | float.count_logits_max_abs | 0.0005 | 3.8e-05 | 1.17e-05 | 3.1e-05 |  | 30 | passed |
| `dotnet.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 30 | passed |
| `dotnet.span_logits` | float.span_logits_relative | 0.0001 | 0.00076 | 2.75e-05 | 0.000256 | 1.4e-05 | 30 | passed |
| `dotnet.scores` | float.scores_max_abs | 2e-05 | 3.37e-06 | 3.32e-09 | 2.79e-08 |  | 30 | passed |
| `dotnet.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 30 | passed |
| `dotnet.entities.confidence` | float.confidence_max_abs | 2e-05 | 1.91e-06 | 4.25e-07 | 1.8e-06 |  | 28 | passed |
| `dotnet.span_logits.vs_python` | csharp_vs_python_same_machine_and_provider.span_logits_relative | 1e-06 | 0 | 0 | 0 | 0 | 30 | passed |
| `dotnet.text_entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 30 | passed |
| `dotnet.text_entities.confidence` | float.confidence_max_abs | 2e-05 | 1.91e-06 | 4.25e-07 | 1.8e-06 |  | 28 | passed |
| `dotnet.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_entities` | entities.identity | exact |  |  |  |  |  | passed |
| `dotnet.batch.span-logits-labels6-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.0007 | 2.17e-05 | 0.000249 | 8.4e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00137 | 3.63e-05 | 0.000325 | 1.21e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part1.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00042 | 2e-05 | 0.00014 | 8.06e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part1.vs_reference` | float.span_logits_relative | 0.0001 | 0.000705 | 2.81e-05 | 0.000222 | 1.49e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part2.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00066 | 2.75e-05 | 0.00025 | 6.28e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part2.vs_reference` | float.span_logits_relative | 0.0001 | 0.00086 | 3.86e-05 | 0.00033 | 1.46e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part3.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00075 | 2.69e-05 | 0.000223 | 8.67e-06 | 6 | passed |
| `dotnet.batch.span-logits-labels6-part3.vs_reference` | float.span_logits_relative | 0.0001 | 0.0004 | 2.62e-05 | 0.000161 | 9.65e-06 | 6 | passed |
| `dotnet.batch.text-entities-labels6-part0.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels6-part0.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels6-part1.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels6-part1.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels6-part2.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels6-part2.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels6-part3.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 6 | passed |
| `dotnet.batch.text-entities-labels6-part3.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 6 | passed |

## Allowed entity exceptions

None: every compared entity list matched exactly, in order.

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cuda/tolerances.yaml` | `6735d00f44f07ad7739a5e921f4496ded0921e515a3423e704c039b12c683957` |
| `model-pipeline/protected/gliner2-base-v1/release/extraction-release/extraction.jsonl` | `dac63a49413cd0fe382b7cecebaa51733e5027ce2589e1329f7735e6974389ec` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Models.Gliner2 processor (Python-compatible normalization, splitter and lower-casing; schema prefix through the owned tokenizer, profile gliner2-base-v1-unigram) | ai-dotnet/tools/Starlings.Fidelity.Tool `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8` | 65fb18f5f39a | `0675f60624929039` |
| export | Python onnxruntime 1.30.0 CUDAExecutionProvider on gliner2-entities-fp32.onnx (route A, fixture inputs); entities decoded with gliner2_entity.decode_entities; reference hidden states from the GLiNER2 encoder on torch 2.14.1+cu130 fp32 CPU, same ids | model-pipeline/reference/gliner2/stage_gliner2.py `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8` | 65fb18f5f39a | `419f4aec835f163e` |
| dotnet | C# Gliner2EntityExtractor: RunModel + decoder on the fixture inputs, and Prepare/RunModel/Decode from text; ONNX Runtime 1.30.0 CUDA (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8` | 65fb18f5f39a | `10a7e3f313ed5f17` |
