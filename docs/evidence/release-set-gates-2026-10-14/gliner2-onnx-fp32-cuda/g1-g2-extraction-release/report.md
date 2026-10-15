# Fidelity report: gliner2-base-v1/onnx-fp32-cuda (entity-extraction)

Date 2026-10-15. Evaluator model-pipeline evaluate 0.1.0 at `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023`. Exit code 0.

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
| `export.last_hidden_state` | float.last_hidden_state_max_abs | 0.0001 | 1.34e-05 | 3.13e-07 | 2.12e-06 |  | 30 | passed |
| `export.count_logits` | float.count_logits_max_abs | 0.0005 | 3.1e-05 | 1.1e-05 | 3e-05 |  | 30 | passed |
| `export.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 30 | passed |
| `export.span_logits` | float.span_logits_relative | 0.0001 | 0.00146 | 2.79e-05 | 0.00027 | 1.24e-05 | 30 | passed |
| `export.scores` | float.scores_max_abs | 2e-05 | 3.07e-06 | 3.61e-09 | 2.63e-08 |  | 30 | passed |
| `export.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 30 | passed |
| `export.entities.confidence` | float.confidence_max_abs | 2e-05 | 1.91e-06 | 4.53e-07 | 1.81e-06 |  | 28 | passed |
| `export.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `export.batch.by-label-count-batch8.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00058 | 2.01e-05 | 0.00017 | 9.45e-06 | 30 | passed |
| `export.batch.by-label-count-batch8.vs_reference` | float.span_logits_relative | 0.0001 | 0.00116 | 3.12e-05 | 0.0003 | 1.37e-05 | 30 | passed |
| `export.batch.by-label-count-batch8-count.vs_single` | float.count_logits_max_abs | 0.0005 | 2.3e-05 | 5.63e-06 | 1.8e-05 |  | 30 | passed |
| `export.batch.by-label-count-batch8-count.vs_reference` | float.count_logits_max_abs | 0.0005 | 4.6e-05 | 1.18e-05 | 3.6e-05 |  | 30 | passed |
| `dotnet.inputs` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  | use_tf32 0 |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.count_logits` | float.count_logits_max_abs | 0.0005 | 3.1e-05 | 1.1e-05 | 3e-05 |  | 30 | passed |
| `dotnet.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 30 | passed |
| `dotnet.span_logits` | float.span_logits_relative | 0.0001 | 0.00146 | 2.79e-05 | 0.00027 | 1.24e-05 | 30 | passed |
| `dotnet.scores` | float.scores_max_abs | 2e-05 | 3.07e-06 | 3.61e-09 | 2.63e-08 |  | 30 | passed |
| `dotnet.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 30 | passed |
| `dotnet.entities.confidence` | float.confidence_max_abs | 2e-05 | 1.88e-06 | 4.59e-07 | 1.79e-06 |  | 28 | passed |
| `dotnet.span_logits.vs_python` | csharp_vs_python_same_machine_and_provider.span_logits_relative | 1e-06 | 0 | 0 | 0 | 0 | 30 | passed |
| `dotnet.text_entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 30 | passed |
| `dotnet.text_entities.confidence` | float.confidence_max_abs | 2e-05 | 1.88e-06 | 4.59e-07 | 1.79e-06 |  | 28 | passed |
| `dotnet.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_entities` | entities.identity | exact |  |  |  |  |  | passed |
| `dotnet.batch.span-logits-labels6-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00058 | 1.93e-05 | 0.00016 | 9.45e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00108 | 3.18e-05 | 0.000357 | 1.31e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part1.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.000197 | 1.69e-05 | 0.000102 | 6.6e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part1.vs_reference` | float.span_logits_relative | 0.0001 | 0.000395 | 2.6e-05 | 0.000163 | 1.37e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part2.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00041 | 2.35e-05 | 0.000196 | 4.73e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part2.vs_reference` | float.span_logits_relative | 0.0001 | 0.00052 | 3.25e-05 | 0.000246 | 8.91e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part3.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00055 | 2.09e-05 | 0.0002 | 5.81e-06 | 6 | passed |
| `dotnet.batch.span-logits-labels6-part3.vs_reference` | float.span_logits_relative | 0.0001 | 0.00116 | 3.4e-05 | 0.000363 | 1.11e-05 | 6 | passed |
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

Status **verified** (merge base of HEAD and origin/main; base `75574190e77f5528a416e87c6e12ea17e3b85d33`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cuda/tolerances.yaml` | `6735d00f44f07ad7739a5e921f4496ded0921e515a3423e704c039b12c683957` |
| `model-pipeline/protected/gliner2-base-v1/release/extraction-release/extraction.jsonl` | `dac63a49413cd0fe382b7cecebaa51733e5027ce2589e1329f7735e6974389ec` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Models.Gliner2 processor (Python-compatible normalization, splitter and lower-casing; schema prefix through the owned tokenizer, profile gliner2-base-v1-unigram) | ai-dotnet/tools/Starlings.Fidelity.Tool `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023` | omarchy | `3033865f7ac99dfe` |
| export | Python onnxruntime 1.30.0 CUDAExecutionProvider on gliner2-entities-fp32.onnx (route A, fixture inputs); entities decoded with gliner2_entity.decode_entities; reference hidden states from the GLiNER2 encoder on torch 2.14.1+cu130 fp32 CPU, same ids | model-pipeline/reference/gliner2/stage_gliner2.py `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023` | omarchy | `9127d7d39dc4670f` |
| dotnet | C# Gliner2EntityExtractor: RunModel + decoder on the fixture inputs, and Prepare/RunModel/Decode from text; ONNX Runtime 1.30.0 CUDA (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023` | omarchy | `620ab42c8a088a14` |
