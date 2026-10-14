# Fidelity report: gliner2-base-v1/onnx-fp32-cuda (entity-extraction)

Date 2026-10-14. Evaluator model-pipeline evaluate 0.1.0 at `2db3dce7ca30269ab9c00eb4b21985d09523969d`. Exit code 0.

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
| `tokenizer.normalized_text` | discrete.normalized_text | exact |  |  |  |  | 58 | passed |
| `tokenizer.words` | discrete.words | exact |  |  |  |  | 58 | passed |
| `tokenizer.start_map` | discrete.character_maps | exact |  |  |  |  | 58 | passed |
| `tokenizer.end_map` | discrete.character_maps | exact |  |  |  |  | 58 | passed |
| `tokenizer.input_ids` | discrete.token_ids | exact |  |  |  |  | 58 | passed |
| `tokenizer.attention_mask` | discrete.attention_mask | exact |  |  |  |  | 58 | passed |
| `tokenizer.word_index` | discrete.word_positions | exact |  |  |  |  | 58 | passed |
| `tokenizer.schema_index` | discrete.schema_positions | exact |  |  |  |  | 58 | passed |
| `tokenizer.character_maps.well_formed` |  |  |  |  |  |  |  | passed |
| `export.inputs` |  |  |  |  |  |  |  | passed |
| `export.provider_options` |  | use_tf32 0 |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs | 0.0001 | 2.48e-05 | 3.49e-07 | 2.32e-06 |  | 58 | passed |
| `export.count_logits` | float.count_logits_max_abs | 0.0005 | 4.9e-05 | 9.89e-06 | 3.64e-05 |  | 58 | passed |
| `export.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 58 | passed |
| `export.span_logits` | float.span_logits_relative | 0.0001 | 0.00306 | 7.65e-05 | 0.000515 | 1.42e-05 | 58 | passed |
| `export.scores` | float.scores_max_abs | 2e-05 | 2.26e-06 | 4.26e-10 | 3.24e-11 |  | 58 | passed |
| `export.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 58 | passed |
| `export.entities.confidence` | float.confidence_max_abs | 2e-05 | 1.13e-06 | 6.57e-08 | 8.48e-07 |  | 54 | passed |
| `export.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `export.batch.by-label-count-batch8.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00304 | 5.67e-05 | 0.00043 | 1.59e-05 | 56 | passed |
| `export.batch.by-label-count-batch8.vs_reference` | float.span_logits_relative | 0.0001 | 0.004 | 7.71e-05 | 0.000525 | 1.47e-05 | 56 | passed |
| `export.batch.by-label-count-batch8-count.vs_single` | float.count_logits_max_abs | 0.0005 | 3.4e-05 | 7.39e-06 | 2.7e-05 |  | 56 | passed |
| `export.batch.by-label-count-batch8-count.vs_reference` | float.count_logits_max_abs | 0.0005 | 4.9e-05 | 1.08e-05 | 3.7e-05 |  | 56 | passed |
| `dotnet.inputs` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  | use_tf32 0 |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.count_logits` | float.count_logits_max_abs | 0.0005 | 4.9e-05 | 9.89e-06 | 3.64e-05 |  | 58 | passed |
| `dotnet.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 58 | passed |
| `dotnet.span_logits` | float.span_logits_relative | 0.0001 | 0.00306 | 7.65e-05 | 0.000515 | 1.42e-05 | 58 | passed |
| `dotnet.scores` | float.scores_max_abs | 2e-05 | 2.26e-06 | 4.26e-10 | 3.24e-11 |  | 58 | passed |
| `dotnet.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 58 | passed |
| `dotnet.entities.confidence` | float.confidence_max_abs | 2e-05 | 1.11e-06 | 6.9e-08 | 8.48e-07 |  | 54 | passed |
| `dotnet.span_logits.vs_python` | csharp_vs_python_same_machine_and_provider.span_logits_relative | 1e-06 | 0 | 0 | 0 | 0 | 58 | passed |
| `dotnet.text_entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 58 | passed |
| `dotnet.text_entities.confidence` | float.confidence_max_abs | 2e-05 | 1.11e-06 | 6.9e-08 | 8.48e-07 |  | 54 | passed |
| `dotnet.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_entities` | entities.identity | exact |  |  |  |  |  | passed |
| `dotnet.batch.span-logits-labels1-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.0004 | 2.63e-05 | 0.000182 | 4.25e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels1-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.0029 | 0.000111 | 0.00244 | 6.73e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00304 | 8.05e-05 | 0.00101 | 8.14e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.004 | 0.000116 | 0.0013 | 9.2e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part1.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00092 | 5.16e-05 | 0.000398 | 1.05e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part1.vs_reference` | float.span_logits_relative | 0.0001 | 0.00105 | 6.55e-05 | 0.000448 | 3.63e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part2.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.0004 | 3.98e-05 | 0.000206 | 2.47e-06 | 2 | passed |
| `dotnet.batch.span-logits-labels2-part2.vs_reference` | float.span_logits_relative | 0.0001 | 0.0006 | 8.25e-05 | 0.0005 | 4.31e-06 | 2 | passed |
| `dotnet.batch.span-logits-labels3-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00068 | 4.37e-05 | 0.0003 | 3.73e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels3-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00073 | 6.01e-05 | 0.0004 | 9.3e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels3-part1.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00299 | 6.24e-05 | 0.000442 | 1.15e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels3-part1.vs_reference` | float.span_logits_relative | 0.0001 | 0.00159 | 8.49e-05 | 0.00054 | 1.47e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels4-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00016 | 2e-05 | 0.00011 | 3.47e-06 | 3 | passed |
| `dotnet.batch.span-logits-labels4-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00042 | 3.54e-05 | 0.000262 | 6.7e-06 | 3 | passed |
| `dotnet.batch.span-logits-labels5-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00039 | 2.66e-05 | 0.00012 | 1.34e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels5-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00038 | 2.65e-05 | 0.000132 | 1.12e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels5-part1.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0.00018 | 2.5e-05 | 0.00012 | 1.59e-05 | 3 | passed |
| `dotnet.batch.span-logits-labels5-part1.vs_reference` | float.span_logits_relative | 0.0001 | 0.00022 | 3.02e-05 | 0.000164 | 1.37e-05 | 3 | passed |
| `dotnet.batch.text-entities-labels1-part0.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels1-part0.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels2-part0.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels2-part0.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels2-part1.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels2-part1.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels2-part2.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 2 | passed |
| `dotnet.batch.text-entities-labels2-part2.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 2 | passed |
| `dotnet.batch.text-entities-labels3-part0.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels3-part0.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels3-part1.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 5 | passed |
| `dotnet.batch.text-entities-labels3-part1.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 5 | passed |
| `dotnet.batch.text-entities-labels4-part0.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 3 | passed |
| `dotnet.batch.text-entities-labels4-part0.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 3 | passed |
| `dotnet.batch.text-entities-labels5-part0.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels5-part0.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 8 | passed |
| `dotnet.batch.text-entities-labels5-part1.vs_single` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 3 | passed |
| `dotnet.batch.text-entities-labels5-part1.vs_reference` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 3 | passed |

## Allowed entity exceptions

None: every compared entity list matched exactly, in order.

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `8feb5069ac6f830f161366d8d2b6fc61cd42a0b3`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cuda/tolerances.yaml` | `6735d00f44f07ad7739a5e921f4496ded0921e515a3423e704c039b12c683957` |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/extraction.jsonl` | `5b381443248fac8a454ec2b6abda166f5b7488a6d05ef954db7a65384a3a5ae0` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Models.Gliner2 processor (Python-compatible normalization, splitter and lower-casing; schema prefix through the owned tokenizer, profile gliner2-base-v1-unigram) | ai-dotnet/tools/Starlings.Fidelity.Tool `2db3dce7ca30269ab9c00eb4b21985d09523969d` | omarchy | `a307f8385c059981` |
| export | Python onnxruntime 1.30.0 CUDAExecutionProvider on gliner2-entities-fp32.onnx (route A, fixture inputs); entities decoded with gliner2_entity.decode_entities; reference hidden states from the GLiNER2 encoder on torch 2.14.1+cu130 fp32 CPU, same ids | model-pipeline/reference/gliner2/stage_gliner2.py `2db3dce7ca30269ab9c00eb4b21985d09523969d` | omarchy | `91c54240d195a8f6` |
| dotnet | C# Gliner2EntityExtractor: RunModel + decoder on the fixture inputs, and Prepare/RunModel/Decode from text; ONNX Runtime 1.30.0 CUDA (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `2db3dce7ca30269ab9c00eb4b21985d09523969d` | omarchy | `d1dafd9f896f0ff9` |
