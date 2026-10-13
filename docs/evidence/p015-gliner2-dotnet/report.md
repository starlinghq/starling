# Fidelity report: gliner2-base-v1/onnx-fp32-cpu (entity-extraction)

Date 2026-10-10. Evaluator model-pipeline evaluate 0.1.0 at `6cf3b95ffec3682be3d9eb0307224ce4a244c0b4`. Exit code 0.

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
| `export.provider_options` |  |  |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs | 0.0001 | 2.29e-05 | 3.75e-07 | 2.56e-06 |  | 58 | passed |
| `export.count_logits` | float.count_logits_max_abs | 0.0005 | 4.8e-05 | 1.14e-05 | 3.8e-05 |  | 58 | passed |
| `export.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 58 | passed |
| `export.span_logits` | float.span_logits_relative | 0.0001 | 0.00776 | 9.29e-05 | 0.000696 | 2.4e-05 | 58 | passed |
| `export.scores` | float.scores_max_abs | 2e-05 | 3.64e-06 | 7.66e-10 | 2.5e-11 |  | 58 | passed |
| `export.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 58 | passed |
| `export.entities.confidence` | float.confidence_max_abs | 2e-05 | 1.82e-06 | 1.46e-07 | 1.82e-06 |  | 54 | passed |
| `export.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `export.batch.by-label-count-batch8.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 56 | passed |
| `export.batch.by-label-count-batch8.vs_reference` | float.span_logits_relative | 0.0001 | 0.00776 | 8.84e-05 | 0.000649 | 2.07e-05 | 56 | passed |
| `export.batch.by-label-count-batch8-count.vs_single` | float.count_logits_max_abs | 0.0005 | 0 | 0 | 0 |  | 56 | passed |
| `export.batch.by-label-count-batch8-count.vs_reference` | float.count_logits_max_abs | 0.0005 | 4.8e-05 | 1.16e-05 | 3.8e-05 |  | 56 | passed |
| `dotnet.inputs` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  |  |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.count_logits` | float.count_logits_max_abs | 0.0005 | 4.8e-05 | 1.14e-05 | 3.8e-05 |  | 58 | passed |
| `dotnet.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 58 | passed |
| `dotnet.span_logits` | float.span_logits_relative | 0.0001 | 0.00776 | 9.29e-05 | 0.000696 | 2.4e-05 | 58 | passed |
| `dotnet.scores` | float.scores_max_abs | 2e-05 | 3.64e-06 | 7.66e-10 | 2.5e-11 |  | 58 | passed |
| `dotnet.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 58 | passed |
| `dotnet.entities.confidence` | float.confidence_max_abs | 2e-05 | 1.81e-06 | 1.5e-07 | 1.81e-06 |  | 54 | passed |
| `dotnet.span_logits.vs_python` | csharp_vs_python_same_machine_and_provider.span_logits_relative | 1e-06 | 0 | 0 | 0 | 0 | 58 | passed |
| `dotnet.text_entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 58 | passed |
| `dotnet.text_entities.confidence` | float.confidence_max_abs | 2e-05 | 1.81e-06 | 1.5e-07 | 1.81e-06 |  | 54 | passed |
| `dotnet.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_entities` | entities.identity | exact |  |  |  |  |  | passed |
| `dotnet.batch.span-logits-labels1-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 8 | passed |
| `dotnet.batch.span-logits-labels1-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00176 | 0.000119 | 0.00138 | 7.37e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00776 | 0.000181 | 0.00268 | 1.93e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part1.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part1.vs_reference` | float.span_logits_relative | 0.0001 | 0.00075 | 7.45e-05 | 0.00044 | 4.84e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels2-part2.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 2 | passed |
| `dotnet.batch.span-logits-labels2-part2.vs_reference` | float.span_logits_relative | 0.0001 | 0.00206 | 0.000191 | 0.00158 | 1.31e-05 | 2 | passed |
| `dotnet.batch.span-logits-labels3-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 8 | passed |
| `dotnet.batch.span-logits-labels3-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00171 | 0.000101 | 0.00074 | 7.9e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels3-part1.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 8 | passed |
| `dotnet.batch.span-logits-labels3-part1.vs_reference` | float.span_logits_relative | 0.0001 | 0.00712 | 9.16e-05 | 0.00063 | 2.07e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels4-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 3 | passed |
| `dotnet.batch.span-logits-labels4-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00078 | 5.26e-05 | 0.000405 | 1.52e-05 | 3 | passed |
| `dotnet.batch.span-logits-labels5-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 8 | passed |
| `dotnet.batch.span-logits-labels5-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00406 | 5.48e-05 | 0.000528 | 1.66e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels5-part1.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 3 | passed |
| `dotnet.batch.span-logits-labels5-part1.vs_reference` | float.span_logits_relative | 0.0001 | 0.00023 | 4.1e-05 | 0.000185 | 7.33e-06 | 3 | passed |
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

Status **verified** (merge base of HEAD and origin/main; base `6dcaad3b0c3d3920a0b3501b00b2bb815126da07`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/tolerances.yaml` | `522ffd8c4945d3b4d7d09368b8d6cab18e87044c4e6f8009c0a40aec91c67f85` |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/extraction.jsonl` | `5b381443248fac8a454ec2b6abda166f5b7488a6d05ef954db7a65384a3a5ae0` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Models.Gliner2 processor (Python-compatible normalization, splitter and lower-casing; schema prefix through the owned tokenizer, profile gliner2-base-v1-unigram) | ai-dotnet/tools/Starlings.Fidelity.Tool `6cf3b95ffec3682be3d9eb0307224ce4a244c0b4` | omarchy | `687c2f1fb23533b9` |
| export | Python onnxruntime 1.30.0 CPUExecutionProvider on gliner2-entities-fp32.onnx (route A, fixture inputs); entities decoded with gliner2_entity.decode_entities; reference hidden states from the GLiNER2 encoder on torch 2.14.1+cpu fp32 CPU, same ids | model-pipeline/reference/gliner2/stage_gliner2.py `554f47f388725c28d3895f995214c7abe35156c7-dirty` | omarchy | `6a5ba7add77b2a30` |
| dotnet | C# Gliner2EntityExtractor: RunModel + decoder on the fixture inputs, and Prepare/RunModel/Decode from text; ONNX Runtime 1.30.0 CPU (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `6cf3b95ffec3682be3d9eb0307224ce4a244c0b4` | omarchy | `ea8987e8ffe716ee` |
