# Fidelity report: gliner2-base-v1/onnx-fp32-cpu (entity-extraction)

Date 2026-10-14. Evaluator model-pipeline evaluate 0.1.0 at `10c016c1e8724983306d79a5db02b84326109c32`. Exit code 0.

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
| `export.provider_options` |  |  |  |  |  |  |  | passed |
| `export.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `export.last_hidden_state` | float.last_hidden_state_max_abs | 0.0001 | 2.48e-05 | 3.67e-07 | 2.51e-06 |  | 30 | passed |
| `export.count_logits` | float.count_logits_max_abs | 0.0005 | 4e-05 | 1.01e-05 | 3.8e-05 |  | 30 | passed |
| `export.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 30 | passed |
| `export.span_logits` | float.span_logits_relative | 0.0001 | 0.00085 | 3.85e-05 | 0.00034 | 1.78e-05 | 30 | passed |
| `export.scores` | float.scores_max_abs | 2e-05 | 2.91e-06 | 4.41e-09 | 3.17e-08 |  | 30 | passed |
| `export.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 30 | passed |
| `export.entities.confidence` | float.confidence_max_abs | 2e-05 | 2.86e-06 | 5.3e-07 | 2.59e-06 |  | 28 | passed |
| `export.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `export.batch.by-label-count-batch8.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 30 | passed |
| `export.batch.by-label-count-batch8.vs_reference` | float.span_logits_relative | 0.0001 | 0.00085 | 3.85e-05 | 0.00034 | 1.78e-05 | 30 | passed |
| `export.batch.by-label-count-batch8-count.vs_single` | float.count_logits_max_abs | 0.0005 | 0 | 0 | 0 |  | 30 | passed |
| `export.batch.by-label-count-batch8-count.vs_reference` | float.count_logits_max_abs | 0.0005 | 4e-05 | 1.01e-05 | 3.8e-05 |  | 30 | passed |
| `dotnet.inputs` |  |  |  |  |  |  |  | passed |
| `dotnet.provider_options` |  |  |  |  |  |  |  | passed |
| `dotnet.finite` | float.nan_or_inf | fail |  |  |  |  |  | passed |
| `dotnet.count_logits` | float.count_logits_max_abs | 0.0005 | 4e-05 | 1.01e-05 | 3.8e-05 |  | 30 | passed |
| `dotnet.gate_decision` | discrete.gate_decision | exact |  |  |  |  | 30 | passed |
| `dotnet.span_logits` | float.span_logits_relative | 0.0001 | 0.00085 | 3.85e-05 | 0.00034 | 1.78e-05 | 30 | passed |
| `dotnet.scores` | float.scores_max_abs | 2e-05 | 2.91e-06 | 4.41e-09 | 3.17e-08 |  | 30 | passed |
| `dotnet.entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 30 | passed |
| `dotnet.entities.confidence` | float.confidence_max_abs | 2e-05 | 2.88e-06 | 5.33e-07 | 2.6e-06 |  | 28 | passed |
| `dotnet.span_logits.vs_python` | csharp_vs_python_same_machine_and_provider.span_logits_relative | 1e-06 | 0 | 0 | 0 | 0 | 30 | passed |
| `dotnet.text_entities` | entities.identity | identity exact, order_swap_max_confidence_delta 2e-05, threshold_flip_max_score_margin 2e-05 |  |  |  |  | 30 | passed |
| `dotnet.text_entities.confidence` | float.confidence_max_abs | 2e-05 | 2.88e-06 | 5.33e-07 | 2.6e-06 |  | 28 | passed |
| `dotnet.batch_evidence.span_logits` | float.padded_batch_vs_single_span_logits_relative | 0.0001 |  |  |  |  |  | passed |
| `dotnet.batch_evidence.text_entities` | entities.identity | exact |  |  |  |  |  | passed |
| `dotnet.batch.span-logits-labels6-part0.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part0.vs_reference` | float.span_logits_relative | 0.0001 | 0.00085 | 4.36e-05 | 0.000419 | 1.78e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part1.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part1.vs_reference` | float.span_logits_relative | 0.0001 | 0.000709 | 3.43e-05 | 0.000269 | 1.12e-05 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part2.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part2.vs_reference` | float.span_logits_relative | 0.0001 | 0.00062 | 4.09e-05 | 0.00036 | 9.03e-06 | 8 | passed |
| `dotnet.batch.span-logits-labels6-part3.vs_single` | float.padded_batch_vs_single_span_logits_relative | 0.0001 | 0 | 0 | 0 | 0 | 6 | passed |
| `dotnet.batch.span-logits-labels6-part3.vs_reference` | float.span_logits_relative | 0.0001 | 0.000584 | 2.97e-05 | 0.00021 | 9.6e-06 | 6 | passed |
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

Status **verified** (merge base of HEAD and origin/main; base `10c016c1e8724983306d79a5db02b84326109c32`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/tolerances.yaml` | `522ffd8c4945d3b4d7d09368b8d6cab18e87044c4e6f8009c0a40aec91c67f85` |
| `model-pipeline/protected/gliner2-base-v1/release/extraction-release/extraction.jsonl` | `dac63a49413cd0fe382b7cecebaa51733e5027ce2589e1329f7735e6974389ec` |

## Candidate inputs

| Stage | Candidate | Producer | Machine | Results SHA-256 |
| --- | --- | --- | --- | --- |
| tokenizer | Starlings.Models.Gliner2 processor (Python-compatible normalization, splitter and lower-casing; schema prefix through the owned tokenizer, profile gliner2-base-v1-unigram) | ai-dotnet/tools/Starlings.Fidelity.Tool `10c016c1e8724983306d79a5db02b84326109c32` | omarchy | `75cc65e0b16ea66f` |
| export | Python onnxruntime 1.30.0 CPUExecutionProvider on gliner2-entities-fp32.onnx (route A, fixture inputs); entities decoded with gliner2_entity.decode_entities; reference hidden states from the GLiNER2 encoder on torch 2.14.1+cpu fp32 CPU, same ids | model-pipeline/reference/gliner2/stage_gliner2.py `10c016c1e8724983306d79a5db02b84326109c32` | omarchy | `0c76415c7d6227aa` |
| dotnet | C# Gliner2EntityExtractor: RunModel + decoder on the fixture inputs, and Prepare/RunModel/Decode from text; ONNX Runtime 1.30.0 CPU (managed binding 1.30.0) | ai-dotnet/tools/Starlings.Fidelity.Tool `10c016c1e8724983306d79a5db02b84326109c32` | omarchy | `64a3f9f8569a404d` |
