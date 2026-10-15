# Fidelity report: qwen3.5-0.8b/ortgenai-fp16x-cuda (generation)

Date 2026-10-15. Evaluator model-pipeline evaluate 0.1.0 at `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023`. Exit code 4.

Fixtures: `model-pipeline/protected/qwen3.5-0.8b/release/release-prompts/reference` (from the --fixtures option).

Protected evidence: **verified** (merge base of HEAD and origin/main).

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every check passed or is not applicable by the manifest |
| G2 | not-run | checks not run: dotnet-ids.step0_logprob, dotnet-text.cached_vs_uncached, dotnet-text.chunked_prefill, dotnet-text.concurrent_requests |

First divergence: **none**.

## Template stage: owned template rendering and owned tokenizer ids against the fixture: passed

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| tokenizer.template_rendering | discrete.template_rendering | exact | passed | compared 32, failed 0, excluded_by_contract [] |  |
| tokenizer.prompt_ids | discrete.prompt_ids | exact | passed | compared 32, failed 0, excluded_by_contract [] |  |

## Export (not applicable: no Python ONNX Runtime GenAI stage is bound): incomplete

No checks.

## .NET greedy generation from the fixture prompt ids: incomplete

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| dotnet-ids.identity |  |  | passed | cases 32, environment True |  |
| dotnet-ids.greedy_ids | discrete.greedy_ids | equal_until_first_divergence_at_margin_below_0.05 | passed | exact 30, compared 32 | allowed near-tie divergences: oasst1-496a8740@24 (0.00249), oasst1-b3321bbb@17 (0.00655) |
| dotnet-ids.finish_reason | discrete.finish_reason | equal_when_ids_equal | passed | equal 32, compared 32 |  |
| dotnet-ids.nan_or_inf | float.nan_or_inf | fail | passed | non_finite_cases [] |  |
| dotnet-ids.step0_argmax | discrete.step0_argmax | equal | passed | equal 32 |  |
| dotnet-ids.step0_top5_logprob | float.step0_top5_logprob_max_abs | 0.06 | passed | max_abs 0.0273, worst_case oasst1-78aeda3c |  |
| dotnet-ids.step0_logprob | float.step0_logprob_max_abs | 0.15 | not-run | top64_max_abs 0.0414, full_vocabulary_max_abs , worst_case  | reference full step-0 vectors not supplied (--reference-tensors); the reference top-64 ids give 0.0414, a lower bound of the full-vocabulary maximum |
| dotnet-ids.streaming_reassembly | discrete.streaming_reassembly | exact | passed | failed [] |  |
| dotnet-ids.usage | discrete.usage | exact | passed | failed [] |  |
| dotnet-ids.provider_options | provider_options | {'use_tf32': 0} | passed |  |  |

## .NET runtime semantics (cached vs uncached, chunked prefill, concurrency): incomplete

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| dotnet-text.cached_vs_uncached | float.cached_vs_uncached_logprob_max_abs | 0.1 | not-run |  | no semantics.json (runner --modes semantics) |
| dotnet-text.chunked_prefill | float.chunked_vs_one_call_prefill_logprob_max_abs | 0.1 | not-run |  | no semantics.json (runner --modes semantics) |
| dotnet-text.concurrent_requests | discrete.concurrent_requests | equal_to_sequential | not-run |  | no semantics.json (runner --modes semantics) |

## Greedy divergences

| Case | Step | Reference margin |
| --- | ---: | ---: |
| oasst1-496a8740 | 24 | 0.00249 |
| oasst1-b3321bbb | 17 | 0.00655 |
