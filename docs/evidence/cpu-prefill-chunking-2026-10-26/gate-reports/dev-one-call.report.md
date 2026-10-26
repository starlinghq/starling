# Fidelity report: qwen3.5-0.8b/ortgenai-fp32x-cpu (generation)

Date 2026-10-26. Evaluator model-pipeline evaluate 0.1.0 at `c3b8e06c3e6d9c920551d02bf2dde13030c00672-dirty`. Exit code 0.

Fixtures: `model-pipeline/protected/qwen3.5-0.8b/reference` (from the manifest).

Protected evidence: **verified** (merge base of HEAD and origin/main).

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every check passed or is not applicable by the manifest |
| G2 | not-run | checks not run: dotnet-ids.step0_logprob, dotnet-text.cached_vs_uncached, dotnet-text.chunked_prefill, dotnet-text.concurrent_requests |

First divergence: **none**.

## Template stage: owned template rendering and owned tokenizer ids against the fixture: passed

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| tokenizer.template_rendering | discrete.template_rendering | exact | passed | compared 21, failed 0 |  |
| tokenizer.prompt_ids | discrete.prompt_ids | exact | passed | compared 21, failed 0 |  |

## Export (not applicable: no Python ONNX Runtime GenAI stage is bound): incomplete

No checks.

## .NET greedy generation from the fixture prompt ids: incomplete

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| dotnet-ids.identity |  |  | passed | cases 22, environment True |  |
| dotnet-ids.greedy_ids | discrete.greedy_ids | exact_except_listed_margin_below_1e-3 | passed | exact 22, compared 22, divergences [] |  |
| dotnet-ids.finish_reason | discrete.finish_reason | equal | passed | equal 22, compared 22 |  |
| dotnet-ids.nan_or_inf | float.nan_or_inf | fail | passed | non_finite_cases [] |  |
| dotnet-ids.step0_argmax | discrete.step0_argmax | equal | passed | equal 22 |  |
| dotnet-ids.step0_top5_logprob | float.step0_top5_logprob_max_abs | 0.002 | passed | max_abs 0.000622, worst_case tools |  |
| dotnet-ids.step0_logprob | float.step0_logprob_max_abs | 0.005 | not-run | top64_max_abs 0.00101, full_vocabulary_max_abs , worst_case  | reference full step-0 vectors not supplied (--reference-tensors); the reference top-64 ids give 0.00101, a lower bound of the full-vocabulary maximum |
| dotnet-ids.streaming_reassembly | discrete.streaming_reassembly | exact | passed | failed [] |  |
| dotnet-ids.usage | discrete.usage | exact | passed | failed [] |  |
| dotnet-ids.provider_options | provider_options | {} | passed | runner {} |  |

## .NET runtime semantics (cached vs uncached, chunked prefill, concurrency): incomplete

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| dotnet-text.cached_vs_uncached | float.cached_vs_uncached_logprob_max_abs | 0.0002 | not-run |  | no semantics.json (runner --modes semantics) |
| dotnet-text.chunked_prefill | float.chunked_vs_one_call_prefill_logprob_max_abs | 0.0001 | not-run |  | no semantics.json (runner --modes semantics) |
| dotnet-text.concurrent_requests | discrete.concurrent_requests | equal_to_sequential | not-run |  | no semantics.json (runner --modes semantics) |

## Excluded by contract

- `tools`: tools prompt: unsupported by the owned renderer (manifest behavior.unsupported)
