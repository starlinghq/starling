# Fidelity report: qwen3.5-0.8b/ortgenai-fp32x-cuda (generation)

Date 2026-10-14. Evaluator model-pipeline evaluate 0.1.0 at `c33d63c1d28c90273295600e641cc3e1fa61d583`. Exit code 4.

Protected evidence: **verified** (merge base of HEAD and origin/main).

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every check passed or is not applicable by the manifest |
| G2 | not-run | checks not run: dotnet-text.chunked_prefill |

First divergence: **none**.

## Template stage: owned template rendering and owned tokenizer ids against the fixture: passed

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| tokenizer.template_rendering | discrete.template_rendering | exact | passed | compared 21, failed 0 |  |
| tokenizer.prompt_ids | discrete.prompt_ids | exact | passed | compared 21, failed 0 |  |

## Export (not applicable: no Python ONNX Runtime GenAI stage is bound): incomplete

No checks.

## .NET greedy generation from the fixture prompt ids: passed

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| dotnet-ids.identity |  |  | passed | cases 22, environment True |  |
| dotnet-ids.greedy_ids | discrete.greedy_ids | exact_except_listed_margin_below_1e-3 | passed | exact 22, compared 22, divergences [] |  |
| dotnet-ids.finish_reason | discrete.finish_reason | equal | passed | equal 22, compared 22 |  |
| dotnet-ids.nan_or_inf | float.nan_or_inf | fail | passed | non_finite_cases [] |  |
| dotnet-ids.step0_argmax | discrete.step0_argmax | equal | passed | equal 22 |  |
| dotnet-ids.step0_top5_logprob | float.step0_top5_logprob_max_abs | 0.002 | passed | max_abs 0.000594, worst_case tools |  |
| dotnet-ids.step0_logprob | float.step0_logprob_max_abs | 0.005 | passed | top64_max_abs 0.00101, full_vocabulary_max_abs 0.00135, worst_case tools |  |
| dotnet-ids.streaming_reassembly | discrete.streaming_reassembly | exact | passed | failed [] |  |
| dotnet-ids.usage | discrete.usage | exact | passed | failed [] |  |
| dotnet-ids.provider_options | provider_options | {'use_tf32': 0} | passed |  |  |

## .NET runtime semantics (cached vs uncached, chunked prefill, concurrency): incomplete

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| dotnet-text.cached_vs_uncached | float.cached_vs_uncached_logprob_max_abs | 0.0002 | passed | max_abs 4.67e-05, steps 12, generated_equals_reference_prefix True |  |
| dotnet-text.chunked_prefill | float.chunked_vs_one_call_prefill_logprob_max_abs |  | not-run | max_abs 5.91e-05, not_measured {} | tolerance key float.chunked_vs_one_call_prefill_logprob_max_abs is not set in the protected tolerances file |
| dotnet-text.concurrent_requests | discrete.concurrent_requests | equal_to_sequential | passed |  |  |

## Excluded by contract

- `tools`: tools prompt: unsupported by the owned renderer (manifest behavior.unsupported)
