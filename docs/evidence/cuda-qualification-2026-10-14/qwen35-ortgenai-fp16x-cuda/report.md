# Fidelity report: qwen3.5-0.8b/ortgenai-fp16x-cuda (generation)

Date 2026-10-14. Evaluator model-pipeline evaluate 0.1.0 at `c33d63c1d28c90273295600e641cc3e1fa61d583`. Exit code 0.

Protected evidence: **verified** (merge base of HEAD and origin/main).

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every check passed or is not applicable by the manifest |
| G2 | passed | every check passed or is not applicable by the manifest |

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
| dotnet-ids.greedy_ids | discrete.greedy_ids | equal_until_first_divergence_at_margin_below_0.05 | passed | exact 19, compared 22 | allowed near-tie divergences: code-sql@112 (0.00194), special-think-tool@95 (0.00272), thinking-on@76 (0.000446) |
| dotnet-ids.finish_reason | discrete.finish_reason | equal_when_ids_equal | passed | equal 22, compared 22 |  |
| dotnet-ids.nan_or_inf | float.nan_or_inf | fail | passed | non_finite_cases [] |  |
| dotnet-ids.step0_argmax | discrete.step0_argmax | equal | passed | equal 22 |  |
| dotnet-ids.step0_top5_logprob | float.step0_top5_logprob_max_abs | 0.06 | passed | max_abs 0.0414, worst_case en-short |  |
| dotnet-ids.step0_logprob | float.step0_logprob_max_abs | 0.15 | passed | top64_max_abs 0.0428, full_vocabulary_max_abs 0.0607, worst_case thinking-on |  |
| dotnet-ids.streaming_reassembly | discrete.streaming_reassembly | exact | passed | failed [] |  |
| dotnet-ids.usage | discrete.usage | exact | passed | failed [] |  |
| dotnet-ids.provider_options | provider_options | {'use_tf32': 0} | passed |  |  |

## .NET runtime semantics (cached vs uncached, chunked prefill, concurrency): passed

| Check | Key | Bound | Status | Metrics | Detail |
| --- | --- | --- | --- | --- | --- |
| dotnet-text.cached_vs_uncached | float.cached_vs_uncached_logprob_max_abs | 0.1 | passed | max_abs 0.0605, steps 12, generated_equals_reference_prefix True |  |
| dotnet-text.chunked_prefill | float.chunked_vs_one_call_prefill_logprob_max_abs | 0.1 | passed | max_abs 0.0485, not_measured {} |  |
| dotnet-text.concurrent_requests | discrete.concurrent_requests | equal_to_sequential | passed |  |  |

## Greedy divergences

| Case | Step | Reference margin |
| --- | ---: | ---: |
| code-sql | 112 | 0.00194 |
| special-think-tool | 95 | 0.00272 |
| thinking-on | 76 | 0.000446 |

## Excluded by contract

- `tools`: tools prompt: unsupported by the owned renderer (manifest behavior.unsupported)
