# P-016: pilot LLM generation backend (Qwen3.5-0.8B on ONNX Runtime GenAI)

Evidence for backlog P-016 ("pinned generation with validated template, stopping, request state, native compatibility, and declared tokenizer ownership boundary"), 2026-10-10. Packages: `ai-dotnet/src/Starlings.Generation.OnnxGenAI` and `ai-dotnet/src/Starlings.Models.Qwen35` (design and API: `ai-dotnet/README.md`, "Generation backend and pilot LLM"). Profiles: `model-pipeline/registry/qwen3.5-0.8b/{ortgenai-fp32x-cpu,ortgenai-fp32x-cuda,ortgenai-fp16x-cuda}/manifest.yaml`. Decisions it builds on: `docs/kickoff.md` (pilot LLM and generation backend, settled 2026-10-08) and `docs/m0/generation-backend-check.md`.

## 1. Setup

- Host: i7-5820K (6 cores, 12 threads, AVX2), 94 GB RAM, Linux x86-64; RTX 4060 Ti 16 GB as GPU 1 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`), driver 610.57.04, CUDA 13.3 at `/opt/cuda`. GPU 0 was not used. The GPU was shared with another resident process holding about 10.7 GB and often at 100% utilization, so about 5.5 GB were free; the CPU ran other jobs (load average 6 to 16). Timings below are exploratory.
- .NET SDK 10.0.100, runtime 10.0.12; `Microsoft.ML.OnnxRuntimeGenAI` / `.Cuda` 0.17.1 (ONNX Runtime 1.30.0).
- Reference: `model-pipeline/reference/generation/run_reference.py` at repository commit `ee3172db0154fe74782d8fef3c41c01fb3a78703` (clean tree), transformers 5.18.0 fp32 on CPU, 8 threads, greedy, on `prompts-v1.jsonl` (22 prompts). Bit-identical to the M0 reference: same renderings, prompt ids and output ids on 22/22 prompts, step-0 log-softmax vectors equal to the last bit. Proposed as fixtures in a separate pull request (`model-pipeline/protected/qwen3.5-0.8b/reference/`, `PROPOSED.md` there).
- Candidate: `ai-dotnet/tools/Starlings.Qwen35.Fidelity` at commit `2dd211eca89b9f0bf0e5ab1a6a4b7deb57165911` (`environment-*.json`: `source_revision`). The concurrency isolation runs of section 5 used the same runner built from the working tree between `066d012` and `2dd211e` (the serialization option present, defaults as committed), fed the reference prompt ids for generation, and rendering every prompt with the owned template and tokenizer separately. Scored with `model-pipeline/reference/generation/compare.py`. Raw outputs (results, step-0 vectors) are outside Git; the summaries are in this directory.

## 2. Assets

`model-pipeline/reference/generation/build_ortgenai_bundle.py` (onnxruntime-genai 0.17.1 builder, `rel-0.17.1` head `83de55a1f3886cddbb32a5517e26adb3f9d65159`, float32 `config.json` overlay `295154cd...`, `exclude_mtp=true`) rebuilt both variants from the pinned snapshot; every output equals the M0 digest:

| File | SHA-256 | Bytes |
| --- | --- | ---: |
| `ortgenai-fp32x/genai_config.json` | `e7e1957b2544fdad6170a9cc81cbbb1b88c8819b7016a609170ad7cbebba81ba` | 1,935 |
| `ortgenai-fp32x/model.onnx` | `4a052c4a1444c9437e00ffd48baecb0838d3d9ca1703c3153dbd46e945e84703` | 308,301 |
| `ortgenai-fp32x/model.onnx.data` | `129c58a320b6de7192713cc9c7e4ba744dcf7fabb0f57c6535021b45b3dc5143` | 3,077,177,344 |
| `ortgenai-fp16x/genai_config.json` | `ded164b67284883a13c6454d7445f814014a1e12277ce323011683cd6a69d055` | 2,109 |
| `ortgenai-fp16x/model.onnx` | `26895023e4be1f528aef04ce3a07a14e70a2421cfb400a49fec639ea9a14f46d` | 316,521 |
| `ortgenai-fp16x/model.onnx.data` | `47f6ee15d95080acc66f7c4acdf4012c83b20063cf0cfeb59a6b7ae09950a3b2` | 1,572,143,104 |
| `transformers-5.18.0/tokenizer.json` (builder copy) | `a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f` | 19,989,339 |
| `chat_template.jinja` (Hub) | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` | 7,755 |

The builder output is used unmodified; the pad id and provider settings are runtime overlays. `Qwen35ModelAssets` pins these files and downloads them from the hosting repository (section 7).

## 3. Fidelity against the transformers fp32 reference (22 prompts, greedy, 128 tokens)

"Exact": whole generated sequence (up to the stop token or 128 tokens) equal. Step-0: max abs log-probability difference over the full vocabulary and over the reference top 5.

| Variant, provider | Exact | Token match | Finish equal | Step-0 argmax | Step-0 max abs dlogp (all / top 5) | Divergences: step (reference margin) | Prefill chunk |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| fp32x, CPU | 22/22 | 1.000 | 22/22 | 22/22 | 1.38e-3 / 6.2e-4 | none | one call |
| fp32x graph, CUDA, `use_tf32=0` | 22/22 | 1.000 | 22/22 | 22/22 | 1.36e-3 / 6.0e-4 | none | 128 |
| fp16x, CUDA | 18/22 | 0.887 | 22/22 | 22/22 | 0.065 / 0.027 | `ru-de-mix` 38 (0.0073), `special-think-tool` 15 (0.0026), `special-broken` 5 (0.0142), `thinking-on` 76 (0.0004) | 512 |

Earlier runs on the same day agree: fp32x CPU with 512-token chunks 22/22 (same step-0 numbers to the last digit; chunked and one-call step-0 logits bit-identical on CPU), fp32x CPU one call 22/22, fp16x CUDA 18/22 with the same four divergences.

Exploratory throughput from the same runs (shared, loaded machine; not a benchmark): decode median over the prompts that generated at least 100 tokens 9.0 tokens/s (fp32x CPU, default threads), 16.9 (fp32x on CUDA), 52.1 (fp16x CUDA, GPU at 100% utilization from the other process; M0 measured 115 on a quieter GPU); prefill of the 1,666-token prompt 12.2 s, 1.5 s and 0.29 s.

- The owned template renders 21/21 text prompts exactly (`enable_thinking=true` included, which GenAI's template engine cannot express); the tools prompt is rejected as unsupported. The owned tokenizer gives the reference prompt ids on all 21 renderings, and on the reference rendering of the tools prompt (22/22, `TemplateTests`).
- The pad id overlay removes M0's outlier: with GenAI's default pad id 248044 the `special-endoftext` prompt (user-typed `<|endoftext|>`) was off by 2.46 at step 0; with the overlay every prompt is within 1.4e-3.
- fp16x diverges on the same 4 prompts as in M0, each at a reference top-1 minus top-2 margin below 0.015 (`ru-de-mix` step 38, 0.0073; `special-think-tool` 15, 0.0026; `special-broken` 5, 0.0142; `thinking-on` 76, 0.0004).
- Streaming: on every prompt and run, the updates concatenate to the result text and ids, and the text equals one decode of the completion ids and the reference text without the stop token. Usage: prompt tokens equal the prompt ids, completion tokens equal the returned ids (stop token excluded).

## 4. Semantics (plan 10.1 LLM list, G4 checklist)

All three runs (`semantics-*.json`) gave the same discrete results:

| Check | Result |
| --- | --- |
| Max-token boundary (`en-short`, reference 7 tokens then `<|im_end|>`) | `MaxNewTokens` 1, 6, 7: `Length` with exactly that many tokens, a prefix of the reference; 8: `Stop` with the 7 reference tokens and `StopTokenId` 248046. |
| Stop sequence (`" of France"`, two reference tokens) | `StopSequence`; text equals the reference text before the match; no streamed piece contains it; an absent stop string changes nothing. |
| Additional stop token id | `Stop` at the first occurrence, ids equal the reference before it. |
| EOS | All 10 prompts that end at `<|im_end|>` end with `Stop` and the same ids; the stop token is not in the text or the usage count. |
| Cached vs uncached (12 steps of `en-list`: the logits that selected each token against a fresh prefill of the same prefix) | tokens equal; max abs log-probability difference 3.4e-5 (fp32x CPU), 6.6e-5 (fp32x CUDA), 0.046 (fp16x CUDA). After `GenerateNextToken`, GenAI reports the logits that selected the token (it runs the model on a token at the next step). |
| Chunked vs one-call prefill (step-0 logits and 32 tokens) | CPU: bit-identical for chunks of 128, 100 and 7. fp16x CUDA: tokens equal, step-0 within 0.050. fp32x CUDA: the one-call prefill of the 1,666-token prompt did not fit in the free device memory (1.65 GB logits buffer); chunked runs equal the reference; `multi-turn` with chunks of 7 within 2.7e-5 of one call. |
| Cancellation | Token cancelled after 10 streamed updates: `OperationCanceledException` 3.1 to 3.3 ms later; an already cancelled token throws before the prefill; the next request on the same model equals the reference. Cancellation is not checked inside a step or a prefill chunk. |
| Streaming reassembly | 22/22 per run (section 3). |
| Request isolation | 4 requests on 2 threads and 2 interleaved streams equal the reference; the stress results are in section 5. |
| Seeded sampling (temperature 0.7, top-k 20, top-p 0.8) | same seed: identical tokens; another seed: different tokens; top-k 1: equals greedy. Same device and model only; not compared across devices (plan 10.1). |
| Text path (messages through the owned template and tokenizer) | `thinking-on` (`enable_thinking=true`), `multi-turn` and `special-endoftext` give the reference prompt ids and the reference's first 24 tokens. |
| Batch independence | not applicable: batching is not offered. |

## 5. Concurrent requests on CUDA: ONNX Runtime GenAI defect and mitigation

The first semantics run on CUDA showed one of four parallel requests differing from the reference. The stress mode repeats 8 rounds of 6 requests (32 tokens) on 1, 2 and 4 threads sharing one model and compares each request with the reference prefix:

| Configuration | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| CUDA fp32x, Starling loop, unserialized | 0/48 | 14/48 (15/48 in a first run) | 19/48 |
| CUDA fp32x, plain GenAI C# API (`rawstress`: `Config`, `Model`, `GeneratorParams`, `Generator`; no Starling code) | 0/48 | 10/48 | |
| CUDA fp16x, plain GenAI C# API | 0/48 | 11/48 | |
| CUDA fp32x, Starling loop, serialized (default) | 0/48 | 0/48 | 0/48 |
| CUDA fp16x, Starling loop, serialized (default) | 0/48 | 0/48 | 0/48 |
| CPU fp32x, Starling loop, unserialized (CPU default) | 0/48 | 0/48 | 0/48 |
| CPU fp32x, plain GenAI C# API | 0/48 | 0/48 | |

Divergences under concurrency happen at reference margins up to 6.2 logits and as early as step 0, so they are wrong results, not near-tie noise. The plain-API reproducer rules out the Starling decode loop, tokenizer and detokenizer (per-request state; the tokenizer is immutable). Mitigation: `GenAIModel` runs the native calls of all its sequences (generator creation, append, step, logits, disposal) under one lock by default on CUDA (`GenAIModelOptions.SerializeNativeCalls`, reported as `Qwen35ModelInfo.SerializesNativeCalls`); requests still interleave step by step, so throughput on CUDA does not scale with concurrent requests. CPU stays concurrent. Statuses: `concurrent_requests.cpu` experimental, `concurrent_requests.cuda.serialized` experimental, `concurrent_requests.cuda.unserialized` unsupported. Tests that would have caught it: `ConcurrentRequestsAreIndependent` now runs 4 rounds of 6 requests on 4 threads (CUDA test project) and 2 rounds on 4 threads (CPU). Minimal reproducer:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> dotnet run --project ai-dotnet/tools/Starlings.Qwen35.Fidelity -c Release -p:UseCuda=true -- \
  --bundle <bundle> --variant fp32x --device cuda --prompts model-pipeline/reference/generation/prompts-v1.jsonl \
  --reference <reference dir> --out <dir> --modes rawstress
```

Not yet reported upstream (microsoft/onnxruntime-genai); the cause inside GenAI or the ORT CUDA provider was not investigated. A report with a standalone reproducer (no Starling code) is drafted in `upstream-report-draft.md` for the maintainer to file (production-readiness M9).

## 6. Failures during the runs

1. Four `dotnet` processes aborted (SIGABRT, cores at 19:57, 20:01, 20:07 and 20:26 local time): all were runs of the fidelity runner on CUDA that died on an unhandled `GenAIModelException` wrapping an ONNX Runtime BFC arena allocation failure ("Failed to allocate memory for requested buffer of size 1654804480", 508559360 and 33554432 bytes). Causes: a one-call prefill of the 1,666-token prompt needs 1.65 GB of fp32 logits; 512-token chunks (508 MB) did not fit next to the fp32x weights in the about 5.5 GB the other GPU process left; the last one loaded a second copy of the weights in the same process (a runner bug, fixed: `rawstress` now runs alone). Device-memory contention, not a library defect; the library surfaces it as `GenAIModelException`. Response: `PrefillChunkTokens` (default 512 on CUDA), and 128-token chunks for the fp32x CUDA runs and tests on this machine.
2. The first `--no-model` release drill failed because `dotnet pack` of `Models.E5.MLNet` was killed during the GPU runs (memory pressure); the rerun on a quieter machine passed (section 8).

## 7. Packaging, platforms, hosting

- Libraries reference only `Microsoft.ML.OnnxRuntimeGenAI.Managed` 0.17.1 (it brings `Microsoft.Extensions.AI.Abstractions` 9.8.0); applications add `Microsoft.ML.OnnxRuntimeGenAI` (CPU) or `.Cuda`. CI fails a `src/` project that references a GenAI native package.
- Per RID (natives present in the packages): CPU linux-x64, linux-arm64, win-x64, win-arm64, osx-arm64; CUDA linux-x64, win-x64. Exercised: linux-x64 CPU and CUDA only; everything else untested.
- The GenAI managed assembly has no version; the adapter reads the package versions from the application's `.deps.json` and refuses others than 0.17.1. GenAI's native library embeds a tokenizer (ONNX Runtime Extensions) that no code path calls.
- Hosting (maintainer decision, 2026-10-10): GitHub release assets must be under 2 GiB and fp32x `model.onnx.data` is 3.08 GB, so the built files are hosted in the Hugging Face repository `Starlinghq/qwen3.5-0.8b-ortgenai` at commit `95a690c1774f144957cb7b76986a9126e8b1bb5f` (`Qwen35ModelAssets.HostingRepository` and `HostingRevision`, and `execution.graph.hosting` in the three `qwen3.5-0.8b` manifests), private for now. Correction 2026-10-13: this record first named the earlier location, `Huggingdws/qwen3.5-0.8b-ortgenai` at `bd8e2c33af181e50684f955e40dd1fdf4026719e` on the maintainer's account; the opt-in download test below and the `ai-0.1.0-preview.5` release drill log ran against that earlier location. Its layout equals the snapshot layout (`chat_template.jinja`, `transformers-5.18.0/tokenizer.json`, `ortgenai-fp32x/`, `ortgenai-fp16x/`, plus README, NOTICE, LICENSE, SHA256SUMS); the digests equal the lock (checked at upload). Every lock file, `chat_template.jinja` included, has a pinned `resolve` source there; downloading needs a read token with access to the repository. The acquisition library sends the token only to `https://huggingface.co` and not to the CDN host the Hub redirects large files to. Opt-in real test, 2026-10-10: the fp16x lock (1.59 GB) downloaded into an empty cache and verified in 55 s (`model.onnx.data` through a 302 to `us.aws.cdn.hf.co`); without a token the first file failed with `ModelAssetAccessDeniedException` (401). The fp32x lock was not downloaded (same code path). Capability `asset_acquisition.download`: experimental.

## 8. Release drill

`tools/release-drill-ai.sh --no-model --seed-qwen35-from <bundle>` at `2dd211e` passed (2026-10-10): every package packed from a clean clone with locked restore; nuspec dependencies exact (`Generation.OnnxGenAI`: `Starlings.Tokenizers [0.1.0-preview.3]`, `Microsoft.ML.OnnxRuntimeGenAI.Managed [0.17.1]`; `Models.Qwen35`: `Acquisition`, `Generation.OnnxGenAI`, `Tokenizers`, all `[0.1.0-preview.3]`), no model files in the packages, SourceLink to the commit; the QwenChat sample restored outside the repository from the local feed (GenAI CPU natives, no GPU natives, no foreign tokenizer, no project references) and built; the seeded fp32x files and runtime `tokenizer.json` verified from the isolated cache, `chat_template.jinja` downloaded from the Hub through the sample's `download` step; then offline (`unshare -rn`):

```
user: What is the capital of France? Answer in one sentence.
assistant: The capital of France is Paris.
[Stop; 24 prompt + 7 completion tokens; prefill 331 ms, 7.4 tokens/s]
```

The CI job runs the drill with `--no-model` (packing, inspection, QwenChat restore and build, no model).

## 9. Proposed tolerances (values only; the maintainer sets `tolerances.yaml`)

| Key | fp32x CPU | fp32x CUDA (`use_tf32=0`) | fp16x CUDA | Observed |
| --- | --- | --- | --- | --- |
| `discrete.template_rendering` | exact (21 text prompts) | same | same | 21/21 |
| `discrete.prompt_ids` | exact (22) | same | same | 22/22 |
| `discrete.greedy_ids` | exact; a divergence is allowed only at a reference margin below 1e-3, each listed | same | equal up to the first divergence, which must be at a reference margin below 0.05 | fp32x 22/22; fp16x 4 divergences, margins 0.0004 to 0.0142 |
| `discrete.finish_reason` | equal | equal | equal when the ids are equal | 22/22 |
| `float.step0_logprob_max_abs` (full vocabulary) | 5e-3 | 5e-3 | 0.15 | 1.38e-3 / 1.36e-3 / 0.065 (reference CPU vs CUDA: 1.6e-4) |
| `float.step0_top5_logprob_max_abs` | 2e-3 | 2e-3 | 0.06 | 6.2e-4 / 6.0e-4 / 0.027 |
| `discrete.step0_argmax` | equal (22) | equal | equal | 22/22 |
| `float.cached_vs_uncached_logprob_max_abs` | 2e-4 | 2e-4 | 0.1 | 3.4e-5 / 6.6e-5 / 0.046 |
| `float.chunked_vs_one_call_prefill_logprob_max_abs` | 1e-4 | (one call does not fit on the test GPU) | 0.1 | 0 / n.a. / 0.048 |
| `discrete.streaming_reassembly`, `discrete.usage` | exact | exact | exact | all runs |
| `discrete.concurrent_requests` | equal to the sequential result | equal (serialized) | equal (serialized) | section 5 |

## 10. Evaluator

Update (workstream R5, 2026-10-13): `model-pipeline evaluate` now has the `generation` task (`model-pipeline/src/model_pipeline/evaluate_generation.py`; `model-pipeline/README.md`, "Generation"). It reads the fidelity runner's output directory directly instead of a stage results envelope, and binds every key of the protected `tolerances.yaml`. The text below is the original gap description.

At the time of this evidence `model-pipeline evaluate` had no generation task. Needed extension (not built here): stage results format version 3 with `"task": "generation"`; stages `template` (per case `rendered`, `prompt_ids` against the fixture, exact; G1) and `dotnet` (per case `output_ids`, `finish_reason`, `step0_top` and optionally a step-0 vector sidecar, `stream_text_equals_result`, `usage`; plus `cached_vs_uncached` and `concurrency` sections; G2/G4), checked against `generation.jsonl` with the keys of section 9 (the near-tie rule reads `step_margin` from the fixture). The fidelity runner already writes everything except the envelope; a converter from its `results.jsonl` and `semantics.json` is about a day.

## 11. P-017 (Microsoft.Extensions.AI) mapping

The core is shaped so that the adapter is thin: `ChatMessage` roles map to `Qwen35Role` (tool messages, tools in `ChatOptions` and non-text contents throw `NotSupportedException`); `ChatOptions.MaxOutputTokens`, `StopSequences`, `Temperature`/`TopK`/`TopP`/`Seed` map to `Qwen35ChatOptions` (sampling only with an explicit seed); `GetStreamingResponseAsync` runs `GenerateStreaming` on a worker thread and maps each `GenerationUpdate` to a `ChatResponseUpdate` (text) and the final one to `FinishReason` (`Stop`/`StopSequence` to `Stop`, `Length` to `Length`) and `UsageContent` (`InputTokenCount` prompt, `OutputTokenCount` completion without the stop token); cancellation stays between steps. This avoids the two defects of GenAI's bundled `OnnxRuntimeGenAIChatClient` (a repeated last piece at EOS; the stop token counted as output).

## 12. Files

- `compare-*.json`: `compare.py` output per run. `semantics-*.json`, `stress-*-default.json` (default serialization: CUDA serialized, CPU concurrent), `stress-*-unserialized.json`, `rawstress-*.json`: runner outputs. `environment-*.json`: runner environment per run. `build-record.json`: the bundle build record.
