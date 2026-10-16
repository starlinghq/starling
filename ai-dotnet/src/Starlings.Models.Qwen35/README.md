# Starlings.Models.Qwen35

Chat generation with [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) at revision `2fc06364715b967f1860aea9cf38778875588b17`, in-process on ONNX Runtime GenAI 0.17.1, offline after an explicit acquisition. The API and the supported platforms may still change. The registry profiles are Experimental; see [Path to Stable](#path-to-stable).

- The pinned chat template is rendered in C# for text and tool-calling conversations (system, user, assistant and tool messages, tool definitions, assistant tool calls, `enable_thinking`); images are rejected. Prompt ids come from the owned tokenizer `Starlings.Tokenizers` (profile `qwen35-bpe-chat`): no Microsoft.ML.Tokenizers, no GenAI tokenizer, no Python.
- Starling's decode loop (`Starlings.Generation.OnnxGenAI`): greedy or seeded sampling, stop at `<|im_end|>` / `<|endoftext|>` and at stop strings, streaming, cancellation between decode steps, usage.
- The package ships the asset locks (sizes and SHA-256), not the model files (fp32x 3.1 GB, fp16x 1.6 GB). Acquisition is an explicit call; generation never downloads.

## Maturity

<!-- catalog:maturity profiles=qwen3.5-0.8b/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Install

```sh
dotnet add package Starlings.Models.Qwen35 --prerelease
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.17.1        # CPU natives
# or, for CUDA: Microsoft.ML.OnnxRuntimeGenAI.Cuda 0.17.1 (reference exactly one)
```

## Use

```csharp
using Starlings.Models.Qwen35;

Qwen35ModelFiles files = Qwen35ModelAssets.OpenLocal(Qwen35Variant.Fp32x);   // offline, verified, never downloads
using var model = Qwen35ChatModel.Load(files);

var messages = new[] { Qwen35Message.User("What is the capital of France?") };
foreach (var update in model.GenerateStreaming(messages, new Qwen35ChatOptions { MaxNewTokens = 128 }))
{
    Console.Write(update.Text);
    if (update.Result is { } done)
    {
        Console.WriteLine($"\n[{done.FinishReason}, {done.Usage.PromptTokens} + {done.Usage.CompletionTokens} tokens]");
    }
}
```

The built graphs are not on the Hub: Starling builds them reproducibly from the pinned checkpoint (`model-pipeline/reference/generation/build_ortgenai_bundle.py`, digests in `Qwen35ModelAssets`) and hosts every file of the locks (with `chat_template.jinja`) in the Hugging Face repository `Starlinghq/qwen3.5-0.8b-ortgenai`, pinned by commit. That repository is public: `Qwen35ModelAssets.DownloadAsync(variant)` needs no token. Alternatively build the files into the snapshot directory or call `Qwen35ModelAssets.OpenDirectory(dir, variant)`.

For Microsoft.Extensions.AI's `IChatClient` (middleware, DI, asynchronous streaming, function invocation), add `Starlings.Models.Qwen35.Extensions.AI` and wrap the model in `Qwen35ChatClient`.

### Tool calling (experimental)

```csharp
using System.Text.Json;

var weather = Qwen35Tool.Function("get_weather", "Get the current weather for a city.", JsonDocument.Parse(
    """{"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}""").RootElement);
var options = new Qwen35ChatOptions { Tools = [weather] };
var conversation = new List<Qwen35Message> { Qwen35Message.User("What is the weather in Paris today?") };

var result = model.Generate(conversation, options);
var parsed = Qwen35ToolCallParser.Parse(result.Text, options);       // text and Qwen35ToolCall parts
foreach (var call in parsed.ToolCalls)
{
    Console.WriteLine($"{call.Name}({call.Arguments})");                // get_weather({"city":"Paris"})
}

conversation.Add(parsed.ToAssistantMessage());
conversation.Add(Qwen35Message.Tool("18 C and sunny"));               // one tool message per result, in call order
result = model.Generate(conversation, options);                        // the answer
```

- `Qwen35ChatOptions.Tools` are rendered into the template's tools block as Python `json.dumps` does (key order kept). `Qwen35Message.ToolCalls` (assistant) and `Qwen35Role.Tool` messages are rendered as the template does: `<tool_call>` blocks after the content, and consecutive tool results in one user turn of `<tool_response>` blocks.
- The model answers in the template's format (`<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n</function>\n</tool_call>`). `Qwen35ToolCallParser` splits a completion, or streamed deltas, into text and calls. Values are cast with the tool's JSON schema as transformers' response parser does (integer, number, boolean, null, object, array; otherwise a string). A malformed or unfinished block is returned as text, verbatim; the parser never throws on model output. With thinking enabled, nothing before `</think>` is parsed.
- Streaming (`Append`, `Complete`) returns text as soon as it cannot start a tool call, and each call when its block closes; the parts equal those of parsing the whole completion.
- Nothing constrains decoding: the model may answer in text, call an undeclared tool, or omit required arguments. Check calls before running them.

## Profiles

| Profile | Variant, provider | Status |
| --- | --- | --- |
| `qwen3.5-0.8b/ortgenai-fp32x` | fp32 graph on CPU (`ortgenai-fp32x-cpu`); the same graph on CUDA with TF32 off (`ortgenai-fp32x-cuda`) | CPU: stable (since 2026-10-16); CUDA: stable on an RTX 3090 (since 2026-10-16) |
| `qwen3.5-0.8b/ortgenai-fp16x` | fp16 graph on CUDA (`ortgenai-fp16x-cuda`) | stable on an RTX 3090 (since 2026-10-16), under its protected near-tie rule (greedy ids may diverge from the fp32 reference where the reference margin is below 0.05) |

## Semantics

- `MaxNewTokens` bounds the completion tokens; a stop token id is never part of the text, the ids or the usage count. A stop string ends generation and is cut from the text; its tokens stay in the ids and the count.
- Streaming updates concatenate to the final text and ids; the last update carries the finish reason (`Stop`, `StopSequence`, `Length`) and usage. Cancellation throws `OperationCanceledException` at the next prefill chunk or decode step; a running chunk or step is not interrupted. Prompts are prefilled in 512-token chunks on CUDA and in one call on CPU (`PrefillChunkTokens`).
- Each request renders and prefills its whole conversation; requests share no state. On CUDA concurrent requests are serialized step by step, because concurrent ONNX Runtime GenAI 0.17.1 generators on the CUDA provider produced wrong tokens; on CPU they run in parallel. Conversation state reuse, batching, forced tool calls (requiring a tool) and reasoning extraction are unsupported in this version.

## Stable contract

This is the contract of profile `qwen3.5-0.8b/ortgenai-fp32x-cpu` (fp32x on Linux x64 CPU). Its Stable label (since 2026-10-16) commits to it under the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md). The statuses are the registry manifest's and `Qwen35Capabilities.Profiles`, which a test keeps equal key for key. `Qwen35Capabilities.All` also lists the experimental CUDA profiles' keys (`.cuda`, `fp16x`).

| Item | Contract |
| --- | --- |
| Profile | `qwen3.5-0.8b/ortgenai-fp32x-cpu` (registry manifest `model-pipeline/registry/qwen3.5-0.8b/ortgenai-fp32x-cpu/manifest.yaml`). |
| Model | `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`, built with the onnxruntime-genai 0.17.1 builder (source `83de55a1f3886cddbb32a5517e26adb3f9d65159`, float32 `config.json` overlay, `exclude_mtp=true`). The files are downloaded with no token from the public hosting repository and verified by SHA-256 (`Qwen35ModelAssets`). |
| Tokenizer and template | `Starlings.Tokenizers` `0.3.0-preview.1`, profile `qwen35-bpe-chat`. The chat template is rendered by `Qwen35ChatTemplate`. GenAI's tokenizer and template engine are never loaded. |
| Runtime | ONNX Runtime GenAI 0.17.1 with its CPU natives (`Microsoft.ML.OnnxRuntimeGenAI`), fp32, .NET 10. |
| Supported platform | Linux x64, CPU. G1 to G4 ran on an Intel Core i7-5820K (AVX2, no AVX-512), .NET 10.0.12. G5 ran on a quiet host with both sides pinned to the same cores. The weekly qualification reruns G1 to G3 on the release sets and a one-hour soak on GitHub-hosted `ubuntu-latest` x64 (4 logical CPUs; transparent huge pages `madvise`). |
| Memory on Linux | No `MALLOC_*` environment setting is required. On hosts with transparent huge pages set to `always`, resident memory can look larger and rise over time with no leak behind it (the qualification soak sets `madvise`, the Ubuntu default). |
| Input | A text conversation (system, user, assistant and tool messages; assistant tool calls) and `Qwen35ChatOptions` (max new tokens, default 512; greedy or seeded sampling; stop strings; `enable_thinking`, default false; tool definitions). Images are rejected. |
| Limits | **Prompt plus `MaxNewTokens` up to 8,192 tokens** (`Qwen35ChatModel.MaxContextTokens`; the model's own context length is 262,144). Above that, `GenerationContextLengthException` before any native call, and nothing is truncated. Prompts up to 1,666 tokens are inside the contract (the accepted fixtures). Prompts of 1,667 to 8,192 tokens run but are outside it (below). At 16,391 tokens, cached vs uncached decoding exceeded the protected bound, which is why the limit is 8,192. |
| Output | Completion text and ids, finish reason (`Stop`, `StopSequence`, `Length`), and usage (prompt tokens; completion tokens excluding a stop token id). Streaming updates concatenate to the final result. Semantics as in [Semantics](#semantics). |
| Concurrency | Concurrent requests run in parallel and are independent. |
| Numerical bound | The protected tolerances of the profile: template rendering and prompt ids exact; greedy ids exact except at recorded near-tie margins below 1e-3; step-0 top-5 log-probabilities within 2e-3 and full-vocabulary step-0 log-probabilities within 5e-3; cached vs uncached within 2e-4; chunked vs one-call prefill within 1e-4. Release prompts: greedy ids 32/32 exact, full vocabulary 9.3e-4, cached vs uncached 2.7e-5. The full step-0 reference vectors regenerate byte-identical only in the recorded reference environment. The accepted vectors are hosted at `Starlinghq/starling-release-references` (commit and SHA-256 in the protected `hosted-reference.json`), and the evaluator checks each against the fixture's digest. |
| Task quality | GSM8K, 50 problems, greedy: exact match at least 0.60 (budget); measured 0.64, equal to the transformers fp32 CPU reference row for row. |
| Performance | G5: decode tokens per second at least 0.95 of the Python ONNX Runtime GenAI run on the same bundle, and time to first token at most 1.10 of it. The qualifying run measured 1.045 [0.823, 1.274] on `short-decode64`. The first run failed (1.119) under host load (see the qualification evidence). |
| Supported | Greedy generation; the owned template for text conversations and the owned tokenizer; stop tokens and stop sequences; max new tokens; streaming; cancellation between decode steps; chunked prefill; usage; concurrent requests; seeded sampling; step-0 logits; explicit acquisition and anonymous download; Native AOT; `Starlings.Models.Qwen35.Extensions.AI` (`IChatClient`: chat, streaming, cancellation, usage, seeded sampling, concurrency, middleware, dependency injection). |
| Outside the Stable contract | Experimental: **tool calling** (the template's tools block, tool calls and tool messages match upstream on the 41 tool cases of the proposed P-029 corpus; the parser, the streaming parser and the `IChatClient` function-invocation loop are tested with scripted sequences and once on the real model; a 0.8B model's tool choice is not qualified). Also experimental: `enable_thinking` (one accepted prompt; the P-029 rows are proposed) and prompts of 1,667 to 8,192 tokens (a proposed fixture, within the protected bounds). CUDA (`ortgenai-fp32x-cuda`, `ortgenai-fp16x-cuda`): their own contract ([Stable contract: CUDA](#stable-contract-cuda)). Windows x64, Linux ARM64 and macOS: untested. |
| Unsupported | sequences over 8,192 tokens, cancellation inside a decode step or prefill chunk, batching several prompts, conversation state reuse, required or specific tool choice (`ToolMode` `RequireAny`/`RequireSpecific`), reasoning extraction, structured output, images and video, beam search and repetition penalties, MTP and speculative decoding, quantized variants, fp16 on CPU, persistence, training, ML.NET. |
| Known issues | The ONNX Runtime GenAI C# API is labeled preview upstream; `Starlings.Generation.OnnxGenAI` isolates it. The runtime version must equal the builder version that wrote the graph (0.17.1). C# and Python load different ONNX Runtime 1.30.0 native builds (NuGet against pip); G5 compares the shipped products. |

## Stable contract: CUDA

This is the contract of profiles `qwen3.5-0.8b/ortgenai-fp32x-cuda` and `qwen3.5-0.8b/ortgenai-fp16x-cuda`, Stable since 2026-10-16. It is the CPU contract above, with these differences. The statuses are the registry manifests' and `Qwen35Capabilities.Profiles`, which a test keeps equal key for key.

| Item | Contract |
| --- | --- |
| Profiles | `qwen3.5-0.8b/ortgenai-fp32x-cuda` (the fp32x graph of the CPU contract, on CUDA with TF32 off) and `qwen3.5-0.8b/ortgenai-fp16x-cuda` (the fp16 graph, `Qwen35Variant.Fp16x`, built by the same builder; digests in `Qwen35ModelAssets`). |
| Runtime | ONNX Runtime GenAI 0.17.1 with its CUDA natives (`Microsoft.ML.OnnxRuntimeGenAI.Cuda`, which brings `Microsoft.ML.OnnxRuntime.Gpu` 1.30.0). The CUDA 13 runtime and cuBLAS must be on the loader path; the package does not ship them. |
| Supported platform | Linux x64 with an **NVIDIA GeForce RTX 3090** (Ampere), driver 580 or later. G1 to G3 on the release sets, the binding G4 one-hour soaks and G5 ran on RTX 3090s. **Regression:** the per-release GPU run (the qualification workflow's `cuda (runpod)` job, for every `ai-v*` tag and on demand) reruns the CUDA tests (the typed model and `IChatClient`, both variants), G1 to G3 on the release sets and a soak smoke test, on an RTX 3090 only; the job refuses any other GPU. Other GPUs, including the RTX 4090 (Ada), are untested and outside the contract. |
| Memory on Linux | No `MALLOC_*` setting is required. |
| Concurrency | Concurrent requests are serialized step by step on CUDA (concurrent ONNX Runtime GenAI 0.17.1 generators on the CUDA provider produced wrong tokens). |
| Cancellation | Checked between decode steps and between prefill chunks: prompts are prefilled in 512-token chunks on CUDA by default (`PrefillChunkTokens`). A running chunk or step is not interrupted. |
| Numerical bound | Each profile's protected tolerances. fp32x CUDA: as fp32x CPU (greedy ids exact except at recorded near-tie margins below 1e-3; full-vocabulary step 0 within 5e-3; cached vs uncached within 2e-4); release prompts: greedy ids 32/32 exact, full vocabulary 9.3e-4, cached vs uncached 4.6e-5. fp16x CUDA: greedy ids equal until a first divergence at a reference margin below 0.05; full-vocabulary step 0 within 0.15, cached vs uncached within 0.1; release prompts: 28/32 exact, the other 4 diverging only at allowed near ties, full vocabulary 0.046. |
| Task quality | GSM8K, 50 problems, greedy: fp32x CUDA 0.64 (budget 0.60); fp16x CUDA 0.60 to 0.64 (budget 0.56). |
| Performance | G5 on an RTX 3090: fp32x decode 0.975 to 1.023 of the Python ONNX Runtime GenAI run (minimum 0.95), time to first token 0.981 to 1.012 (budget 1.10); fp16x decode 1.042 to 1.058, time to first token 0.922 to 0.997. |
| Supported | As in the CPU contract, on CUDA: greedy generation; the owned template (text) and tokenizer; stop tokens and sequences; max new tokens; streaming; cancellation; chunked prefill; usage; serialized concurrent requests; seeded sampling; step-0 logits; explicit acquisition and anonymous download; `Starlings.Models.Qwen35.Extensions.AI` (`IChatClient`). |
| Outside the Stable contract | Experimental: Native AOT with the CUDA natives (one local run, `tools/aot-check-ai.sh --cuda`, not in the per-release run); tool calling, `enable_thinking` and prompts of 1,667 to 8,192 tokens, as on CPU. Windows x64 CUDA: untested. |

Evidence: [binding CUDA G4 and G5 round](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-16-cuda/README.md), [per-release GPU regression, run 38857860186](https://github.com/starlinghq/starling/blob/main/docs/evidence/gpu-regression-2026-10-16/README.md), release-set gates for [fp32x CUDA](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/qwen35-ortgenai-fp32x-cuda/README.md) and [fp16x CUDA](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/qwen35-ortgenai-fp16x-cuda/README.md).

## Path to Stable

**Status 2026-10-16: `ortgenai-fp32x-cpu` is Stable** ([contract](#stable-contract)), and so are `ortgenai-fp32x-cuda` and `ortgenai-fp16x-cuda` on an RTX 3090 ([contract](#stable-contract-cuda)): every gate is recorded passed, and every check of both passed in the first per-release GPU run (38857860186). Every gate G0 to G7 is recorded passed. G5 passed on the planned quiet-host rerun ([evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-15/README.md#qwen-cpu-time-to-first-token-diagnosis-on-a-quiet-host)) after a first run failed under host load. The weekly qualification workflow reruns its release-set G1 to G3 and a one-hour soak, green in [run 38848321624](https://github.com/starlinghq/starling/actions/runs/38848321624). G2 there reads the full step-0 reference vectors from `Starlinghq/starling-release-references`, which the protected `hosted-reference.json` pins. GPU regression runs for every release (an `ai-v*` tag) and on demand, on an RTX 3090 only, in the qualification workflow's `cuda (runpod)` job; CPU regression runs weekly. The checklist below predates the release-set gates and is kept as history.

The order of promotion (`docs/plan/production-readiness.md`) takes `ortgenai-fp32x-cpu` to Preview first. The fp16x profile's near-tie divergences (18 of 22 development cases, 28 of 32 release prompts exact) are within its protected near-tie rule, which is its contract. Gate evidence is in the [catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md) once draft PR #4 lands. Agents produce evidence; only the maintainer records a gate or changes maturity.

- [ ] G0 provenance: today a checklist ([P-016 evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/p016-qwen35/README.md)). The built files are publicly hosted since 2026-10-14 (M3 done); recording the gate result stays with the maintainer.
- [ ] G1 and G2: today checklists. On development fixtures: renderings and prompt ids 21/21, greedy 22/22, and step-0 1.38e-3 against a 5e-3 bound, rerun at tokenizer `0.3.0-preview.1` (draft PR #6). The generation evaluator task that machine-checks these against the protected tolerances is in draft PR #5. Still needed: the protected P-030 generation set (M1), and the GPU runs for the CUDA profiles (M6).
- [ ] P-029 template corpus beyond the 21 prompts, so the owned renderer can leave experimental (R8). The proposed corpus (121 cases, 41 of them tool calling) awaits the maintainer.
- [ ] Tool calling: a tool-use evaluation set and budget, so the model's tool choice is measured rather than shown on one prompt.
- [ ] G3 task quality: needs a fixed generation evaluation, a dataset and a budget (M1, M2).
- [ ] G4 resource correctness: today a checklist for the LLM semantics and the CUDA concurrency defect (draft PR #6). Needs a one-hour soak and the plan 10.2 tests (R1), long-context tests above 1,666 tokens (done: 1,667 to 8,192 experimental, longer rejected), and the filed upstream report with CUDA serialization confirmed as the documented behavior (M9).
- [ ] G5 performance: the harness is in draft PR #7 (an uncontrolled run, not G5 evidence). Needs a controlled run against a budget (M2, M6).
- [ ] G6 distribution: the release drill on seeded files is a checklist (draft PR #6). Needs acquisition from public hosting (R9 after M3), and Windows x64 (R6, R9).
- [ ] G7 integration: the `IChatClient` adapter is tested. Needs a representative downstream use (M7).
- [x] `aot`, long context and fp16x `extensions_ai`: `aot` and fp16x `extensions_ai` experimental; long context split into `long_context_1667_to_8192_tokens` (experimental) and `long_context_over_8192_tokens` (unsupported, rejected) ([capability closure](https://github.com/starlinghq/starling/blob/main/docs/evidence/capability-closure-2026-10-14/README.md)). The validator refuses Stable with any `untested` capability.
- [ ] Qualification CI (P-018, R6), the compatibility matrix rows, the support policy, and a second release owner (M7).

## Attribution and license

Qwen3.5-0.8B by the Qwen team, Alibaba Cloud (Copyright 2026 Alibaba Cloud), Apache-2.0 (`LICENSE` in the Hub repository at the pinned commit). The built ONNX files are derived from those weights under the same license. `Qwen35ChatTemplate` is a C# port of the checkpoint's `chat_template.jinja` (Apache-2.0). This package's code is MIT OR Apache-2.0.

Documentation and evidence: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#generation-backend-and-pilot-llm-p-016).
