# Starlings.Models.Qwen35

**Preview.** Chat generation with [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) at revision `2fc06364715b967f1860aea9cf38778875588b17`, in-process on ONNX Runtime GenAI 0.17.1, offline after an explicit acquisition. The API and the supported platforms may still change. The registry profiles are Experimental; see [Path to Stable](#path-to-stable).

- The pinned chat template is rendered in C# for text conversations (system, user, assistant, `enable_thinking`); tools, tool messages and images are rejected. Prompt ids come from the owned tokenizer `Starlings.Tokenizers` (profile `qwen35-bpe-chat`): no Microsoft.ML.Tokenizers, no GenAI tokenizer, no Python.
- Starling's decode loop (`Starlings.Generation.OnnxGenAI`): greedy or seeded sampling, stop at `<|im_end|>` / `<|endoftext|>` and at stop strings, streaming, cancellation between decode steps, usage.
- The package ships the asset locks (sizes and SHA-256), not the model files (fp32x 3.1 GB, fp16x 1.6 GB). Acquisition is an explicit call; generation never downloads.

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

For Microsoft.Extensions.AI's `IChatClient` (middleware, DI, asynchronous streaming), add `Starlings.Models.Qwen35.Extensions.AI` and wrap the model in `Qwen35ChatClient`.

## Profiles

| Profile | Variant, provider | Status |
| --- | --- | --- |
| `qwen3.5-0.8b/ortgenai-fp32x` | fp32 graph on CPU; the same graph on CUDA with TF32 off | experimental |
| `qwen3.5-0.8b/ortgenai-fp16x` | fp16 graph on CUDA | experimental (near-tie divergences from the fp32 reference) |

## Semantics

- `MaxNewTokens` bounds the completion tokens; a stop token id is never part of the text, the ids or the usage count. A stop string ends generation and is cut from the text; its tokens stay in the ids and the count.
- Streaming updates concatenate to the final text and ids; the last update carries the finish reason (`Stop`, `StopSequence`, `Length`) and usage. Cancellation throws `OperationCanceledException` at the next prefill chunk or decode step; a running chunk or step is not interrupted. Prompts are prefilled in 512-token chunks on CUDA and in one call on CPU (`PrefillChunkTokens`).
- Each request renders and prefills its whole conversation; requests share no state. On CUDA concurrent requests are serialized step by step, because concurrent ONNX Runtime GenAI 0.17.1 generators on the CUDA provider produced wrong tokens; on CPU they run in parallel. Conversation state reuse, batching, tool calling and reasoning extraction are unsupported in this version.

## Stable contract

This is the contract a Stable profile would commit to. Today every profile is **Experimental**. The statuses below are the current ones, from the registry manifests and `Qwen35Capabilities.All`. Support levels are defined in the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md).

| Item | Contract |
| --- | --- |
| Model | `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`, built with the onnxruntime-genai 0.17.1 builder (source `83de55a1f3886cddbb32a5517e26adb3f9d65159`, float32 `config.json` overlay, `exclude_mtp=true`). Files are verified by SHA-256 (`Qwen35ModelAssets`). |
| Tokenizer and template | `Starlings.Tokenizers` `0.3.0-preview.1`, profile `qwen35-bpe-chat`. The chat template is rendered by `Qwen35ChatTemplate` (text subset; experimental until P-029). GenAI's tokenizer and template engine are never loaded. |
| Runtime | ONNX Runtime GenAI 0.17.1. `ortgenai-fp32x`: CPU, and CUDA with TF32 off. `ortgenai-fp16x`: CUDA. |
| Input | A text conversation (system, user, assistant messages) and `Qwen35ChatOptions` (max new tokens, default 512; greedy or seeded sampling; stop strings; `enable_thinking`, default false). Tools, tool messages and images are rejected. |
| Limits | Prompt plus `MaxNewTokens` up to the model's context length (262,144). Above that, `GenerationContextLengthException`, and nothing is truncated. Tested only up to 1,666 prompt tokens. |
| Output | Completion text and ids, finish reason (`Stop`, `StopSequence`, `Length`), and usage (prompt tokens; completion tokens excluding a stop token id). Streaming updates concatenate to the final result. Semantics as in [Semantics](#semantics). |
| Concurrency | CPU: concurrent requests run in parallel. CUDA: requests are serialized step by step, because concurrent GenAI 0.17.1 generators produced wrong tokens (upstream report pending, M9). |
| Numerical bound | The protected tolerances of each profile: the template stage exact, greedy ids exact except at recorded near-tie margins, and step-0 log-probabilities within the bound (fp32x CPU: 5e-3). |
| Supported | None yet. |
| Experimental | greedy generation (fp32x CPU and CUDA, fp16x CUDA), owned template (text) and `enable_thinking`, owned tokenizer, stop tokens and stop sequences, max new tokens, streaming, cancellation between prefill chunks and decode steps, chunked prefill, usage, concurrent requests (CPU parallel, CUDA serialized), seeded sampling, step-0 logits, asset acquisition, `Starlings.Models.Qwen35.Extensions.AI` (`IChatClient`) on fp32x. Linux x64. |
| Untested | prompts over 1,666 tokens, the Extensions.AI adapter on fp16x, Native AOT, Windows, Linux ARM64, macOS. |
| Unsupported | cancellation inside a decode step or prefill chunk, batching several prompts, unserialized concurrent requests on CUDA, conversation state reuse, tool calling, reasoning extraction, structured output, images and video, beam search and repetition penalties, MTP and speculative decoding, quantized variants, TF32 on CUDA, persistence, training, ML.NET. |

## Path to Stable

The order of promotion (`docs/plan/production-readiness.md`) takes `ortgenai-fp32x-cpu` to Preview first. `ortgenai-fp16x-cuda` likely stays Preview, because of near-tie divergences (18 of 22 greedy cases exact). Gate evidence is in the [catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md) once draft PR #4 lands. Agents produce evidence; only the maintainer records a gate or changes maturity.

- [ ] G0 provenance: today a checklist ([P-016 evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/p016-qwen35/README.md)). The built files are publicly hosted since 2026-10-14 (M3 done); recording the gate result stays with the maintainer.
- [ ] G1 and G2: today checklists. On development fixtures: renderings and prompt ids 21/21, greedy 22/22, and step-0 1.38e-3 against a 5e-3 bound, rerun at tokenizer `0.3.0-preview.1` (draft PR #6). The generation evaluator task that machine-checks these against the protected tolerances is in draft PR #5. Still needed: the protected P-030 generation set (M1), and the GPU runs for the CUDA profiles (M6).
- [ ] P-029 template corpus beyond the 21 prompts, so the owned renderer can leave experimental (R8).
- [ ] G3 task quality: needs a fixed generation evaluation, a dataset and a budget (M1, M2).
- [ ] G4 resource correctness: today a checklist for the LLM semantics and the CUDA concurrency defect (draft PR #6). Needs a one-hour soak and the plan 10.2 tests (R1), long-context tests above 1,666 tokens or a narrowed claim (R8), and the filed upstream report with CUDA serialization confirmed as the documented behavior (M9).
- [ ] G5 performance: the harness is in draft PR #7 (an uncontrolled run, not G5 evidence). Needs a controlled run against a budget (M2, M6).
- [ ] G6 distribution: the release drill on seeded files is a checklist (draft PR #6). Needs acquisition from public hosting (R9 after M3), and Windows x64 (R6, R9).
- [ ] G7 integration: the `IChatClient` adapter is tested. Needs a representative downstream use (M7).
- [ ] `aot`, `long_context_over_1666_tokens` and fp16x `extensions_ai`: tested, or recorded as `unsupported`. The validator refuses Stable with any `untested` capability.
- [ ] Qualification CI (P-018, R6), the compatibility matrix rows, the support policy, and a second release owner (M7).

## Attribution and license

Qwen3.5-0.8B by the Qwen team, Alibaba Cloud (Copyright 2026 Alibaba Cloud), Apache-2.0 (`LICENSE` in the Hub repository at the pinned commit). The built ONNX files are derived from those weights under the same license. `Qwen35ChatTemplate` is a C# port of the checkpoint's `chat_template.jinja` (Apache-2.0). This package's code is MIT OR Apache-2.0.

Documentation and evidence: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#generation-backend-and-pilot-llm-p-016).
