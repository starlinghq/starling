# Starlings.Models.Qwen35

**Preview.** Chat generation with [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) at revision `2fc06364715b967f1860aea9cf38778875588b17`, in-process on ONNX Runtime GenAI 0.17.1, offline after an explicit acquisition. The API and the supported platforms may still change.

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

The built graphs are not on the Hub: Starling builds them reproducibly from the pinned checkpoint (`model-pipeline/reference/generation/build_ortgenai_bundle.py`, digests in `Qwen35ModelAssets`) and hosts every file of the locks (with `chat_template.jinja`) in the Hugging Face repository `Starlinghq/qwen3.5-0.8b-ortgenai`, pinned by commit. That repository is private for now (it moves to an organization later): `Qwen35ModelAssets.DownloadAsync(variant)` needs a Hugging Face read token with access to it (`ModelAcquisitionOptions.HuggingFaceToken` or `$HF_TOKEN`) and otherwise fails with `ModelAssetAccessDeniedException`. Alternatively build the files into the snapshot directory or call `Qwen35ModelAssets.OpenDirectory(dir, variant)`.

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

## Attribution and license

Qwen3.5-0.8B by the Qwen team, Alibaba Cloud (Copyright 2026 Alibaba Cloud), Apache-2.0 (`LICENSE` in the Hub repository at the pinned commit). The built ONNX files are derived from those weights under the same license. `Qwen35ChatTemplate` is a C# port of the checkpoint's `chat_template.jinja` (Apache-2.0). This package's code is MIT OR Apache-2.0.

Documentation and evidence: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#generation-backend-and-pilot-llm-p-016).
