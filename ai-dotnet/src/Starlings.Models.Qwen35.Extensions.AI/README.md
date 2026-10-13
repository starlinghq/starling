# Starlings.Models.Qwen35.Extensions.AI

**Preview.** [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) `IChatClient` over the `Qwen/Qwen3.5-0.8B` chat model in `Starlings.Models.Qwen35`: the same template, tokenizer, decode loop and tokens, usable with Microsoft.Extensions.AI middleware and dependency injection. The API may still change.

## Install

```sh
dotnet add package Starlings.Models.Qwen35.Extensions.AI --prerelease
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.17.1        # CPU natives; or Microsoft.ML.OnnxRuntimeGenAI.Cuda (exactly one)
```

## Use

```csharp
using Starlings.Models.Qwen35;
using Starlings.Models.Qwen35.Extensions.AI;
using Microsoft.Extensions.AI;

Qwen35ModelFiles files = Qwen35ModelAssets.OpenLocal(Qwen35Variant.Fp32x);   // offline, verified, never downloads
using IChatClient client = Qwen35ChatClient.Create(files);

ChatResponse response = await client.GetResponseAsync("What is the capital of France?", new ChatOptions { MaxOutputTokens = 128 });
Console.WriteLine($"{response.Text} [{response.FinishReason}, {response.Usage?.InputTokenCount} + {response.Usage?.OutputTokenCount} tokens]");

await foreach (ChatResponseUpdate update in client.GetStreamingResponseAsync("Why is the sky blue?"))
{
    Console.Write(update.Text);
}

// Or with DI: services.AddQwen35ChatClient(files);
```

- Roles system, user and assistant with text content. Tools, tool messages, function calls, images and other content are rejected with `NotSupportedException`.
- `MaxOutputTokens`, `StopSequences`, `Temperature`/`TopP`/`TopK` with an explicit `Seed` (greedy without them; sampling without a seed is rejected), and `AdditionalProperties["enable_thinking"]`. Every other option that would change the result is rejected instead of being ignored.
- Streaming updates are complete-character text deltas; they concatenate to the non-streamed text, and the last update carries the finish reason (`Stop` or `Length`) and usage (stop token excluded).
- Each request runs on a dedicated background thread (ONNX Runtime GenAI steps are blocking native calls). Cancellation takes effect between prefill chunks and decode steps.

Status: Linux x64 CPU and CUDA experimental. See the [documentation and evidence](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#microsoftextensionsai-chat-client-p-017).
