# Starlings.Models.Qwen3Embedding.Extensions.AI

**Experimental.** A Microsoft.Extensions.AI `IEmbeddingGenerator<string, Embedding<float>>` over `Qwen3TextEmbedder` from [Starlings.Models.Qwen3Embedding](https://github.com/starlinghq/starling/tree/main/ai-dotnet/src/Starlings.Models.Qwen3Embedding): same tokenizer, same graph, same numbers. Read that package's README for the profile, how to build the graphs (they are not hosted yet) and the capability statuses.

```csharp
using Microsoft.Extensions.AI;
using Starlings.Models.Qwen3Embedding.Extensions.AI;

using var generator = Qwen3EmbeddingGenerator.Create("/models/qwen3-embedding-0.6b-release");
var documents = await generator.GenerateAsync(["The capital of China is Beijing."]);
var queries = await generator.GenerateAsync(["What is the capital of China?"], new EmbeddingGenerationOptions
{
    AdditionalProperties = new() { [Qwen3EmbeddingGenerator.PromptPropertyName] = "query" },
});
```

- **Prompt.** `none` by default (documents); `query` adds the upstream query instruction. Set it per instance (`defaultPrompt`) or per request (`qwen3_prompt`).
- **Execution.** Inline on the calling thread; the returned task is already completed. Failures are in the task.
- **Options.** `Dimensions` must be null or 1024, `ModelId` null or `Qwen/Qwen3-Embedding-0.6B`, no `RawRepresentationFactory`, and no other `AdditionalProperties` key; anything else fails with `NotSupportedException`.
- **Usage.** Input token count, prompt and the appended `<|endoftext|>` included, after truncation to 8192.

Micro-batching, streaming, a scheduler and DI registration are not offered (unsupported).
