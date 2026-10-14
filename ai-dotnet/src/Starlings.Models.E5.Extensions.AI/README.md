# Starlings.Models.E5.Extensions.AI

**Preview.** [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) `IEmbeddingGenerator<string, Embedding<float>>` over the multilingual-e5-small embedder in `Starlings.Models.E5`: the same tokenizer, encoder and numbers, usable with Microsoft.Extensions.AI middleware and dependency injection. The API may still change.

## Install

```sh
dotnet add package Starlings.Models.E5.Extensions.AI --prerelease
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # or Microsoft.ML.OnnxRuntime.Gpu.Linux; exactly one
```

## Use

```csharp
using Starlings.Models.E5;
using Starlings.Models.E5.Extensions.AI;
using Microsoft.Extensions.AI;

await E5ModelAssets.DownloadAsync();                 // explicit acquisition, once
E5ModelFiles files = E5ModelAssets.OpenLocal();      // offline

using IEmbeddingGenerator<string, Embedding<float>> generator = E5EmbeddingGenerator.Create(files,
    new E5EmbeddingGeneratorOptions { InputType = E5InputType.Query });   // adds "query: "; the default adds nothing
var embeddings = await generator.GenerateAsync(["how much protein should a female eat"]);

// Or with DI: services.AddE5EmbeddingGenerator(files);
```

- Prefixes are explicit: `E5InputType.Query` or `Passage` per generator, or per request with `AdditionalProperties["e5_input_type"]`.
- `GenerateAsync` runs synchronously on the calling thread and returns a completed task.
- Unsupported options (`Dimensions` other than 384, another `ModelId`, unknown additional properties) fail with `NotSupportedException` instead of being ignored.

Status: Linux x64 CPU experimental; over a CUDA session untested. See the [release notes](https://github.com/starlinghq/starling/blob/main/docs/releases/ai-0.1.0-preview.6.md).
