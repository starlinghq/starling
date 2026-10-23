# Starlings.Models.E5.Extensions.AI

[Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) `IEmbeddingGenerator<string, Embedding<float>>` over the multilingual-e5-small embedder in `Starlings.Models.E5`: the same tokenizer, encoder and numbers, usable with Microsoft.Extensions.AI middleware and dependency injection. The API may still change.

## Maturity

<!-- catalog:maturity profiles=multilingual-e5-small/onnx-fp32-cpu,multilingual-e5-small/onnx-fp32-cuda -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Install

```sh
dotnet add package Starlings.Models.E5.Extensions.AI
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
- `GenerateAsync` runs synchronously on the calling thread and returns a completed task; failures and cancellation are reported through that task, never thrown synchronously.
- Unsupported options (`Dimensions` other than 384, another `ModelId`, a `RawRepresentationFactory`, unknown additional properties) fail with `NotSupportedException` instead of being ignored.

## Micro-batching (experimental, opt-in)

```csharp
var generator = E5EmbeddingGenerator.Create(files, new E5EmbeddingGeneratorOptions
{
    MicroBatching = new() { MaxRows = 8, MaxWait = TimeSpan.FromMilliseconds(2) },   // null (default): off
});
```

With `MicroBatching` set, a `GenerateAsync` call with exactly one value is tokenized and validated on the calling thread (an invalid value fails only its own call), then queued; one dispatcher thread embeds the queued values of concurrent callers in padded batches of at most `MaxRows` values and `MaxTokens` padded token cells, dispatched when full or when the oldest has waited `MaxWait` (default zero: no added latency, a batch holds what queued while the previous one ran). Outputs then depend on the concurrent traffic: equal to a single call within the profile's padded-batch tolerance, not bit for bit reproducible. Calls with several values run as one explicit batch, as without the option. Cancelling a queued call removes it; cancelling a dispatched one cancels only that caller's task (the batch keeps running for the others). A full queue (`QueueCapacity`) makes callers wait for a slot. `Dispose` faults queued calls with `ObjectDisposedException` and waits for a running batch. `MicroBatchingStatistics` reports batches, items, padding and queue and run time. Status: experimental (capability `extensions_ai.micro_batching`): weight-free tests and one real-model test within the padded-batch tolerance on Linux x64 CPU; no benchmark adoption decision, no CUDA run and no soak yet.

Status: supported on the Stable e5 `onnx-fp32-cpu` profile (Linux x64 CPU; G7 passed; the contract is in the `Starlings.Models.E5` README); supported over a CUDA session on the Stable `onnx-fp32-cuda` profile (Linux x64, RTX 3090; 46 protected texts within the CUDA profile's bound, rerun by the per-release GPU regression; the contract is the README's "Stable contract: CUDA"). See the [0.2.0 release notes](https://github.com/starlinghq/starling/blob/main/docs/releases/ai-0.2.0.md).
