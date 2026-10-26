# Starlings.Models.E5.MLNet

[ML.NET](https://learn.microsoft.com/dotnet/machine-learning/) estimator and transformer over the multilingual-e5-small embedder in `Starlings.Models.E5`: a text column becomes a fixed-size `Vector<Single, 384>` embedding column (L2-normalized) and, optionally, an `Int32` token-count column. Same tokenizer, encoder and numbers as the typed API. The API may still change.

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
dotnet add package Starlings.Models.E5.MLNet
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # or Microsoft.ML.OnnxRuntime.Gpu.Linux; exactly one
```

## Use

```csharp
using Starlings.Models.E5;
using Starlings.Models.E5.MLNet;
using Microsoft.ML;

await E5ModelAssets.DownloadAsync();                 // explicit acquisition, once
E5ModelFiles files = E5ModelAssets.OpenLocal();      // offline

var ml = new MLContext();
IDataView data = ml.Data.LoadFromEnumerable(new[] { new Doc { Text = "how much protein should a female eat" } });
var pipeline = ml.Transforms.ApplyE5Embedding(files, outputColumnName: "Embedding", inputColumnName: "Text",
    new E5EmbeddingOptions { Prefix = E5TextPrefix.Query, TokenCountColumnName = "Tokens" });   // default: no prefix
using E5EmbeddingTransformer model = pipeline.Fit(data);     // loads the model; trains nothing; owns the session
IDataView embedded = model.Transform(data);                  // lazy

// Or over an embedder you already have (the pipeline then owns nothing):
// ml.Transforms.ApplyE5Embedding(embedder, "Embedding", "Text")
```

- **One inference per row.** A row is embedded the first time an output getter is called on it; further getter calls on that row (embedding or token count, any number of times) reuse the result. Rows whose output getters are never called are not embedded, and cursors that request no output column never read the text.
- **Batching.** `BatchSize = 1` (default) embeds row by row and equals `E5TextEmbedder.Embed` exactly. `BatchSize > 1` batches sequential unshuffled cursors through a second, look-ahead cursor over the text column (upstream transforms of that column run twice); row ids are compared and a mismatch throws. Batches are padded to their longest row: on one CPU this was 2.7x faster for 256 similar short texts and 4x slower for the 46 fixture texts (2 to 512 tokens). Shuffled cursors, parallel cursor sets and prediction engines stay row-wise.
- **Threads.** Views can be cursored concurrently; `GetRowCursorSet` returns one wrapped cursor per input cursor (row-wise), embedding in parallel on the shared, thread-safe encoder.
- **Null and empty text** are the same in ML.NET (an empty `ReadOnlyMemory<char>`) and are embedded as the empty string (`<s></s>`, 2 tokens).
- **Cancellation: none.** ML.NET cursors take no `CancellationToken`; a getter that is embedding returns when the row or batch is done. Stop iterating and dispose the cursor to stop. Use `E5TextEmbedder.EmbedBatch(texts, cancellationToken)` for cancellable work.
- **Persistence: unsupported.** `mlContext.Model.Save` throws `NotSupportedException` for this transformer, alone or in a chain. The model contract is the pinned asset lock (`E5ModelAssets.Lock`), resolved at run time: save the rest of the pipeline and append `ApplyE5Embedding(E5ModelAssets.OpenLocal(), ...)` with the same column names and options after loading it.
- Microsoft.ML brings Microsoft.ML.CpuMath, Microsoft.ML.DataView, Newtonsoft.Json, System.CodeDom and the LdaNative/CpuMathNative native libraries; no tokenizer package.

Status: pipeline composition (row-wise and look-ahead batching) supported on the Stable e5 `onnx-fp32-cpu` profile (Linux x64 CPU; G7 passed; the contract is in the `Starlings.Models.E5` README); supported over a CUDA session on the Stable `onnx-fp32-cuda` profile (Linux x64, RTX 3090; rerun by the per-release GPU regression); Native AOT unsupported (Microsoft.ML); persistence, cancellation and training unsupported. Listed in the [0.2.0 release notes](https://github.com/starlinghq/starling/blob/main/docs/releases/ai-0.2.0.md) (0.2.0 adds the maturity table; 0.1.0-preview.3 to 0.1.0-preview.6 rebuilt it unchanged); evidence in the [repository README](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#mlnet-embedding-transform-p-012).
