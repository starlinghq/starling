# Starlings.Models.EmbeddingGemma2.MLNet

Text only. An ML.NET estimator and transformer that turn a text column into an EmbeddingGemma 2 embedding column (`Vector<Single, 768>`, or a Matryoshka size of 512, 256 or 128, L2-normalized) with the embedder in `Starlings.Models.EmbeddingGemma2`. The API may still change.

## Maturity

<!-- catalog:maturity profiles=embeddinggemma-2/onnx-* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `embeddinggemma-2/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Install

```sh
dotnet add package Starlings.Models.EmbeddingGemma2.MLNet --prerelease
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # or Microsoft.ML.OnnxRuntime.Gpu.Linux; exactly one
```

## Use

```csharp
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.MLNet;
using Microsoft.ML;

var ml = new MLContext();
var files = EmbeddingGemma2ModelAssets.OpenLocal();      // offline, verified
// The pinned graph plus the owned tokenizer (Starlings.Tokenizers, profile gemma-bpe-embeddinggemma2).
using var transformer = ml.Transforms.ApplyEmbeddingGemma2Embedding(files, "Embedding", "Text",
        new EmbeddingGemma2EmbeddingOptions { Prompt = EmbeddingGemma2Prompt.Document, BatchSize = 8 })
    .Fit(data);
IDataView embedded = transformer.Transform(data);
```

- One inference per row, on the first output getter call; rows never read are never embedded.
- Prompts are explicit (`Prompt`, default none). The dimension comes from `Encoder.Dimension` (768, 512, 256 or 128).
- `BatchSize` above 1 batches sequential, unshuffled cursors with a look-ahead cursor; row ids are checked.
- Saving the transformer throws (`mlnet.persistence: unsupported`); cursors take no cancellation token.

Status: supported on the Stable `embeddinggemma-2/onnx-fp32-cpu` profile (Linux x64 CPU; G7 passed; the contract is in the `Starlings.Models.EmbeddingGemma2` README); over a CUDA session experimental (document rows within the CUDA profile's bound); Native AOT unsupported (Microsoft.ML).

## License and usage policy

The package is MIT OR Apache-2.0. The model weights and Starling's ONNX graph derived from them are Apache-2.0 (Google DeepMind). The model card states:

> Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
