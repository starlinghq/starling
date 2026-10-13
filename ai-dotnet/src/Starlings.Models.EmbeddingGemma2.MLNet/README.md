# Starlings.Models.EmbeddingGemma2.MLNet

**Preview.** **Experimental, text only.** An ML.NET estimator and transformer that turn a text column into an EmbeddingGemma 2 embedding column (`Vector<Single, 768>`, or a Matryoshka size of 512, 256 or 128, L2-normalized) with the embedder in `Starlings.Models.EmbeddingGemma2`. The API may still change.

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

Status: Linux x64 CPU experimental; over a CUDA session untested.

## License and usage policy

The package is MIT OR Apache-2.0. The model weights and Starling's ONNX graph derived from them are Apache-2.0 (Google DeepMind). The model card states:

> Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
