# Starlings.Models.E5

**Preview.** Sentence embeddings from [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) at revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, on ONNX Runtime 1.30.0, in-process and offline after an explicit download. The API and the supported platforms may still change. Stable waits for qualification CI (backlog P-018).

- Text goes through the owned tokenizer `Starlings.Tokenizers` (exact against huggingface/tokenizers on the published fixtures). No Microsoft.ML.Tokenizers and no native tokenizer library.
- Batching with right padding, mean pooling over the attention mask, L2 normalization: 384-dimensional unit vectors.
- The package ships the model's asset lock (repository, commit, file sizes and SHA-256), not the 487 MB of model files. Acquisition is an explicit call; inference never downloads.

## Install

```sh
dotnet add package Starlings.Models.E5 --prerelease
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # CPU natives
# or, for CUDA on Linux x64: Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0 (reference exactly one)
```

The library references only `Microsoft.ML.OnnxRuntime.Managed` 1.30.0; the application chooses the native runtime.

## Use

```csharp
using Starlings.Models.E5;
using Starlings.Runtime.Onnx;

await E5ModelAssets.DownloadAsync();                 // once, explicit, needs the network; verifies SHA-256
E5ModelFiles files = E5ModelAssets.OpenLocal();      // offline, verified, never downloads

using var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
{
    Session = new OnnxSessionOptions { ExpectedModelSha256 = E5ModelAssets.ModelSha256 },
});
var embedder = new E5TextEmbedder(E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256), encoder);

float[] query = embedder.Embed(E5TextEmbedder.QueryPrefix + "how much protein should a female eat");
float[][] passages = embedder.EmbedBatch([E5TextEmbedder.PassagePrefix + "As a general guideline, ..."]);
```

e5 expects `"query: "` or `"passage: "` at the start of every text; nothing is added for you. Vectors are normalized, so cosine similarity is the dot product.

## Status

Profile `multilingual-e5-small/onnx-fp32-cpu`: Linux x64 CPU experimental (46 fixture texts within 2.4e-7 of the PyTorch reference, bound 1e-5). Windows, macOS, ARM64, Native AOT: untested. Evidence, known issues and reproduction commands: [release notes](https://github.com/starlinghq/starling/blob/main/docs/releases/ai-0.1.0-preview.5.md). Model license: MIT, stated only in the model card.
