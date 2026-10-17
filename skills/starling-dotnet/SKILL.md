---
name: starling-dotnet
description: Use when adding local AI to a .NET app with the Starlings.* NuGet packages from starlinghq/starling. Covers text embeddings (multilingual-e5-small, EmbeddingGemma 2), image and video embeddings (EmbeddingGemma 2), zero-shot entity extraction (GLiNER2), and local chat (Qwen3.5-0.8B) on ONNX Runtime, CPU or CUDA, through Microsoft.Extensions.AI (IEmbeddingGenerator, IChatClient) or typed APIs.
---

# Using Starling in a .NET app

Starling ships open models as NuGet libraries that run in-process on ONNX Runtime. Each model is pinned to an exact upstream commit, its files are checked by SHA-256, and its outputs are tested against the original Python implementation. Repository: https://github.com/starlinghq/starling.

## Pick the package

| Task | Model | Package | Native runtime the app must add |
| --- | --- | --- | --- |
| Text embeddings, multilingual, small (384 dims) | multilingual-e5-small | `Starlings.Models.E5.Extensions.AI` | `Microsoft.ML.OnnxRuntime` 1.30.0 |
| Text embeddings, higher quality (768 dims, Matryoshka) | EmbeddingGemma 2 | `Starlings.Models.EmbeddingGemma2.Extensions.AI` | `Microsoft.ML.OnnxRuntime` 1.30.0 |
| Image search (text to image), shared text-image space | EmbeddingGemma 2 | above plus `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` | `Microsoft.ML.OnnxRuntime` 1.30.0 |
| Video search | EmbeddingGemma 2 | above plus `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` (FFmpeg 8) or `.Video.FFmpeg9` | `Microsoft.ML.OnnxRuntime` 1.30.0 and FFmpeg shared libraries |
| Entity extraction with any labels (NER) | GLiNER2 | `Starlings.Models.Gliner2` | `Microsoft.ML.OnnxRuntime` 1.30.0 |
| Local chat and generation | Qwen3.5-0.8B | `Starlings.Models.Qwen35.Extensions.AI` | `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 |

ML.NET transformers exist for e5 and EmbeddingGemma 2 (`*.MLNet`). The target framework is `net10.0`.

**Always add exactly one native ONNX Runtime package to the application project**, at the version in the table. The libraries reference only the managed binding. For CUDA on Linux x64, use `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 instead of `Microsoft.ML.OnnxRuntime`, never both.

## Load models: download once, then run offline

Every model has a `*ModelAssets` class:

- `DownloadAsync()` fetches the pinned files, or verifies cached ones. It is the only call that uses the network; call it at setup or deploy time.
- `OpenLocal()` loads from the cache and never downloads. It is synchronous and blocking (file I/O plus SHA-256), and returns the same `*ModelFiles` type as `DownloadAsync`. If a file is missing it throws `ModelAssetMissingException`, and if a file is corrupt, `ModelAssetIntegrityException`. Use it in production paths that must not touch the network.

```csharp
E5ModelFiles files = E5ModelAssets.OpenLocal();                                  // e5, GLiNER2, EmbeddingGemma 2 text
var images = EmbeddingGemma2ModelAssets.OpenLocalImageAssets();                  // EmbeddingGemma 2 image/video
var qwen = Qwen35ModelAssets.OpenLocal(Qwen35Variant.Fp32x);                     // Qwen takes the variant
var custom = E5ModelAssets.OpenLocal(new ModelAcquisitionOptions { CacheDirectory = "/srv/models" });
```

The cache directory is `$STARLING_MODELS` if set, otherwise the user cache directory. To set it in code, pass `new ModelAcquisitionOptions { CacheDirectory = "..." }` (namespace `Starlings.Acquisition`) to `DownloadAsync` or `OpenLocal`. No repository is gated; `HF_TOKEN` is honoured if set. Approximate download sizes: e5 470 MB, GLiNER2 800 MB, EmbeddingGemma 2 up to 2.7 GB with the image assets, Qwen fp32x 3 GB.

Defaults when you pass no options: the CPU execution provider, no prompt or prefix added, and a GLiNER2 confidence threshold of 0.5. Each `Create` call loads its own copy of the model, so create one instance per model and share it rather than creating one per request.

## Verified patterns

### Text embeddings with e5 (`IEmbeddingGenerator`)

```csharp
using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Starlings.Models.E5;
using Starlings.Models.E5.Extensions.AI;

E5ModelFiles files = await E5ModelAssets.DownloadAsync();
using var embedder = E5EmbeddingGenerator.Create(files);

// e5 REQUIRES these prefixes; nothing adds them by default.
var docs = await embedder.GenerateAsync(["passage: The Rhine flows through Cologne."]);
var query = await embedder.GenerateVectorAsync("query: Which river runs through Cologne?");
float score = TensorPrimitives.CosineSimilarity(query.Span, docs[0].Vector.Span);
```

Instead of writing the prefixes, you can set `new E5EmbeddingGeneratorOptions { InputType = E5InputType.Query }` (or `Passage`) when calling `Create`. Don't do both, or the prefix is added twice.

### Text embeddings with EmbeddingGemma 2

```csharp
using Microsoft.Extensions.AI;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Extensions.AI;

EmbeddingGemma2ModelFiles files = await EmbeddingGemma2ModelAssets.DownloadAsync();
using IEmbeddingGenerator<string, Embedding<float>> queries = EmbeddingGemma2EmbeddingGenerator.Create(
    files, new EmbeddingGemma2EmbeddingGeneratorOptions { Prompt = EmbeddingGemma2Prompt.Query });
var v = await queries.GenerateVectorAsync("how do I reset my password");            // 768 dims
var small = await queries.GenerateAsync(["..."], new EmbeddingGenerationOptions { Dimensions = 256 }); // Matryoshka
```

Embed queries with `EmbeddingGemma2Prompt.Query` and documents with `EmbeddingGemma2Prompt.Document`. To use one generator (one model load) for both, set the generator's default to `Query` and override it per request:

```csharp
var asDocuments = new EmbeddingGenerationOptions
{
    AdditionalProperties = new() { [EmbeddingGemma2EmbeddingGenerator.PromptPropertyName] = EmbeddingGemma2Prompt.Document },
};
var docVectors = await queries.GenerateAsync(documents, asDocuments);
```

Task prompts from the model card are also available, for example `EmbeddingGemma2Prompt.FromName("Clustering")` or `"Classification"` (names are case-sensitive; see `EmbeddingGemma2Prompt.Named`). Inputs over 8,192 tokens throw `EmbeddingGemma2InputTooLongException`; they are never silently truncated, so chunk long documents first.

### Image search with EmbeddingGemma 2

```csharp
using System.Numerics.Tensors;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;

EmbeddingGemma2ImageModelFiles files = await EmbeddingGemma2ModelAssets.DownloadImageAssetsAsync();
using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files);

float[] image = embedder.Embed(EmbeddingGemma2ImageDecoder.DecodeFile("photo.jpg")); // images take no prompt
float[] query = embedder.Embed(EmbeddingGemma2MultimodalInput.FromText("a photo of a cat"), EmbeddingGemma2Prompt.Query);
float score = TensorPrimitives.CosineSimilarity(image, query);
```

Use `EmbedBatch` for many images. `EmbeddingGemma2ImageDecoder` reads PNG and JPEG; unsupported images throw `EmbeddingGemma2UnsupportedImageException`. For `IEmbeddingGenerator<DataContent, Embedding<float>>`, use `EmbeddingGemma2MultimodalEmbeddingGenerator`. Video works the same way through the FFmpeg adapters; see `ai-dotnet/samples/EmbeddingGemma2Search`.

### Entity extraction with GLiNER2

```csharp
using Starlings.Models.Gliner2;

Gliner2ModelFiles files = await Gliner2ModelAssets.DownloadAsync();
using var extractor = Gliner2EntityExtractor.Create(files);

var result = extractor.Extract("Tim Cook announced the iPhone 15 in Cupertino.", ["person", "product", "city"]);
foreach (var e in result.Entities)
    Console.WriteLine($"{e.Label}: {e.Text} [{e.Start}, {e.End}) {e.Confidence:F2}");
// person: Tim Cook [0, 8) 1.00 / product: iPhone 15 [23, 32) 1.00 / city: Cupertino [36, 45) 1.00
```

Labels are free text and chosen per call. Offsets index the original string (`text[e.Start..e.End] == e.Text`). Use `ExtractBatch(texts, labels, threshold)` for throughput. `threshold` is a confidence between 0 and 1 (default 0.5; set it for all calls with `Gliner2ExtractorOptions.Threshold`). Raise it for precision, lower it for recall. Inputs over 512 tokens throw `Gliner2InputTooLongException`, so split long text.

### Local chat with Qwen3.5 (`IChatClient`)

```csharp
using Microsoft.Extensions.AI;
using Starlings.Models.Qwen35;
using Starlings.Models.Qwen35.Extensions.AI;

Qwen35ModelFiles files = await Qwen35ModelAssets.DownloadAsync(Qwen35Variant.Fp32x); // Fp32x: CPU or CUDA
using IChatClient chat = Qwen35ChatClient.Create(files);

await foreach (var update in chat.GetStreamingResponseAsync("In one sentence, why is the sky blue?"))
    Console.Write(update.Text);
```

`Qwen35Variant.Fp16x` is CUDA-only and an approximate variant; prefer `Fp32x` unless you measured that Fp16x is good enough. Tool calling (`ChatOptions.Tools`) needs 0.2.0 or later; it is not in 0.1.0-preview.6.

### CUDA

```csharp
using Starlings.Runtime.Onnx;

var session = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 };
using var extractor = Gliner2EntityExtractor.Create(files, new Gliner2ExtractorOptions { Session = session });
```

CPU is the default; CUDA is opt-in per model. Each model's options type has a `Session` property, for example `Gliner2ExtractorOptions` (namespace `Starlings.Models.Gliner2`), `E5EncoderOptions` and `EmbeddingGemma2MultimodalOptions`. Use CUDA only on Linux x64, with `Microsoft.ML.OnnxRuntime.Gpu.Linux` and the CUDA 13 runtime on the loader path.

## Rules of thumb

- Dispose embedders, extractors and chat clients (`using`). They own native sessions.
- Compare embeddings with cosine similarity (`System.Numerics.Tensors.TensorPrimitives.CosineSimilarity`; the package comes in transitively).
- Don't truncate input silently. The libraries reject over-long input with typed exceptions; catch them and chunk.
- Don't add `Microsoft.ML.Tokenizers` or another tokenizer for these models. Tokenization is built in (`Starlings.Tokenizers`), and it matches upstream token for token.
- Check maturity before promising support. Stable today on linux-x64: the CPU profiles of e5, EmbeddingGemma 2 text and image, GLiNER2 and Qwen fp32x, and the CUDA profiles of e5, EmbeddingGemma 2 text and image, GLiNER2 and Qwen fp32x and fp16x, on an NVIDIA GeForce RTX 3090 only (other GPUs are untested). EmbeddingGemma 2 video is experimental on CPU and CUDA. GLiNER2 on Linux needs `MALLOC_MMAP_THRESHOLD_=65536` in the process environment (CPU and CUDA). Windows CPU is experimental. macOS and ARM64 are untested. The current status is in the compatibility matrix linked below.
- Versions: use `0.2.0` (the first release with Stable profiles), and keep every `Starlings.*` package at the same version.

## Where to look

- Complete programs: https://github.com/starlinghq/starling/tree/main/ai-dotnet/samples (`E5Search`, `EmbeddingGemma2Search` with video and CUDA, `Gliner2Ner`, `QwenChat`).
- Model catalog with per-gate evidence: https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md
- Platform and provider status: https://github.com/starlinghq/starling/blob/main/docs/catalog/compatibility-matrix.md
- Benchmarks against Python on the same backend: https://github.com/starlinghq/starling/tree/main/docs/benchmarks
- Report a regression or request a model: https://github.com/starlinghq/starling/issues/new/choose
