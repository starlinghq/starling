# Starlings.Models.EmbeddingGemma2.Extensions.AI

**Preview.** **Experimental.** [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) `IEmbeddingGenerator<string, Embedding<float>>` over the EmbeddingGemma 2 text embedder (and generators for images and videos, [below](#images-and-videos)) in `Starlings.Models.EmbeddingGemma2`: the same tokenizer, encoder and numbers, usable with Microsoft.Extensions.AI middleware and dependency injection. The API may still change.

## Install

```sh
dotnet add package Starlings.Models.EmbeddingGemma2.Extensions.AI --prerelease
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # or Microsoft.ML.OnnxRuntime.Gpu.Linux; exactly one
```

## Use

```csharp
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Extensions.AI;
using Microsoft.Extensions.AI;

EmbeddingGemma2ModelFiles files = EmbeddingGemma2ModelAssets.OpenLocal();   // offline, verified

// The pinned graph plus the owned tokenizer (Starlings.Tokenizers, profile gemma-bpe-embeddinggemma2).
using IEmbeddingGenerator<string, Embedding<float>> generator = EmbeddingGemma2EmbeddingGenerator.Create(files,
    new EmbeddingGemma2EmbeddingGeneratorOptions { Prompt = EmbeddingGemma2Prompt.Query });   // the default adds nothing
var embeddings = await generator.GenerateAsync(["how much protein should a female eat"],
    new EmbeddingGenerationOptions { Dimensions = 256 });                                     // 768, 512, 256 or 128
```

- Or with DI: `services.AddEmbeddingGemma2EmbeddingGenerator(files)`.
- Prompts are explicit: `Prompt` per generator, or per request with `AdditionalProperties["embeddinggemma2_prompt"]` (an `EmbeddingGemma2Prompt`, `"none"`, or one of the 20 names in the model's `config_sentence_transformers.json`, case-sensitive).
- `Dimensions` maps to the Matryoshka sizes 768, 512, 256 and 128 (truncate, then normalize again); other values fail with `NotSupportedException`.
- Inputs above 8,192 tokens fail with `EmbeddingGemma2InputTooLongException`; nothing is truncated.
- `GenerateAsync` runs synchronously on the calling thread and returns a completed task.
- Unsupported options (another `ModelId`, `RawRepresentationFactory`, unknown additional properties) fail with `NotSupportedException` instead of being ignored.

Status: Linux x64 CPU experimental (fixture texts with and without prompts within the binding bound); over CUDA sessions experimental (text, image and video generators within each CUDA profile's bound).

## Images and videos

`EmbeddingGemma2MultimodalEmbeddingGenerator` (over `EmbeddingGemma2MultimodalEmbedder`, experimental) implements `IEmbeddingGenerator<DataContent, Embedding<float>>` (one image or video per value) and `IEmbeddingGenerator<IEnumerable<AIContent>, Embedding<float>>` (text, image and video parts in order). `DataContent` holds encoded bytes, so decoding is a hook: `ImageDecoder` for `image/*` (for example `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp`) and `VideoDecoder` for `video/*` (for example `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` on your own FFmpeg 8.x):

```csharp
var decoder = new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = "/opt/ffmpeg-8/lib" });
using var media = new EmbeddingGemma2MultimodalEmbeddingGenerator(embedder, new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions
{
    ImageDecoder = (bytes, _) => EmbeddingGemma2ImageDecoder.Decode(bytes.Span),
    VideoDecoder = (bytes, _) => decoder.Decode(bytes).Video,   // upstream's sampling: 1 fps, at most 32 frames
});
```

Values with no matching decoder, other media types (audio, documents), `UriContent` and other content types fail with `NotSupportedException`; text containing media placeholders fails with `ArgumentException`; decoded frames without timing fail with `EmbeddingGemma2VideoTimingRequiredException` unless the embedder allows them. Usage counts every media token.

## License and usage policy

The package is MIT OR Apache-2.0. The model weights and Starling's ONNX graph derived from them are Apache-2.0 (Google DeepMind). The model card states:

> Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
