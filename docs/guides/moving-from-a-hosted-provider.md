# Moving from a hosted embedding API to Starling

You have an `IEmbeddingGenerator<string, Embedding<float>>` backed by a hosted provider — this guide starts from OpenAI's `text-embedding-3-small`, and the steps are the same for any other hosted generator. The code change at the call site really is one line, as the README says. The rest of the move is three things a one-line change cannot do for you: re-indexing your stored vectors, the e5 input prefixes, and the change in vector dimensions. This guide uses `multilingual-e5-small`; [choosing an embedding model](choosing-an-embedding-model.md) covers when to pick a different one.

## 1. Swap the generator (the one line)

Before, with the hosted provider:

```csharp
using Microsoft.Extensions.AI;
using OpenAI;

IEmbeddingGenerator<string, Embedding<float>> generator =
    new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY"))
        .GetEmbeddingClient("text-embedding-3-small")
        .AsIEmbeddingGenerator();
```

Install the Starling packages (exactly one ONNX Runtime package), and remove the provider package if nothing else uses it:

```sh
dotnet add package Starlings.Models.E5.Extensions.AI
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0   # or Microsoft.ML.OnnxRuntime.Gpu.Linux
dotnet remove package Microsoft.Extensions.AI.OpenAI
```

After:

```csharp
using Starlings.Models.E5;
using Starlings.Models.E5.Extensions.AI;
using Microsoft.Extensions.AI;

await E5ModelAssets.DownloadAsync();              // once: downloads the pinned model files and verifies their SHA-256
E5ModelFiles files = E5ModelAssets.OpenLocal();   // offline load; inference never downloads anything

using IEmbeddingGenerator<string, Embedding<float>> generator =
    E5EmbeddingGenerator.Create(files, new E5EmbeddingGeneratorOptions { InputType = E5InputType.Query });
```

That is the whole code change: `GenerateAsync` calls, Microsoft.Extensions.AI middleware, and vector-store integrations keep working because the interface is unchanged. With dependency injection the swap is the registration instead: `services.AddE5EmbeddingGenerator(files)`.

Two operational differences come with the swap. The model files (about 487 MB for e5-small) are downloaded once by `DownloadAsync` — run it during deployment or first-run setup, not per request — and embedding now runs on your own CPU or GPU instead of a metered API, so there is no API key, no per-token bill, and no network call at inference time.

## 2. Re-index your stored vectors

Vectors from different models live in different vector spaces. A stored vector produced by your hosted model and a Starling vector of the same text are not comparable: a similarity score computed across the two models is meaningless, and ranking a Starling query vector against hosted document vectors returns noise. The dimensions also change (next section), so most vector stores reject the new vectors against an index created for the old ones.

Plan the move as a rebuild, not an update:

1. Create a new index or collection sized for the new model (384 dimensions for e5-small).
2. Re-embed every document with the Starling generator and write the new vectors there.
3. Switch reads to the new index, verify retrieval quality on your own queries, then drop the old index.

There is no shortcut that keeps the old vectors: budget the re-embedding run as local compute over your whole corpus, and measure retrieval quality on your own documents before switching production traffic — a small local model is not a large hosted one.

## 3. Add the e5 prefixes

e5 models are trained with an instruction prefix: `query: ` before a question and `passage: ` before a document. Embedding without them, or with the wrong one, measurably hurts retrieval. The Starling generator adds nothing by default — configure the input type per generator, as in step 1 for queries and like this for the indexing side:

```csharp
using IEmbeddingGenerator<string, Embedding<float>> passageGenerator =
    E5EmbeddingGenerator.Create(files, new E5EmbeddingGeneratorOptions { InputType = E5InputType.Passage });
```

Or set it per request on a shared generator, with `E5EmbeddingGenerator.InputTypePropertyName`:

```csharp
GeneratedEmbeddings<Embedding<float>> embeddings = await generator.GenerateAsync(
    passages,
    new EmbeddingGenerationOptions
    {
        AdditionalProperties = new() { [E5EmbeddingGenerator.InputTypePropertyName] = "passage" },
    });
```

Whichever route you choose, do not also prepend `query: ` or `passage: ` to the text yourself — a configured input type plus a manual prefix double-prefixes the input. Other model families use different prompts instead of these prefixes (EmbeddingGemma 2 takes `EmbeddingGemma2Prompt.Query` and `EmbeddingGemma2Prompt.Document`); see [choosing an embedding model](choosing-an-embedding-model.md).

## 4. Mind the vector dimensions

Your hosted model's output size and e5-small's differ:

| Model | Output dimensions |
| --- | --- |
| `text-embedding-3-small` (hosted) | 1536 by default |
| `text-embedding-3-large` (hosted) | 3072 by default |
| `text-embedding-ada-002` (hosted) | 1536 |
| `multilingual-e5-small` (Starling) | 384, always |

Recreate vector columns and indexes at 384 when you rebuild (step 2). Code that asked the hosted API to shorten its vectors through `EmbeddingGenerationOptions.Dimensions` needs rethinking: the E5 generator rejects a `Dimensions` value other than 384 with `NotSupportedException` rather than silently truncating. If selectable vector sizes are a requirement, EmbeddingGemma 2 supports Matryoshka sizes of 768, 512, 256, or 128 — again, [choosing an embedding model](choosing-an-embedding-model.md) has the comparison.

## Checklist

- [ ] Package swap and the one-line generator change (or the DI registration).
- [ ] `E5ModelAssets.DownloadAsync()` run once in deployment or setup.
- [ ] Input types configured: queries as queries, passages as passages, no manual prefixes on top.
- [ ] New index created at 384 dimensions and every document re-embedded into it.
- [ ] Retrieval quality measured on your own queries before reads switch over.
