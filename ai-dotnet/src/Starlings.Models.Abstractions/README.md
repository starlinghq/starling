# Starlings.Models.Abstractions

Types the Starlings model packages share, with no ONNX Runtime dependency (it references only `Starlings.Acquisition`, which needs nothing beyond the BCL). Code that works with several models can depend on these instead of each package's own types. The API may still change.

| Type | What it is | Implemented by |
| --- | --- | --- |
| `CapabilityStatus` | `Supported`, `Unsupported`, `Experimental`, `Untested`. A missing entry never means supported. | the capability tables of `Starlings.Models.E5`, `Starlings.Models.EmbeddingGemma2`, `Starlings.Models.Gliner2` (`XCapabilities.Statuses`) |
| `ModelProfileCapabilities` | The capability record of one registry profile, generated from its manifest (see below) | `E5Capabilities.ProfileCapabilities`, `EmbeddingGemma2Capabilities.ProfileCapabilities`, `Gliner2Capabilities.ProfileCapabilities` |
| `ITextEmbedder` | `Dimension`, `MaxSequenceLength`, `EmbedBatch(texts, tokenCounts, cancellationToken)`, `CountTokens(text)`, `CountTokens(texts)`, `Validate(text)` (a `TokenLimitCheck`) | `E5TextEmbedder`, `EmbeddingGemma2TextEmbedder` (no prompt), `BertTextEmbedder` (prompt `none`) |
| `IReranker` | `Score(query, documents, cancellationToken)`: one raw score per document | `BertReranker` |
| `Reranking`, `RerankedItem<T>` | Rerank a first-stage retriever's top-k (for example a `Microsoft.Extensions.VectorData` `SearchAsync`) with an `IReranker` or any `Func<string, IReadOnlyList<string>, Task<float[]>>` (Experimental, see below) | |
| `IModelAssets<TFiles>` | A package's asset lock and explicit acquisition (`DownloadAsync`, `OpenLocal`, `OpenDirectory`) as one object | `E5ModelAssets.Assets`, `EmbeddingGemma2ModelAssets.Assets`, `Gliner2ModelAssets.Assets` |
| `TokenLimitCheck` | `TokenCount`, `MaxTokens`, `ExceedsLimit`, `TokenCountIsLowerBound`: an input checked against the limit without embedding it | returned by `ITextEmbedder.Validate` |
| `IChunkingTokenizer`, `TokenSpan` | Untruncated `CountTokens`, `SpecialTokenCount` and `GetTokenSpans` (UTF-16 offsets of each token): what `TextChunker` needs | `E5Tokenizer`, `EmbeddingGemma2Tokenizer`, `BertTextEmbedder.ChunkingTokenizer` |
| `TextChunker`, `TextChunkerOptions`, `TextChunk` | Sliding windows of tokens with overlap over a model's tokenizer, returning chunk offsets; optional pooled embedding (Experimental, see below) | |
| `InputTooLongException` | An `ArgumentException` for input above the limit, with `InputIndex`, `TokenCount`, `MaxTokens` and `TokenCountIsLowerBound` | thrown by `E5Encoder` for id sequences above `MaxSequenceLength`; base of `EmbeddingGemma2InputTooLongException` and `Gliner2InputTooLongException` |

## Maturity

<!-- catalog:maturity profiles=multilingual-e5-small/onnx-fp32-cpu,multilingual-e5-small/onnx-fp32-cuda,embeddinggemma-2/*,gliner2-base-v1/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/image-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/image-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `embeddinggemma-2/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/video-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/video-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `gliner2-base-v1/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; linux-x64 cuda: supported; windows-x64 cpu: experimental |
| `gliner2-base-v1/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `multilingual-e5-small/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Capability records

`ModelProfileCapabilities` is generated, not hand-written: `model-pipeline catalog` writes each package's `ProfileCapabilities.g.cs` from the registry manifests (`model-pipeline/registry/<profile>/manifest.yaml`), the same input as the published catalog (`docs/catalog/catalog.json`), and CI fails when a generated file is stale. Each package's contract tests check every record against its manifest with an independent reader.

| Field | Manifest source | Status |
| --- | --- | --- |
| `ProfileId` | `id` | contract |
| `Maturity` | `maturity` | contract |
| `MaxInputTokens` | `behavior.max_input_tokens` | contract |
| `Truncation` (`Rejects`, `TruncatesRight`, `TruncatesLeft`, `Chunks`) | `behavior.truncation` | contract |
| `Dimension` (null for a profile without embeddings) | `behavior.defaults.dimension` | contract |
| `QualifiedProviders` | providers of the `compatibility` rows with status `supported` | contract |
| `Tasks` | `behavior.tasks` | informational |
| `Platforms` | every `compatibility` row | informational |
| `StopTokenIds` | `behavior.defaults.stop_token_ids` | informational |
| `Capabilities` | the `capabilities` map, key for key | informational |

Contract fields of a Stable profile change only with a new profile or a release note under the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md). The fine-grained, package-wide keys (for example `batch.token_budget`) stay in each package's `XCapabilities.Statuses`.

```csharp
using Starlings.Models;
using Starlings.Models.E5;

ModelProfileCapabilities e5 = E5Capabilities.ProfileCapabilities[E5Capabilities.StableProfile];
// e5.MaxInputTokens == 512, e5.Truncation == TruncationPolicy.TruncatesRight, e5.Dimension == 384

static float[][] EmbedAll(ITextEmbedder embedder, IReadOnlyList<string> texts)
{
    var counts = new int[texts.Count];
    return embedder.EmbedBatch(texts, counts);
}
```

## Token counting and chunking (experimental)

Every text embedder counts tokens without embedding: `CountTokens(text)` and `CountTokens(texts)` give the tokens a text needs, special tokens included, before any truncation (so a truncating profile can report more than `MaxSequenceLength`). `Validate(text)` returns a `TokenLimitCheck` and never throws; `ExceedsLimit` means the profile would truncate the text (e5, BERT) or reject it (EmbeddingGemma 2). An implementation may skip tokenization when the length alone proves a text too long; `TokenCountIsLowerBound` is then true. Capability `token_counting`: experimental.

`TextChunker` splits a long text into windows of at most `MaxTokens` tokens (special tokens and `Prefix` included), each repeating `OverlapTokens` tokens of the previous one, using the model's own tokenizer through `IChunkingTokenizer`. Each `TextChunk` has `Start`, `Length`, `Text` and `TokenCount`; every chunk is tokenized again with the prefix and shrunk until it fits, because a substring can need more tokens alone than inside the whole text. Text between two tokens at a chunk boundary (usually a space) belongs to neither chunk.

```csharp
var chunker = new TextChunker(e5.ChunkingTokenizer!, new TextChunkerOptions
{
    MaxTokens = e5.MaxSequenceLength,
    OverlapTokens = 32,
    Prefix = E5TextEmbedder.PassagePrefix,
});
IReadOnlyList<TextChunk> chunks = chunker.Chunk(document);
float[] pooled = chunker.EmbedPooled(e5, document);
```

`EmbedPooled` embeds every chunk (prefix prepended) in one `EmbedBatch` call and returns the mean of the L2-normalized chunk vectors, L2-normalized again (`TextChunker.Pool`). **A pooled vector is not the reference model's output and carries no parity claim**: the reference truncates or rejects such text, and no fidelity gate covers pooled vectors. Each chunk's own embedding is the model's output for that chunk text, with the profile's usual fidelity. Capability `chunking`: experimental.

## Reranking retrieval results (experimental)

Microsoft.Extensions.AI has no reranker abstraction, so `Reranking` connects an `IReranker` (today `BertReranker`, for example `bge-reranker-v2-m3`) to a retrieval pipeline without referencing `Microsoft.Extensions.VectorData` or `Microsoft.Extensions.AI`: the candidates are any sequence, and a selector picks the text to score.

- `reranker.AsScoringFunction(cancellationToken)` returns a `Func<string, IReadOnlyList<string>, Task<float[]>>`: `Score(query, documents)` as a delegate, for pipelines that take a scorer rather than an `IReranker`.
- `reranker.RerankAsync(query, candidates, text, top, cancellationToken)` reads an `IAsyncEnumerable<T>` (such as the results of `SearchAsync`) to the end, scores all candidates with one `Score` call, and returns `RerankedItem<T>(Item, Index, Score)` by descending score, ties in input order, keeping at most `top`. `Reranking.RerankAsync(query, candidates, text, score, top, cancellationToken)` does the same with any scorer delegate. `reranker.Rerank(...)` is the synchronous form over an `IEnumerable<T>`.

Scoring runs on the calling thread, and the delegate returns a completed task (failures and cancellation are in the task, never thrown synchronously), as the Starlings `IEmbeddingGenerator` implementations do by default. No candidates means no reranker call and an empty result. A text selector that returns null throws `ArgumentException`; a scorer whose score count differs from the candidate count throws `InvalidOperationException`. Scores are the reranker's raw output (a logit for `BertReranker`), so `RerankedItem<T>.Score` is comparable only within one call; the first-stage score stays available on the item.

```csharp
using Microsoft.Extensions.VectorData;
using Starlings.Models;
using Starlings.Models.Bert;

// collection: a VectorStoreCollection<string, Passage> whose embedding generator is, for example, e5's.
using var reranker = BertReranker.Create(BertModelProfile.BuiltIn("bge-reranker-v2-m3"), rerankerSnapshot);
string question = "Which river flows through Cologne?";
IReadOnlyList<RerankedItem<VectorSearchResult<Passage>>> best = await reranker.RerankAsync(
    question,
    collection.SearchAsync("query: " + question, top: 20), // the embedder's top-k, with e5's query prefix
    hit => hit.Record.Text,                                // the reranker scores the plain query and passage text
    top: 3);
foreach (var r in best)
    Console.WriteLine($"{r.Score:F3} (was #{r.Index + 1}, {r.Item.Score:F3})  {r.Item.Record.Text}");
```

The helpers are API only: their maturity is the reranker profile's (every reranker profile is Experimental; see the `Starlings.Models.Bert` README).

## Migrating from the per-package types

The per-package status enums (`Starlings.Models.E5.CapabilityStatus`, `EmbeddingGemma2CapabilityStatus`, `Gliner2CapabilityStatus`), the tables typed with them (`XCapabilities.All`, `XCapabilities.Profiles`) and `EmbeddingGemma2InputTooLongException.MaximumTokens` are marked `[Obsolete]` and are removed in the next minor version. Use `Starlings.Models.CapabilityStatus`, `XCapabilities.Statuses`, `XCapabilities.ProfileCapabilities[profile].Capabilities` and `MaxTokens`. The enum values have the same names and order, so a cast keeps the meaning. A file that imports both `Starlings.Models` and `Starlings.Models.E5` sees two types named `CapabilityStatus` until the obsolete one is removed; qualify it or use an alias (`using Status = Starlings.Models.CapabilityStatus;`).
