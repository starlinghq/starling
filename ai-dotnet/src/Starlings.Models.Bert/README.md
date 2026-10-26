# BERT embeddings and reranking

Experimental. This package is not published to nuget.org: it is not part of the 0.2.0 release
(`docs/releases/ai-0.2.0.md`, "Not published"). Build it from this repository, for example with a
`ProjectReference` to `ai-dotnet/src/Starlings.Models.Bert/Starlings.Models.Bert.csproj`. The API may still change.

Experimental CPU fp32 implementations for seven explicitly pinned candidate contracts:
`sentence-transformers/all-MiniLM-L6-v2` (mean/L2, 256 tokens), `BAAI/bge-small-en-v1.5`
(CLS/L2, 512 tokens), `BAAI/bge-base-en-v1.5` (CLS/L2, 512 tokens, 768-wide),
`BAAI/bge-m3` (CLS/L2, 8192 tokens, 1024-wide, XLM-R), `cross-encoder/ms-marco-MiniLM-L6-v2`
(paired inputs, raw scalar logits, 512 tokens), `BAAI/bge-reranker-v2-m3` (XLM-R cross-encoder, raw scalar
logits, 8192 tokens, from the Starling-built graph `Starlinghq/bge-reranker-v2-m3-onnx`), and `nomic-ai/nomic-embed-text-v1.5`
(mean, no L2, 8192 tokens, 768-wide). Candidate parity is
development evidence; protected release qualification requires independent fixture adoption.

Load a reviewed candidate profile with `BertModelProfile.Load`. `BertTextEmbedder.Create`
provides embeddings; `BertReranker.Create` provides scores and document ranking. Both
verify the graph and tokenizer against the profile's immutable lock and load one reusable
ONNX session. Acquisition is explicit through `ModelAssetStore`; inference never downloads.
The application supplies the CPU native ONNX Runtime package.

BGE query instructions are explicit: call `Embed(text, "query")` to apply the declared
instruction; `Embed(text)` embeds text as supplied. Passages never acquire a query prefix.
Reranking returns logits with identity activation, not calibrated probabilities. Input pairs
are truncated with the pinned tokenizer's longest-first policy, including special tokens.

Batches are chunked and right-padded with mask 0: `maxBatchSize` rows per call (default 32).
`Create` also takes two experimental options, both off by default (outputs then unchanged):
`maxBatchTokens` caps padded cells (rows x longest row) per call, and a row longer than the
cap runs alone; `sortByLength` chunks rows in ascending token length and returns results in
input order, so a row's output can differ from the default within padded-batch tolerance. Threads default to two intra-op and one
inter-op worker. Calls are synchronous; cancellation follows ONNX Runtime graph-node
boundaries. Dispose an instance after outstanding calls complete. Streaming, training,
persistence, AOT, GPU providers, and unlisted model contracts are unsupported or untested.

## Profiles

The candidate contracts live in the repository, not in the package:
`model-pipeline/candidates/<name>/profile.json`. `BertModelProfile.Parse` accepts only these seven
repositories, ONNX Runtime 1.30.0 on `CPUExecutionProvider` in fp32, and the exact values below; anything
else throws `NotSupportedException`.

| Candidate | Profile id | Task | Max tokens | Pooling | Width | Normalize | Prompts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `minilm` | `all-minilm-l6-v2/onnx-fp32-cpu` | embeddings | 256 | mean | 384 | l2 | `none` |
| `bge-small` | `bge-small-en-v1.5/onnx-fp32-cpu` | embeddings | 512 | cls | 384 | l2 | `none`, `query` |
| `bge-base` | `bge-base-en-v1.5/onnx-fp32-cpu` | embeddings | 512 | cls | 768 | l2 | `none`, `query` |
| `bge-m3` | `bge-m3/onnx-fp32-cpu` | embeddings | 8192 | cls | 1024 | l2 | `none` |
| `ms-marco` | `ms-marco-minilm-l6-v2/onnx-fp32-cpu` | reranking | 512 | none | 384 (one logit out) | none | `none` |
| `bge-reranker-v2-m3` | `bge-reranker-v2-m3/onnx-fp32-cpu` | reranking | 8192 | none | 1024 (one logit out) | none | `none` |
| `nomic-embed` | `nomic-embed-text-v1.5/onnx-fp32-cpu` | embeddings | 8192 | mean | 768 | none | `none` |

nomic-embed-text-v1.5 vectors are not unit length; normalize them before cosine or dot-product comparison.
Its profile declares only the `none` prompt, so `Embed(text, "query")` throws `ArgumentException`; any task
prefix is part of the text the caller passes.

bge-reranker-v2-m3 has no ONNX graph upstream. Its profile pins `Starlinghq/bge-reranker-v2-m3-onnx`, which holds
the original files of `BAAI/bge-reranker-v2-m3` at `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` unchanged plus an
fp32 export (`model-pipeline/reference/bert/export_reranker_onnx.py`). Pairs are encoded `<s> query </s></s> document </s>`
without segment ids. Scores are raw logits (the model card's `normalize=False` default); apply a sigmoid for its 0-1
mapping. FlagEmbedding's fp16 inference, its LLM-based and layerwise rerankers, and query/passage length splits
(`max_length` per side) are not implemented; truncation is longest-first over the pair.

## Use

```csharp
using Starlings.Acquisition;
using Starlings.Models.Bert;

var profile = BertModelProfile.Load("model-pipeline/candidates/bge-small/profile.json");
using var store = new ModelAssetStore();
ModelSnapshot snapshot = await store.DownloadAsync(profile.RuntimeLock);   // explicit network step

using var embedder = BertTextEmbedder.Create(profile, snapshot.Directory);
float[] query = embedder.Embed("how do magnets work", "query");
float[][] passages = embedder.EmbedBatch(["Magnets attract iron.", "Paris is in France."]);
```

`profile.RuntimeLock` names the files inference needs at the pinned revision: `onnx/model.onnx`, its external
weights `onnx/model.onnx_data` when the profile pins them (bge-m3 and bge-reranker-v2-m3, whose graphs exceed 2 GB),
and `tokenizer.json`; `profile.SourceLock` is the full upstream file list used by reference jobs. `Create` verifies the
directory by size and SHA-256 (`ModelAssetStore.VerifyDirectory`) and never downloads.

## API

- `BertModelProfile`: `Load(path)`, `Parse(json)`, `Id`, `Task` (`BertTask.Embeddings` or `BertTask.Reranking`),
  `MaxTokens` (including special tokens and the prompt), `Pooling`, `HiddenSize`, `Normalize`, `Prompts`,
  `VocabularySize`, `ClsTokenId`, `SepTokenId`, `PadTokenId`, `OutputName`, `TokenizerProfile`, `TokenTypeIds`,
  `SourceLock`, `RuntimeLock`, and the constants `BgeQueryInstruction`, `GraphFile`, `GraphDataFile`, `TokenizerFile`.
- `BertTextEmbedder.Create(profile, snapshot, threads = 2, maxBatchSize = 32)`: `Embed`, `EmbedBatch`
  (input order), `EmbedTokens`, `Tokenize`, `Dimension`, `MaxSequenceLength`, `Profile`, `Diagnostics`, and the owned
  `Session` (advanced). It is an `ITextEmbedder` (`Starlings.Models.Abstractions`): `EmbedBatch(texts, tokenCounts)`
  embeds with the prompt `none` and reports each text's token count. `CountTokens(text)`, `CountTokens(texts)` and
  `Validate(text)` count before truncation without embedding, and `ChunkingTokenizer` (the profile's tokenizer without
  truncation) feeds `TextChunker`; both experimental (`BertCapabilities.TokenCounting`).
- `BertReranker.Create(profile, snapshot, threads = 2, maxBatchSize = 32)`: `Score(BertPair)`, `ScoreBatch`,
  `ScoreTokens`, `Tokenize`, and `Rank(query, documents)`, which returns `BertRankedDocument(Index, Document, Score)`
  by descending score, ties in original order. It is an `IReranker`: `Score(query, documents)` returns the raw logits
  of `ScoreBatch` in document order.
- `BertInput(InputIds, AttentionMask, TokenTypeIds)` for pre-tokenized rows: equal lengths from 2 to `MaxTokens`,
  the profile's CLS id first and SEP id as the last visible token, right padding with the profile's pad id and mask 0.
- `BertCapabilities`: text, batch and cancellation `experimental`; streaming, persistence and training
  `unsupported`; AOT `untested`.

`threads` must be 1 to 64 and `maxBatchSize` 1 to 128. `Create` with a profile of the other task throws
`ArgumentException`.

## Errors

- `NotSupportedException`: the profile JSON or the graph signature differs from an implemented contract.
- `ArgumentException`: the wrong task for `Create`, an undeclared prompt name, or an invalid `BertInput` row.
- `ModelAssetMissingException`, `ModelAssetIntegrityException` (`Starlings.Acquisition`): a runtime file is missing
  or its size or SHA-256 differs.
- `OnnxRuntimeUnavailableException` (`Starlings.Runtime.Onnx`): no native ONNX Runtime 1.30.0; add
  `Microsoft.ML.OnnxRuntime` 1.30.0 to the application.
- `InvalidDataException`: the graph returned a NaN or infinite value.
- `OperationCanceledException`: the token was cancelled; checked between batch chunks and at graph-node boundaries.
