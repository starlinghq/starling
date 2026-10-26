# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Starlings.Models.Abstractions and the encoder de-duplication (backlog A2, D3)

**New package `Starlings.Models.Abstractions`** (references only `Starlings.Acquisition`; no ONNX Runtime). `Starlings.Models.E5`, `Starlings.Models.EmbeddingGemma2`, `Starlings.Models.Gliner2` and `Starlings.Models.Bert` reference it. Its types, all additive:

- `CapabilityStatus`: one status enum for every package.
- `ModelProfileCapabilities` (with `ProfileMaturity`, `TruncationPolicy`, `ProfilePlatform`): the machine-readable capability record of a profile, generated from its registry manifest by `model-pipeline catalog` into each package's `ProfileCapabilities.g.cs`. Contract fields: profile id, maturity, max input tokens, truncation policy, dimension, qualified providers. Informational: tasks, platform rows, stop token ids, the manifest's capabilities map. `docs/catalog/catalog.json` gains the same `behavior` fields and `qualified_providers` per profile. `model-pipeline catalog --check` (CI) fails when a generated file is stale, and each package's contract tests compare every record with its manifest field for field.
- `ITextEmbedder`: implemented by `E5TextEmbedder`, `EmbeddingGemma2TextEmbedder` (new overload `EmbedBatch(texts, tokenCounts, cancellationToken)` with no prompt; new `Dimension` and `MaxSequenceLength`) and `BertTextEmbedder` (new overload with the prompt `none`; new `MaxSequenceLength`). `E5TextEmbedder` gains `Dimension` and `MaxSequenceLength`.
- `IReranker`: implemented by `BertReranker` (new `Score(query, documents, cancellationToken)`, the raw logits of `ScoreBatch`).
- `IModelAssets<TFiles>`: `E5ModelAssets.Assets`, `EmbeddingGemma2ModelAssets.Assets` and `Gliner2ModelAssets.Assets` expose the static lock and acquisition members as an instance.
- `InputTooLongException` (an `ArgumentException` with `InputIndex`, `TokenCount`, `MaxTokens`): now the base of `EmbeddingGemma2InputTooLongException` and `Gliner2InputTooLongException`. `E5Encoder` throws it (instead of a plain `ArgumentException`, same message) for an id sequence above `MaxSequenceLength`.

**New members on the capability classes.** `E5Capabilities`, `EmbeddingGemma2Capabilities` and `Gliner2Capabilities` gain `Statuses` (the package-wide table in the shared enum) and `ProfileCapabilities` (the generated record of every profile the package serves, Stable or not).

**Obsolete, removed in the next minor version.** Each still works and returns the same values as before:

| Obsolete member | Replacement |
| --- | --- |
| `Starlings.Models.E5.CapabilityStatus`, `EmbeddingGemma2CapabilityStatus`, `Gliner2CapabilityStatus` | `Starlings.Models.CapabilityStatus` (same names and order) |
| `E5Capabilities.All`, `EmbeddingGemma2Capabilities.All`, `Gliner2Capabilities.All` | `XCapabilities.Statuses` |
| `E5Capabilities.Profiles`, `EmbeddingGemma2Capabilities.Profiles`, `Gliner2Capabilities.Profiles` | `XCapabilities.ProfileCapabilities[profile].Capabilities` |
| `EmbeddingGemma2InputTooLongException.MaximumTokens` | `MaxTokens` (the shared name; GLiNER2 already used it) |

**Encoder de-duplication, outputs unchanged.** Right-padded packing, mean and CLS pooling with L2 normalization (including the Matryoshka truncation) and the chunk run moved into one internal `EncoderCore` in `Starlings.Runtime.Onnx`, used by `E5Encoder`, `EmbeddingGemma2Encoder` (text and the multimodal text path) and the BERT route. The arithmetic and its order are unchanged. On CPU, embeddings and reranker scores from the moved code are bit for bit those of the code it replaces for e5, EmbeddingGemma 2 (768 and 128 dimensions, prompts, token budget with length sorting) and four BERT-route profiles (mean pooling, CLS pooling, an unnormalized embedder and a reranker). The tolerance-based CPU parity tests of these packages, the EmbeddingGemma 2 image tests included, pass before and after. The ML.NET transform internals (data view, cursors, bindings, row mapper) and the request and result handling of the two text `IEmbeddingGenerator`s are shared source now instead of two copies; public types and behavior are unchanged.

**Capability statuses.** None changed. No Stable profile's contract, gate, tolerance or `protected/` file changed.

## Token counting, chunking, streaming embeddings and an execution scheduler (backlog C2 to C5)

All additive and experimental; no Stable profile's contract, capability status, gate, tolerance or `protected/` file changed.

- **Token counting (C3).** `ITextEmbedder` gains `CountTokens(text)`, `CountTokens(texts)` and `Validate(text)`, implemented by `E5TextEmbedder`, `EmbeddingGemma2TextEmbedder` (with prompt overloads) and `BertTextEmbedder`. Counts include special tokens and are taken before truncation. `Validate` returns the new `TokenLimitCheck` (`TokenCount`, `MaxTokens`, `ExceedsLimit`, `TokenCountIsLowerBound`) and never throws. Capability `token_counting` (e5, EmbeddingGemma 2) and `BertCapabilities.TokenCounting`: experimental.
- **Chunking (C2).** `TextChunker` in `Starlings.Models.Abstractions`: sliding windows of tokens with overlap and a prefix budget over a model's tokenizer (`IChunkingTokenizer`, implemented by `E5Tokenizer`, `EmbeddingGemma2Tokenizer` and `BertTextEmbedder.ChunkingTokenizer`), returning chunk offsets, text and token counts; `EmbedPooled` returns the re-normalized mean of the normalized chunk vectors. Pooled vectors are not the reference model's output and carry no parity claim. Capability `chunking`: experimental.
- **EmbeddingGemma 2 rejects certainly-too-long text without tokenizing it (C2).** `EmbeddingGemma2Tokenizer` computes the longest token of its `tokenizer.json` at load (31 UTF-16 code units for the pinned file, `MaxCharactersPerToken`). The text embedder, the text generator (its micro-batching path included) and the new `EmbeddingGemma2Tokenizer.Tokenize(text, maxTokens)` reject text above `(limit - 2) * 31` code units (253,890 at the default 8,192 tokens) before tokenizing it, with `EmbeddingGemma2InputTooLongException` whose `TokenCount` is that lower bound and whose new `TokenCountIsLowerBound` (on `InputTooLongException`) is true. Shorter text is checked exactly, as before; embeddings are unchanged.
- **Streaming (C4).** `EmbedStreamAsync(IAsyncEnumerable<string>, options, cancellationToken)` on `E5EmbeddingGenerator` and `EmbeddingGemma2EmbeddingGenerator`: values are tokenized as they arrive and grouped exactly as the encoder groups an explicit batch, so with `SortByLength` off the embeddings equal one `GenerateAsync` call bit for bit. New capability `extensions_ai.stream`: experimental. The profile capability `streaming` stays unsupported on every profile.
- **Execution scheduler (C5).** `ExecutionScheduler` (`TaskScheduler?`) on `E5EmbeddingGeneratorOptions`, `EmbeddingGemma2EmbeddingGeneratorOptions`, `EmbeddingGemma2MultimodalEmbeddingGeneratorOptions` and `Gliner2ExtractorOptions`. Null (default) keeps today's inline behavior; a scheduler runs `GenerateAsync`, the stream's encoder runs and `ExtractAsync` as tasks on it. Every async member documents the contract: inline unless a scheduler is set. New capabilities `extensions_ai.execution_scheduler` (e5, EmbeddingGemma 2) and `execution_scheduler` (GLiNER2): experimental.

## Upgrading

- Nothing breaks: every former member is still there, marked `[Obsolete]`. Projects that treat warnings as errors see CS0618 until they move to the replacements above.
- A file that imports both `Starlings.Models` and `Starlings.Models.E5` sees two types named `CapabilityStatus` until the obsolete one is removed: qualify it or alias it (`using Status = Starlings.Models.CapabilityStatus;`).
- Code that tested for the exact type `ArgumentException` (not `is ArgumentException`) on an over-long e5 id sequence now gets `InputTooLongException`, a subclass.
- A class outside these packages that implements `ITextEmbedder` (new in this release) must also implement `CountTokens(text)`, `CountTokens(texts)` and `Validate(text)`.
- Add `Starlings.Models.Abstractions` at the same version as the other `Starlings.*` packages; it comes in transitively with e5, EmbeddingGemma 2 and GLiNER2.
