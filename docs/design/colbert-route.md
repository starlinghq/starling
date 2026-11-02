# A late-interaction (ColBERT) route: design note

Status: **proposed**, 2026-11-01 (#113). Nothing in this note changes product code, a capability status, a manifest or a tolerance. It is input for a maintainer decision, not a support claim.

## Problem

`colbert-ir/colbertv2.0` (MIT) scores a query against a document with late interaction: every token gets its own vector, and the score is the sum over query tokens of the best dot product with any document token (MaxSim). Starling's encoder route, `bert-onnx-cpu-v1`, returns one pooled vector per text (`task: embeddings`) or one logit per pair (`task: reranking`); its contract table has no way to say "keep every token", and `ITextEmbedder` and `IReranker` in `Starlings.Models.Abstractions` cannot express a multi-vector result.

The checkpoint's own `artifact.metadata` gives the contract: `dim` 128, `query_maxlen` 32, `doc_maxlen` 180, `mask_punctuation` true, `attend_to_mask_tokens` false, `similarity` cosine. The backbone is BERT base uncased (`model_type` `bert`, `vocab_size` 30522, `do_lower_case` true).

## Constraints

- **Processing contract to reproduce exactly** (from the colbert-ai source at a pinned commit; the markers and padding below are to be verified there, not taken from this note):
  - a marker token after `[CLS]`: `[Q]` for queries, `[D]` for documents, conventionally the vocabulary's `[unused0]` and `[unused1]`;
  - queries padded with `[MASK]` to 32 tokens. `attend_to_mask_tokens` decides only whether the encoder's attention mask covers the `[MASK]` padding (false here, so the encoder does not attend to it); the `[MASK]` output vectors are always kept in the query matrix and take part in MaxSim scoring (query augmentation). Confirm both points against the pinned colbert-ai `QueryTokenizer` and query encoder before building the reference;
  - documents truncated to 180 tokens, and punctuation tokens (a skiplist built from `string.punctuation`) and padding removed from the document vectors;
  - each token vector projected by a 768-to-128 linear layer (no bias) and L2-normalized.
- **Tokenizer.** Uncased WordPiece; the existing `bert-wordpiece-uncased` profile very likely fits. Confirm by loading the pinned `tokenizer.json` and comparing against the reference on the route's corpus; if confirmed, no tokenizer work is needed.
- **No remote code.** The checkpoint's architecture is `HF_ColBERT`, which lives in the colbert-ai package, not transformers. The reference loads the built-in `BertModel` from the pinned weights and applies the projection layer read from the same safetensors file, in code in this repository. The colbert-ai package is used only as the documented behavior to reproduce (cited, not imported), unless the maintainer decides otherwise.
- **Upstream ONNX.** The repository ships a `model.onnx`. As with #101, it may be used only after it is compared against the Python reference; the default is our own export (backbone plus projection, output `[batch, seq, 128]`).
- **Scope.** Encoding and exact MaxSim scoring only. Indexing, PLAID compression (`nbits`), centroids and approximate search are out of scope; they belong to the vector store. CPU only; no GPU budget needed.

## API options

**A. Extend `ITextEmbedder`** with a multi-vector method. Rejected: it changes a shared interface every single-vector embedder implements, and `Dimension` would mean something else.

**B. `IEmbeddingGenerator` with one `Embedding<float>` per token.** Rejected: loses the text-to-vectors grouping and the query/document distinction, and Microsoft.Extensions.AI has no multi-vector type to map to.

**C. A new typed interface in `Starlings.Models.Abstractions`** (no Microsoft.Extensions.AI dependency, as the package has none today):
- `IMultiVectorEmbedder`: `Dimension` (128), `MaxQueryTokens`, `MaxDocumentTokens`, `EmbedQueries(texts)` and `EmbedDocuments(texts)`, each returning one `MultiVectorEmbedding` per text, plus token counting and validation in the style of `ITextEmbedder`.
- `MultiVectorEmbedding`: a token count and one contiguous `float[]` of `tokens * dimension` values (row-major), so callers can store it without copying per token.
- `MaxSim.Score(query, document)` and a batch form over many documents: a static, allocation-free scorer, exact float32, documented summation order so results are reproducible.
- An `IReranker` adapter over the two, so ColBERT can be dropped in wherever a reranker is taken today.

## Recommendation

Option C, with a new route `colbert-onnx-cpu-v1` and a library `Starlings.Models.ColBert`. Keeping query and document encoding as separate methods makes the asymmetric processing (markers, mask padding, skiplist) impossible to get wrong at the call site. The flat `float[]` layout keeps the door open to a span-based storage API later without breaking callers.

Parity is per token vector (after projection and normalization) and per MaxSim score, on a corpus with punctuation-heavy documents, queries shorter and longer than 32 tokens, and documents beyond 180 tokens. Thresholds are maintainer-owned.

## Checklist (one PR each)

1. Maintainer decision on the API shape (option C) and the route name. Pin `colbert-ir/colbertv2.0` and the colbert-ai source commit used as the behavior reference by full commit SHA; verify the marker ids, mask padding and skiplist against that source and update this note.
2. `Starlings.Models.Abstractions`: `IMultiVectorEmbedder`, `MultiVectorEmbedding`, the MaxSim scorer with unit tests (hand-computed scores, empty and single-token edge cases), capability keys. No model yet.
3. `model-pipeline`: reference runner (built-in `BertModel` plus projection, in-repo code), ONNX export of backbone plus projection, and a check of the upstream `model.onnx` against it; a token-level parity evaluator; corpus with its source recorded.
4. `Starlings.Models.ColBert`: asset lock, query and document preprocessing over `bert-wordpiece-uncased`, encoding, `IMultiVectorEmbedder` and the `IReranker` adapter, capability statuses.
5. The `colbert-onnx-cpu-v1` plugin, candidate profile and recipe; onboarding run on CPU, evidence under `docs/evidence/`, profile Experimental. The maintainer adds tolerances and the registry manifest.
6. Later, separately: a Microsoft.Extensions.AI adapter if one becomes expressible; other late-interaction checkpoints on the same route.
