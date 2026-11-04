# Starlings.Models.Qwen3Embedding

**Experimental.** [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) text embeddings and [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) query-document reranking on ONNX Runtime CPU (fp32), with owned tokenization from `Starlings.Tokenizers` (profiles `qwen3-bpe-embedding` and `qwen3-bpe-chat`). Both checkpoints are a Qwen3 decoder used non-autoregressively (hidden 1024, 28 layers); design: starlinghq/starling#144.

⭐ If this package is useful, [star Starling on GitHub](https://github.com/starlinghq/starling).

The graphs are hosted publicly on Hugging Face (`Starlinghq/qwen3-embedding-0.6b-onnx`, `Starlinghq/qwen3-reranker-0.6b-onnx`), pinned by commit, size and SHA-256. Nothing here is Preview or Stable.

<!-- catalog:maturity profiles=qwen3-embedding-0.6b/*,qwen3-reranker-0.6b/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `qwen3-embedding-0.6b/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: unsupported; windows-x64 cpu: untested |
<!-- /catalog:maturity -->

## Profiles

| Profile | Upstream | Output |
| --- | --- | --- |
| `Qwen3EmbeddingProfile.Embedding06B` (`qwen3-embedding-0.6b`) | `Qwen/Qwen3-Embedding-0.6B` at `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | hidden state at the last token (`<\|endoftext\|>`, appended by the tokenizer), L2-normalized, 1024 floats |
| `Qwen3EmbeddingProfile.Reranker06B` (`qwen3-reranker-0.6b`) | `Qwen/Qwen3-Reranker-0.6B` at `e61197ed45024b0ed8a2d74b80b4d909f1255473` | raw log-odds `logit[yes=9693] - logit[no=2152]` at the last position (sentence-transformers `LogitScore`); `Qwen3Reranker.Sigmoid` gives the card's `P(yes)` |

Each profile pins the size and SHA-256 of every file of the released directory: `model.onnx` and `model.onnx.data` (about 2.4 GB of external weights) from the export recipe, and the upstream `tokenizer.json`, `config.json`, `tokenizer_config.json` and `config_sentence_transformers.json` or `chat_template.jinja`. A directory is verified before a session is created.

## Getting the files

```csharp
using Starlings.Models.Qwen3Embedding;

// Setup step, once (about 2.4 GB per profile; public, no token needed). Inference never downloads.
var embeddingFiles = await Qwen3EmbeddingModelAssets.DownloadAsync(Qwen3EmbeddingProfile.Embedding06B);
var rerankerFiles = await Qwen3EmbeddingModelAssets.DownloadAsync(Qwen3EmbeddingProfile.Reranker06B);
// At startup, offline: Qwen3EmbeddingModelAssets.OpenLocal(profile) returns the verified cached snapshot.
```

| Profile | Hosting repository | Commit |
| --- | --- | --- |
| `qwen3-embedding-0.6b` | [`Starlinghq/qwen3-embedding-0.6b-onnx`](https://huggingface.co/Starlinghq/qwen3-embedding-0.6b-onnx) | `0585005027acc7232133808914cd8bc2a4bd1ea8` |
| `qwen3-reranker-0.6b` | [`Starlinghq/qwen3-reranker-0.6b-onnx`](https://huggingface.co/Starlinghq/qwen3-reranker-0.6b-onnx) | `0f951df2baec8e9ac7f884f349d2820a34aa85e5` |

The hosted graphs are Starling's export of the pinned checkpoints with the hash-locked recipe in [`model-pipeline/reference/qwen3-embedding/`](https://github.com/starlinghq/starling/tree/main/model-pipeline/reference/qwen3-embedding) (`requirements.txt`, CPU, fp32). The recipe reproduces them byte for byte, so a directory built locally opens the same way:

```sh
python export_qwen3_onnx.py --model embedding --model-dir <Qwen/Qwen3-Embedding-0.6B@97b0c614> --out <export>
python release_qwen3_onnx.py --model embedding --model-dir <same snapshot> --export <export>/model.onnx --out <release>
python export_qwen3_onnx.py --model reranker --model-dir <Qwen/Qwen3-Reranker-0.6B@e61197ed> --out <export>
python release_qwen3_onnx.py --model reranker --model-dir <same snapshot> --export <export>/model.onnx --out <release>
```

## Use

```csharp
using Starlings.Models.Qwen3Embedding;

using var embedder = Qwen3TextEmbedder.Create(embeddingFiles.Directory);   // or a directory built with the recipe
float[] query = embedder.Embed("What is the capital of China?", prompt: "query");
float[] document = embedder.Embed("The capital of China is Beijing.");   // prompt "none"
float cosine = query.Zip(document, (a, b) => a * b).Sum();                // both are L2-normalized

using var reranker = Qwen3Reranker.Create(rerankerFiles.Directory);
float[] scores = reranker.Score("What is the capital of China?", ["The capital of China is Beijing.", "Gravity is a force."]);
```

- **Prompts.** `query` is the upstream instruction `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"` (no trailing space); `none` and `document` are empty. `Qwen3TextEmbedder.FormatQuery(task, query)` writes a query for a task of your own. **Warning:** the `ITextEmbedder` members (`EmbedBatch(texts, tokenCounts)`) embed text exactly as given (prompt `none`), and `Qwen3EmbeddingGenerator` uses `none` unless a prompt is set, so a generic retrieval consumer that passes raw queries gets them embedded without the instruction and lower retrieval quality, with no error. Format queries with `FormatQuery`, or use `prompt: "query"`.
- **Reranker template.** `prefix + "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}" + suffix`, each part tokenized without special tokens, as the model card's transformers snippet. The instruction defaults to `Qwen3EmbeddingProfile.DefaultRerankInstruction`.
- **Length.** 8192 tokens per input. Longer text is truncated from the right: the embedding keeps the appended `<|endoftext|>`, the reranker keeps the template prefix and suffix and cuts the content (`<Instruct>`, `<Query>`, `<Document>` as one sequence) from the right, so the document tail goes first and a very long query can cut away the whole document. The graph uses eager attention, so memory grows with the square of the length: one 8192-token row needs roughly 4.3 GB of attention scores per layer. `Qwen3EmbeddingOptions.MaxBatchTokens` (default 8192 padded cells) makes a maximum-length row run alone; a budget of 16,384 lets two such rows share a call and roughly doubles that.
- **Batching.** Rows are right-padded with `<|endoftext|>` (151643) and read at `sum(mask) - 1`, in input order, `MaxBatchSize` rows (default 8) per call.
- **Microsoft.Extensions.AI.** `Starlings.Models.Qwen3Embedding.Extensions.AI` adds `Qwen3EmbeddingGenerator`, an `IEmbeddingGenerator<string, Embedding<float>>`.

The application references the ONNX Runtime native package (`Microsoft.ML.OnnxRuntime`).

## Capabilities

| Capability | Status |
| --- | --- |
| Embeddings, reranking (ONNX Runtime CPU, fp32) | experimental |
| Named prompts and custom instructions | experimental |
| Batching; inputs up to 8192 tokens (right truncation) | experimental |
| `IEmbeddingGenerator` (Extensions.AI package) | experimental |
| Matryoshka (MRL) dimensions below 1024 | unsupported (rejected) |
| CUDA | unsupported (CPU sessions only) |
| Download of the released directories (`Qwen3EmbeddingModelAssets`) | experimental |
| fp16/bf16 graphs, the 4B and 8B checkpoints, more than 8192 tokens | unsupported |

`Qwen3EmbeddingCapabilities` carries the same statuses in code.

## Evidence

The `qwen3-embedding-onnx-cpu-v1` onboarding route ran for real on CPU on 2026-11-02 ([`docs/evidence/qwen3-embedding-onboarding-2026-11-02/`](https://github.com/starlinghq/starling/tree/main/docs/evidence/qwen3-embedding-onboarding-2026-11-02)). Both runs rebuilt the graphs with the recipe and found them byte-identical to the hosted files.

- **Embedding: passed** (558 checks). .NET embeddings from text and from ids within 6.0e-7 of PyTorch fp32 (bound 1e-4), the card's four query-document cosines within 3.6e-7 of the published values, token ids equal on all 62 cases. Registry profile `qwen3-embedding-0.6b/onnx-fp32-cpu`, Experimental.
- **Reranker: failed on one bound.** .NET scores within 2.4e-5 of PyTorch (bound 1e-3) and token ids equal on all 56 cases, but the graph's final hidden state differs from PyTorch by up to 3.7e-3 on two long inputs (2,058 and 6,649 tokens), above the predeclared 2e-3. The bound was not changed; the reranker profile has no registry manifest until the maintainer decides it.
- The longest evaluated inputs have about 6,600 tokens, not the full 8,192.
- The tokenizer profiles are exact on 137/137 conformance cases in starlinghq/tokenizers.

Weights: Apache-2.0 (Qwen). Template attribution: `THIRD-PARTY-NOTICES.md`.
