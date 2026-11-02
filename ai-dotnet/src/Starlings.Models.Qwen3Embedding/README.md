# Starlings.Models.Qwen3Embedding

**Experimental.** [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) text embeddings and [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) query-document reranking on ONNX Runtime CPU (fp32), with owned tokenization from `Starlings.Tokenizers` (profiles `qwen3-bpe-embedding` and `qwen3-bpe-chat`). Both checkpoints are a Qwen3 decoder used non-autoregressively (hidden 1024, 28 layers); design: starlinghq/starling#144.

There is no registry manifest, onboarding evidence or hosted graph for these profiles yet, so nothing here is Preview or Stable, and there is no download step.

## Profiles

| Profile | Upstream | Output |
| --- | --- | --- |
| `Qwen3EmbeddingProfile.Embedding06B` (`qwen3-embedding-0.6b`) | `Qwen/Qwen3-Embedding-0.6B` at `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | hidden state at the last token (`<\|endoftext\|>`, appended by the tokenizer), L2-normalized, 1024 floats |
| `Qwen3EmbeddingProfile.Reranker06B` (`qwen3-reranker-0.6b`) | `Qwen/Qwen3-Reranker-0.6B` at `e61197ed45024b0ed8a2d74b80b4d909f1255473` | raw log-odds `logit[yes=9693] - logit[no=2152]` at the last position (sentence-transformers `LogitScore`); `Qwen3Reranker.Sigmoid` gives the card's `P(yes)` |

Each profile pins the size and SHA-256 of every file of the released directory: `model.onnx` and `model.onnx.data` (about 2.4 GB of external weights) from the export recipe, and the upstream `tokenizer.json`, `config.json`, `tokenizer_config.json` and `config_sentence_transformers.json` or `chat_template.jinja`. A directory is verified before a session is created.

## Building the graphs

The graphs are built locally, on CPU, with the hash-locked recipe in [`model-pipeline/reference/qwen3-embedding/`](https://github.com/starlinghq/starling/tree/main/model-pipeline/reference/qwen3-embedding) (`requirements.txt`), from a snapshot of the pinned upstream revision:

```sh
python export_qwen3_onnx.py --model embedding --model-dir <Qwen/Qwen3-Embedding-0.6B@97b0c614> --out <export>
python release_qwen3_onnx.py --model embedding --model-dir <same snapshot> --export <export>/model.onnx --out <release>
python export_qwen3_onnx.py --model reranker --model-dir <Qwen/Qwen3-Reranker-0.6B@e61197ed> --out <export>
python release_qwen3_onnx.py --model reranker --model-dir <same snapshot> --export <export>/model.onnx --out <release>
```

Two independent exports give the same bytes, which are the digests the profiles pin.

## Use

```csharp
using Starlings.Models.Qwen3Embedding;

using var embedder = Qwen3TextEmbedder.Create("/models/qwen3-embedding-0.6b-release");
float[] query = embedder.Embed("What is the capital of China?", prompt: "query");
float[] document = embedder.Embed("The capital of China is Beijing.");   // prompt "none"
float cosine = query.Zip(document, (a, b) => a * b).Sum();                // both are L2-normalized

using var reranker = Qwen3Reranker.Create("/models/qwen3-reranker-0.6b-release");
float[] scores = reranker.Score("What is the capital of China?", ["The capital of China is Beijing.", "Gravity is a force."]);
```

- **Prompts.** `query` is the upstream instruction `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"` (no trailing space); `none` and `document` are empty. `Qwen3TextEmbedder.FormatQuery(task, query)` writes a query for a task of your own.
- **Reranker template.** `prefix + "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}" + suffix`, each part tokenized without special tokens, as the model card's transformers snippet. The instruction defaults to `Qwen3EmbeddingProfile.DefaultRerankInstruction`.
- **Length.** 8192 tokens per input. Longer text is truncated from the right: the embedding keeps the appended `<|endoftext|>`, the reranker truncates only the instruction, query and document text. Attention memory grows with the square of the length; `Qwen3EmbeddingOptions.MaxBatchTokens` (default 16,384 padded cells) keeps long rows in small calls.
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
| Matryoshka (MRL) dimensions below 1024 | untested (rejected) |
| CUDA | untested (CPU sessions only) |
| Download of the graphs | unsupported until the graphs are hosted |
| fp16/bf16 graphs, the 4B and 8B checkpoints, more than 8192 tokens | unsupported |

`Qwen3EmbeddingCapabilities` carries the same statuses in code.

## Evidence so far

- The embedding graph reproduces the model card's four query-document cosines within 4e-7 (published anchor, 1e-3 bound) and sentence-transformers `encode` within 2.1e-7, on CPU (recipe `verify_qwen3_onnx.py`).
- This package's tests compare, when a released directory is available, the embeddings and reranker scores against PyTorch fp32 references for the card's examples and a few edge cases, and check that padded batches equal single rows.
- The tokenizer profiles are exact on 137/137 conformance cases in starlinghq/tokenizers.

Weights: Apache-2.0 (Qwen). Template attribution: `THIRD-PARTY-NOTICES.md`.
