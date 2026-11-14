# LocalRag sample

Answer a question over a folder of Markdown/text files entirely locally. EmbeddingGemma 2 retrieves paragraphs through `IEmbeddingGenerator<string, Embedding<float>>`; Qwen3.5 answers through `IChatClient`. The sample references only published packages and requires the .NET 10 SDK.

From this directory:

```sh
dotnet build -c Release
dotnet run -c Release --no-build -- download
dotnet run -c Release --no-build -- ask --docs docs "When is the Willow Bookshop open?"
```

`download` explicitly acquires and verifies the text embedding and Qwen fp32x assets (roughly 4 GB combined); it does not fetch image/video assets. `ask` loads verified local files and never downloads. Both commands accept `--cache <dir>`; otherwise they use `$STARLING_MODELS` or the user cache (`~/.cache/starlings/models` on Linux). `StarlingsVersion` pins the published packages at `0.5.0`. The sample uses `PackageReference`s only, no project references, and runs on CPU as written (`Microsoft.ML.OnnxRuntime` 1.30.0 and `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1; it has no CUDA option). `packages.lock.json` records the dependency graph; verify with `dotnet restore --locked-mode`.

The program recursively reads `.md` and `.txt` files. Markdown files split on ATX headings (`#` through `######`) so chunks do not span sections, then on blank lines within each section; fenced code blocks are never treated as headings. Non-Markdown files split on blank lines only. Each nonempty chunk is embedded using `EmbeddingGemma2Prompt.Document`. It uses `EmbeddingGemma2Prompt.Query` for the question, ranks by cosine similarity, and sends the top three paragraphs (or all if fewer exist) as numbered context. A single embedding generator handles both prompts. All embedding and chat inference uses Microsoft.Extensions.AI interfaces; Starling APIs configure and acquire the models. Both native packages are needed: ONNX Runtime for embeddings and ONNX Runtime GenAI for chat.

## Index cache

The first `ask` run embeds every chunk and writes the embeddings to `<docs>/.localrag-index.json`, next to the corpus. The file records a fingerprint of the corpus (a SHA-256 over each file's relative path, size, and last write time, in a stable order) plus the chunk texts and their vectors. On later runs the index is reloaded instead of re-embedding whenever the fingerprint matches and the recomputed chunks are identical; it prints `Loaded cached index ...` in that case, or `Embedded ... and saved the index ...` when it rebuilt. Any change to the set of files, their sizes, their timestamps, or their contents (which changes the chunks) invalidates the index and it is rebuilt from scratch. Delete `.localrag-index.json` to force a rebuild, and keep it out of version control — it is machine-generated cache data.

Actual stdout on Linux x64 CPU, .NET SDK 10.0.100 and Starlings 0.5.0, on 2026-11-04, using the included documents, an existing cache and no network interface:

```sh
dotnet run -c Release --no-build -- ask --docs docs "When is the Willow Bookshop open?"
```

```text
Indexed 4 paragraphs.
[1] bookshop.md paragraph 1 (cosine 0.9016)
[2] travel.txt paragraph 1 (cosine 0.7852)
[3] bookshop.md paragraph 3 (cosine 0.7462)
Question: When is the Willow Bookshop open?
Answer: The Willow Bookshop is open Tuesday through Saturday from 10:00 to 18:00.

If this was useful, star https://github.com/starlinghq/starling
```

This is a small-folder demonstration: it keeps the text and scores in memory and re-embeds the corpus only when the persisted index is stale (see [Index cache](#index-cache)). Long paragraphs are not silently truncated: the embedding model rejects inputs over its 8,192-token limit, and Qwen rejects a combined context/question/output beyond its context limit. Split long paragraphs or use a smaller document collection if those limits are reached. Generation is greedy with at most 256 output tokens. Scores and generated wording can vary across hardware.

The prompt requests source citations and an admission when context is insufficient; the small chat model may omit citations (as in this run) or give incorrect answers. Check the printed source paragraphs before relying on an answer. Invalid arguments return exit code 2; missing/empty document folders, missing models and inference errors return 1. Platform support is described in the [compatibility matrix](../../../docs/catalog/compatibility-matrix.md); this sample run covers Linux x64 CPU only.
