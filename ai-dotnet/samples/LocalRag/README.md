# LocalRag sample

Answer a question over a folder of Markdown/text files entirely locally. EmbeddingGemma 2 retrieves paragraphs through `IEmbeddingGenerator<string, Embedding<float>>`; Qwen3.5 answers through `IChatClient`. The sample references only published packages and requires the .NET 10 SDK.

From this directory:

```sh
dotnet build -c Release
dotnet run -c Release --no-build -- download
dotnet run -c Release --no-build -- ask --docs docs "When is the Willow Bookshop open?"
```

`download` explicitly acquires and verifies the text embedding and Qwen fp32x assets (roughly 4 GB combined); it does not fetch image/video assets. `ask` loads verified local files and never downloads. Both commands accept `--cache <dir>`; otherwise they use `$STARLING_MODELS` or the user cache (`~/.cache/starlings/models` on Linux). `StarlingsVersion` pins the published packages; it is `0.1.0-preview.6`, the version this sample was written and run with, not the current `0.2.0`. The sample uses `PackageReference`s only, no project references, and runs on CPU as written (`Microsoft.ML.OnnxRuntime` 1.30.0 and `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1; it has no CUDA option). `packages.lock.json` records the dependency graph; verify with `dotnet restore --locked-mode`.

The program recursively reads `.md` and `.txt` files, splits on blank lines, and embeds each nonempty paragraph using `EmbeddingGemma2Prompt.Document`. It uses `EmbeddingGemma2Prompt.Query` for the question, ranks by cosine similarity, and sends the top three paragraphs (or all if fewer exist) as numbered context. A single embedding generator handles both prompts. All embedding and chat inference uses Microsoft.Extensions.AI interfaces; Starling APIs configure and acquire the models. Both native packages are needed: ONNX Runtime for embeddings and ONNX Runtime GenAI for chat.

Actual stdout on Linux x64 CPU, .NET SDK 10.0.100, runtime 10.0.12 and Starlings 0.1.0-preview.6, on 2026-10-17, using the included documents and an existing cache:

```sh
dotnet run -c Release --no-build -- ask --docs docs --cache ~/.cache/starling-models "When is the Willow Bookshop open?"
```

```text
Indexed 4 paragraphs.
[1] bookshop.md paragraph 1 (cosine 0.9016)
[2] travel.txt paragraph 1 (cosine 0.7852)
[3] bookshop.md paragraph 3 (cosine 0.7462)
Question: When is the Willow Bookshop open?
Answer: The Willow Bookshop is open Tuesday through Saturday from 10:00 to 18:00.
```

This is a small-folder demonstration: it re-embeds every paragraph on each question and keeps the text and scores in memory. It does not parse Markdown structure or persist an index. Long paragraphs are not silently truncated: the embedding model rejects inputs over its 8,192-token limit, and Qwen rejects a combined context/question/output beyond its context limit. Split long paragraphs or use a smaller document collection if those limits are reached. Generation is greedy with at most 256 output tokens. Scores and generated wording can vary across hardware.

The prompt requests source citations and an admission when context is insufficient; the small chat model may omit citations (as in this run) or give incorrect answers. Check the printed source paragraphs before relying on an answer. Invalid arguments return exit code 2; missing/empty document folders, missing models and inference errors return 1. Platform support is described in the [compatibility matrix](../../../docs/catalog/compatibility-matrix.md); this sample run covers Linux x64 CPU only.
