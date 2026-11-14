# Starling local RAG console app

Answer a question over a folder of Markdown/text files entirely locally. EmbeddingGemma 2 retrieves paragraphs through `IEmbeddingGenerator<string, Embedding<float>>`; Qwen3.5 answers through `IChatClient`. No API key, no Python sidecar — only the published [Starlings](https://www.nuget.org/packages?q=Starlings) packages and the .NET 10 SDK.

## Run it

```sh
dotnet run -- download
dotnet run -- ask --docs docs "When is the Willow Bookshop open?"
```

`download` acquires and verifies the text embedding and Qwen3.5 fp32x assets (roughly 4 GB combined) into the local model cache (`$STARLING_MODELS` or `~/.cache/starlings/models` on Linux; override with `--cache <dir>`). `ask` loads verified local files and never downloads.

`ask` recursively reads `.md` and `.txt` files, splits them into paragraphs (ATX headings for Markdown), embeds each with the document prompt prefix, ranks by cosine similarity against the query prompt, and asks Qwen3.5 for a cited answer over the top three paragraphs. Embeddings are cached in `<docs>/.localrag-index.json` and reused while the corpus is unchanged; delete that file to force a rebuild, and keep it out of version control.

On CUDA (`--NativeRuntime cuda`, or edit the `#if` block in the `.csproj`), the app references `Microsoft.ML.OnnxRuntime.Gpu` and `Microsoft.ML.OnnxRuntimeGenAI.Cuda` instead of the CPU natives — exactly one native runtime either way.

## Prompt prefixes

EmbeddingGemma 2 needs different prompt prefixes for documents and queries; getting them wrong silently hurts retrieval quality. This template uses `EmbeddingGemma2EmbeddingGenerator.PromptPropertyName` with `EmbeddingGemma2Prompt.Document` for paragraph embeddings and `EmbeddingGemma2Prompt.Query` (the generator default) for the question — one generator, both prompt types.

## Model choice

If you prefer [e5](https://huggingface.co/intfloat/multilingual-e5-small) (`Starlings.Models.E5.Extensions.AI`, `E5Prompt.Passage`/`E5Prompt.Query`) swap the embedding package and call sites; the structure is otherwise identical. The sample [`ai-dotnet/samples/LocalRag`](https://github.com/starlinghq/starling/tree/main/ai-dotnet/samples/LocalRag) shows the full pattern, including the persisted index cache.

The packages are pinned at 0.5.0 (pass `-StarlingsVersion <version>` at scaffold time to change it). Both model families are on ONNX Runtime (embeddings) and ONNX Runtime GenAI (chat).
