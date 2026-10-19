# ai-dotnet release 0.3.0

**Kev and QwenChat reach nuget.org.** This release follows `0.2.0` (published 2026-10-17) and carries the work merged since (#116 to #127). It adds two packages to the publishable set, `Starlings.Models.Kev` and `Starlings.Models.QwenChat`, both with **Experimental** profiles only, and hosts their model files publicly. It also adds opt-in batching to the embedding and extraction packages and makes the stop-string matcher public (#124).

**Publishing.** The `release` workflow publishes this release from the tag `ai-v0.3.0` on `main`, after a maintainer approves the `release` environment: the 17 packages of the publishable set below, through NuGet trusted publishing, plus a GitHub release with this record as notes.

**Tokenizer.** Unchanged: every package that tokenizes pins `Starlings.Tokenizers [0.3.0]`.

**Maturity.** Maturity belongs to profiles, not packages (each package README has a generated maturity table). Every Stable profile of 0.2.0 is still Stable, with the same contract; no gate, tolerance or `protected/` file changed. The new profiles are Experimental: no gate has run on them, and their capabilities are at most Experimental.

| Profile | Maturity | Admitted |
| --- | --- | --- |
| `kev-0.8b/onnx-fp32-cpu`, `kev-0.8b/onnx-fp32-cuda` | Experimental | `model-pipeline/admissions/kev.md`, CUDA profiles in #116 |
| `kev-4b/onnx-fp32-cpu`, `kev-4b/onnx-fp32-cuda` | Experimental | `model-pipeline/admissions/kev.md`, CUDA profiles in #116 |
| `qwen3-0.6b/ortgenai-fp32-cpu` | Experimental | `model-pipeline/admissions/qwen-chat.md` (#127) |
| `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu` | Experimental | `model-pipeline/admissions/qwen-chat.md` (#127) |

The Stable profiles are listed in [`ai-0.2.0.md`](ai-0.2.0.md). The [catalog](../catalog/README.md) links every gate to its evidence.

## Packages

All packages target `net10.0`, are licensed MIT OR Apache-2.0 and have SourceLink to the commit. Sibling and tokenizer dependencies are exact (`[version]`).

| Package | Version | Change since 0.2.0 |
| --- | --- | --- |
| `Starlings.Tokenizers` | 0.3.0 | dependency, unchanged |
| `Starlings.Runtime.Onnx` | 0.3.0 | `Starlings.Runtime.Onnx.Batching.MicroBatcher` and `MicroBatchingOptions`: opt-in cross-request micro-batching, Experimental; the batcher owns cancellation, so one caller cancelling never cancels a shared run (#126) |
| `Starlings.Acquisition` | 0.3.0 | README maturity table lists the Kev and QwenChat profiles (#116, #127) |
| `Starlings.Models.E5` | 0.3.0 | opt-in `SortByLength` and `MaxBatchTokens` for explicit batches, results in input order, Experimental (#126); defaults unchanged |
| `Starlings.Models.E5.Extensions.AI` | 0.3.0 | opt-in `MicroBatching` on the `IEmbeddingGenerator`, Experimental (#126) |
| `Starlings.Models.E5.MLNet` | 0.3.0 | documentation only (#123) |
| `Starlings.Models.Gliner2` | 0.3.0 | opt-in `SortByLength`, `MaxBatchTokens` and `MicroBatching` on `ExtractAsync` (only requests with the same labels and descriptions share a run), Experimental (#126) |
| `Starlings.Generation.OnnxGenAI` | 0.3.0 | `StopSequenceMatcher` is public, with argument validation and a defensive copy of the stops; `TextGenerator` behaves as before (#125) |
| `Starlings.Models.Qwen35` | 0.3.0 | documentation only (#123) |
| `Starlings.Models.Qwen35.Extensions.AI` | 0.3.0 | documentation only (#123) |
| `Starlings.Models.EmbeddingGemma2` | 0.3.0 | opt-in `SortByLength` and `MaxBatchTokens` for explicit text batches (encoder and multimodal text options), Experimental (#126); defaults unchanged |
| `Starlings.Models.EmbeddingGemma2.Extensions.AI` | 0.3.0 | opt-in `MicroBatching` on the text `IEmbeddingGenerator`, Experimental (#126) |
| `Starlings.Models.EmbeddingGemma2.MLNet` | 0.3.0 | documentation only (#123) |
| `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` | 0.3.0 | documentation only (#123) |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` | 0.3.0 | documentation only (#123) |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` | 0.3.0 | documentation only (#123) |
| `Starlings.Models.Kev` | 0.3.0 | **first publish**: Kev 1.0 decision models (Kev-0.8B, Kev-4B), CPU and CUDA profiles, explicit download from public hosting (`KevModelAssets.DownloadAsync`), Experimental (#91, #116); internal batched-rows spike, `batched_rows` stays Unsupported (#126) |
| `Starlings.Models.QwenChat` | 0.3.0 | **first publish**: Qwen3-0.6B and Qwen2.5-0.5B-Instruct text chat on ONNX Runtime GenAI 0.17.1, CPU fp32, with the bundle lock pinned in code and explicit download from public hosting (`QwenChatModelAssets.DownloadAsync`), Experimental (#127) |

**Publishable set (17 packages):** every row at 0.3.0. Nothing is held (`HELD` is empty in `.github/workflows/release.yml`).

**Not published:** `Starlings.Models.Bert` (the BERT route: bge and nomic embeddings, #120, #121, batching options #126) builds and tests on `main` but is not in the release drill's package set.

ML.NET under Native AOT is unsupported (Microsoft.ML emits code at run time).

## Upgrading from 0.2.0

- Raise every `Starlings.*` reference to `0.3.0` together.
- No default changed: batching options are off unless set, and Stable profiles produce the same outputs.
- Kev on Linux needs `MALLOC_MMAP_THRESHOLD_=65536` in the process environment for bounded memory in long-running processes (package README), as GLiNER2 and EmbeddingGemma 2 video do.
- QwenChat: the application adds `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1. `QwenChatOptions.MaxNewTokens` defaults to 16 (the corpus setting); set your own.

## Hosting

The files Starling builds are downloadable without a token, each pinned by commit and by SHA-256 in the packages' locks. New since 0.2.0:

| Repository | Revision | Files |
| --- | --- | --- |
| `Starlinghq/kev-0.8b-onnx` | `8a9fe0175d294a89c6fa963a2245eb32e3fa1505` | Kev-0.8B graph, pointer head, tokenizer |
| `Starlinghq/kev-4b-onnx` | `8179da507de7de4bddfdbfa44894ff24b1dcd959` | Kev-4B graph, pointer head, tokenizer |
| `Starlinghq/qwen3-0.6b-ortgenai` | `c24cd59cad1444863ddfdb372736da610a09c9c4` | Qwen3-0.6B GenAI build, tokenizer, chat template |
| `Starlinghq/qwen2.5-0.5b-instruct-ortgenai` | `6b6957c8305ef950450da2c0c846bce6fac3e8a0` | Qwen2.5-0.5B-Instruct GenAI build, tokenizer, chat template |

Unchanged since 0.2.0: `Starlinghq/gliner2-base-v1-onnx`, `Starlinghq/embeddinggemma-2-onnx`, `Starlinghq/embeddinggemma-2-vision-onnx` and `Starlinghq/qwen3.5-0.8b-ortgenai` at the revisions in [`ai-0.2.0.md`](ai-0.2.0.md#hosting).

## Evidence

- Release drill: the `release` workflow's candidate job (`tools/release-drill-ai.sh --no-model`) packs every package from a clean clone, checks each nuspec and the per-profile maturity statements, and installs the packages into clean consumers outside the checkout.
- Kev: `docs/evidence/kev-cuda-2026-10-17/` and the admission record. QwenChat: `docs/evidence/qwenchat-onboarding-2026-10-18/` (development parity 14/14 per profile and an offline clean-consumer drill). Batching: `docs/evidence/batching-sweep-2026-10-18/`.
- Gates: `docs/catalog/README.md` (generated), with links to `docs/evidence/`.
- Regression: the qualification workflow on the release tag (CPU release gates and soaks; `cuda (runpod)` on the RTX 3090).
