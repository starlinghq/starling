# Samples

Console programs (and one Blazor WebAssembly app) that use the published `Starlings.*` packages from nuget.org. Each sample has its own README with its commands and options; some also show recorded output.

| Sample | What it shows | Models | Packages (`PackageReference`) | Version | CPU / CUDA |
| --- | --- | --- | --- | --- | --- |
| [E5Search](E5Search/README.md) | Multilingual semantic search; typed API and `IEmbeddingGenerator` give the same vectors | multilingual-e5-small | `Starlings.Models.E5.Extensions.AI` | 0.2.0 | CPU. CUDA: swap the natives and set the provider in `Program.cs` (no flag) |
| [E5BrowserSearch](E5BrowserSearch/README.md) | Multilingual semantic search entirely in the browser (Blazor WebAssembly): download with SHA-256 check and Cache Storage, `IEmbeddingGenerator` with e5 prefixes | multilingual-e5-small (fp32 from the Hub, or self-hosted int8-embed) | `Starlings.Models.E5.Browser`, `Starlings.Runtime.OnnxWeb.WebGpu` (optional) | 0.3.0 (follows the repository version; the packages are first published with 0.4.0) | Browser: WebAssembly (CPU). WebGPU optional, Experimental |
| [E5VectorStore](E5VectorStore/README.md) | e5 as the embedding generator of an in-memory `Microsoft.Extensions.VectorData` collection | multilingual-e5-small | `Starlings.Models.E5.Extensions.AI`, `CommunityToolkit.VectorData.InMemory` 1.0.1 | 0.1.0-preview.6 | CPU only |
| [EmbeddingGemma2Search](EmbeddingGemma2Search/README.md) | Text search with Matryoshka 256, text-to-image and text-to-video search, `IEmbeddingGenerator` for text and `DataContent` | EmbeddingGemma 2 (text, vision, multimodal text graphs) | `Starlings.Models.EmbeddingGemma2.Extensions.AI`, `.Imaging.SkiaSharp`, `.Video.FFmpeg` | 0.2.0 | CPU. `videos --cuda` with the CUDA natives |
| [Gliner2Ner](Gliner2Ner/README.md) | Zero-shot entity extraction with chosen labels and threshold; over-long input rejected | GLiNER2 base v1 | `Starlings.Models.Gliner2` | 0.2.0 | CPU. CUDA: swap the natives and set the provider in `Program.cs` (no flag) |
| [LocalRag](LocalRag/README.md) | Paragraph retrieval over a folder of `.md` and `.txt` files, answered by a local chat model, all through Microsoft.Extensions.AI | EmbeddingGemma 2 (text), Qwen3.5-0.8B fp32x | `Starlings.Models.EmbeddingGemma2.Extensions.AI`, `Starlings.Models.Qwen35.Extensions.AI` | 0.1.0-preview.6 | CPU only |
| [QwenChat](QwenChat/README.md) | Streaming console chat and one-shot `ask`, typed API or `IChatClient` | Qwen3.5-0.8B fp32x or fp16x | `Starlings.Models.Qwen35`, `Starlings.Models.Qwen35.Extensions.AI` | 0.2.0 | CPU (fp32x). `--cuda` with the GenAI CUDA natives (fp32x or fp16x) |
| [TicketTriage](TicketTriage/README.md) | Ticket router trained on English and used in five languages: e5 features in an ML.NET pipeline, GLiNER2 as a `CustomMapping` | multilingual-e5-small, GLiNER2 base v1 | `Starlings.Models.E5.MLNet`, `Starlings.Models.Gliner2` | 0.2.0 | CPU only |

"Version" is the `StarlingsVersion` property of the sample's project file. Every `Starlings.*` reference in one application must have the same version. The `QwenChat` sample uses `Starlings.Models.Qwen35`; it is unrelated to the experimental `Starlings.Models.QwenChat` package, which is not published.

## Common prerequisites

- The .NET 10 SDK. Every sample targets `net10.0`.
- For each runtime an application uses, exactly one native package, referenced by the sample itself: `Microsoft.ML.OnnxRuntime` 1.30.0 (CPU) for the ONNX models, `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 (CPU) for Qwen3.5 (`LocalRag` uses both runtimes). For CUDA, replace the CPU package with `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 or `Microsoft.ML.OnnxRuntimeGenAI.Cuda` 0.17.1; never reference the CPU and CUDA package of one runtime together. CUDA means Linux x64 with the CUDA 13 runtime on the loader path.
- Network access for the `download` command only. Every other command loads verified local files and never downloads. Files go to `--cache <dir>`, else `$STARLING_MODELS`, else the user cache directory (`~/.cache/starlings/models` on Linux). No token is needed.
- Disk for the models (from the packages' asset locks): e5 about 487 MB, GLiNER2 about 840 MB, EmbeddingGemma 2 text about 1.1 GB and 2.9 GB with the image assets, Qwen3.5 fp32x about 3.1 GB and fp16x about 1.6 GB.
- For `EmbeddingGemma2Search videos`: your own LGPL FFmpeg 8 shared libraries (none are shipped).

## How the samples build

The samples are package consumers, not part of the repository build: `samples/Directory.Build.props` and `samples/Directory.Build.targets` stop the repository's central package management, locked restore, analyzer and packing settings from applying, so a sample builds the same inside the checkout and in a copy anywhere else. Each sample states its package versions in its own project file. `E5VectorStore`, `LocalRag` and `TicketTriage` also restore with a `packages.lock.json`; check it with `dotnet restore --locked-mode`.

`tools/release-drill-ai.sh` packs the packages into a local feed and builds `E5Search`, `Gliner2Ner`, `QwenChat` and `EmbeddingGemma2Search` from a copy outside the repository; it fails if their `StarlingsVersion` differs from the version it packs.

Maturity belongs to model profiles, not to samples or packages. All 13 registry profiles are Stable; Stable on CUDA means Linux x64 with an NVIDIA GeForce RTX 3090. See the [catalog](../../docs/catalog/README.md) and the [compatibility matrix](../../docs/catalog/compatibility-matrix.md).
