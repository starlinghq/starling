# ai-dotnet release 0.2.0

**The first release with Stable profiles**, and the first ai-dotnet version without a preview suffix (`docs/kickoff.md`, "ai-dotnet package versions and maturity", 2026-10-15). It follows `0.1.0-preview.6` (published 2026-10-14) and carries the qualification work merged since (#16 to #87).

**Publishing.** The `release` workflow publishes this release from the tag `ai-v0.2.0` on `main`, after a maintainer approves the `release` environment: the 15 packages of the publishable set below, through NuGet trusted publishing, plus a GitHub release with this record as notes.

**Tokenizer.** Every package that tokenizes pins `Starlings.Tokenizers [0.3.0]`, released from starlinghq/tokenizers before this one. 0.3.0 is the code of `0.3.0-preview.1` released without a preview suffix (a stable package cannot depend on a prerelease one, NuGet NU5104); its conformance fixtures pass unchanged, so token ids, offsets and decoded text are those of the qualified `0.3.0-preview.1`. The manifests' `processing.tokenizer.version` is `0.3.0`.

**Maturity.** Maturity belongs to profiles, not packages: each package README has a maturity table generated from the registry manifests (`model-pipeline catalog`). A Stable profile passed every gate G0 to G7 on its P-030 release sets, with the maintainer recording each result (M10), and has regression support: weekly CPU qualification on hosted runners, and a GPU run on an NVIDIA GeForce RTX 3090 for every release tag (`.github/workflows/qualification.yml`, job `cuda (runpod)`). The packages stay at `0.x`: the public C# API may change between minor versions; a Stable profile's contract (`docs/support-policy.md`, the package README's "Stable contract") does not.

| Profile | Maturity | Promoted |
| --- | --- | --- |
| `gliner2-base-v1/onnx-fp32-cpu` | **Stable** | 2026-10-15 (#59) |
| `multilingual-e5-small/onnx-fp32-cpu` | **Stable** | 2026-10-16 (#68) |
| `embeddinggemma-2/onnx-fp32-cpu` | **Stable** | 2026-10-16 (#68) |
| `embeddinggemma-2/image-fp32-cpu` | **Stable** | 2026-10-16 (#75) |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | **Stable** | 2026-10-16 (#75) |
| `multilingual-e5-small/onnx-fp32-cuda` | **Stable** (RTX 3090) | 2026-10-16 (#82) |
| `gliner2-base-v1/onnx-fp32-cuda` | **Stable** (RTX 3090) | 2026-10-16 (#82) |
| `embeddinggemma-2/onnx-fp32-cuda` | **Stable** (RTX 3090) | 2026-10-16 (#82) |
| `embeddinggemma-2/image-fp32-cuda` | **Stable** (RTX 3090) | 2026-10-16 (#82) |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | **Stable** (RTX 3090) | 2026-10-16 (#82) |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | **Stable** (RTX 3090) | 2026-10-16 (#82) |
| `embeddinggemma-2/video-fp32-cpu` | Experimental | not promoted: hosted G2 on the video release set failed in the stable-flakes qualification run 38967975211 (first divergence `dotnet-ids`), undiagnosed |
| `embeddinggemma-2/video-fp32-cuda` | Experimental | not promoted: waits for a green per-release GPU run on the RTX 3090 for this profile |

Stable on CUDA means Linux x64 with an RTX 3090; other GPUs are untested. Windows CPU is experimental; macOS and ARM64 are untested. The [catalog](../catalog/README.md) links every gate to its evidence, and the [compatibility matrix](../catalog/compatibility-matrix.md) lists each platform and provider.

**Single release owner.** Stable promotions carry the recorded single-owner exception (`docs/support-policy.md`, M7): a second release owner is still open.

## Packages

All packages target `net10.0`, are licensed MIT OR Apache-2.0 and have SourceLink to the commit. Sibling and tokenizer dependencies are exact (`[version]`).

| Package | Version | Change since 0.1.0-preview.6 |
| --- | --- | --- |
| `Starlings.Tokenizers` | 0.3.0 | dependency, released from starlinghq/tokenizers; same code as 0.3.0-preview.1 |
| `Starlings.Runtime.Onnx` | 0.2.0 | a CUDA run is never terminated mid-graph (#61, GLiNER2 CUDA wrong output); heap trim on session release (#66); container-aware intra-op threads (#63); dedicated run threads (#86, #87) |
| `Starlings.Acquisition` | 0.2.0 | version only |
| `Starlings.Models.E5` | 0.2.0 | Stable CPU and CUDA profiles; dedicated run threads by default on CPU (#87) |
| `Starlings.Models.E5.Extensions.AI` | 0.2.0 | version only |
| `Starlings.Models.E5.MLNet` | 0.2.0 | version only |
| `Starlings.Models.Gliner2` | 0.2.0 | Stable CPU and CUDA profiles; per-token id memoization (#54) |
| `Starlings.Generation.OnnxGenAI` | 0.2.0 | version only |
| `Starlings.Models.Qwen35` | 0.2.0 | **tool calling**, experimental (#17): template rendering of tools and a parser for the tool-call format; Stable fp32x CPU, fp32x CUDA and fp16x CUDA profiles |
| `Starlings.Models.Qwen35.Extensions.AI` | 0.2.0 | `IChatClient` tool calling (`AIFunction` tools, `FunctionCallContent`, `FunctionResultContent`, `UseFunctionInvocation`) |
| `Starlings.Models.EmbeddingGemma2` | 0.2.0 | Stable text and image profiles (CPU and CUDA); video decoded as torchcodec does (#41); AVX2, row- and frame-parallel image preprocessing (#55); CPU sessions without ONNX Runtime's CPU arena (#83) |
| `Starlings.Models.EmbeddingGemma2.Extensions.AI` | 0.2.0 | version only |
| `Starlings.Models.EmbeddingGemma2.MLNet` | 0.2.0 | version only |
| `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` | 0.2.0 | version only |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` | 0.2.0 | **first publish** (D2 cleared, #48): FFmpeg 8 adapter, seeking as torchcodec does |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` | 0.2.0 | **first publish** (D2 cleared, #48): FFmpeg 9 adapter, bindings verified against the pinned n9.0.1 headers (#47) |

**Publishable set (15 packages):** every row at 0.2.0. Nothing is held (`HELD` is empty in `.github/workflows/release.yml`).

ML.NET under Native AOT is unsupported (Microsoft.ML emits code at run time); every other package publishes and runs under Native AOT (`docs/evidence/aot/`).

## Upgrading from 0.1.0-preview.6

- Raise every `Starlings.*` reference to `0.2.0` together; `--prerelease` is no longer needed.
- `Starlings.Tokenizers` moves from `0.3.0-preview.1` to `0.3.0` transitively; outputs are unchanged.
- GLiNER2 and EmbeddingGemma 2 video on Linux need `MALLOC_MMAP_THRESHOLD_=65536` in the process environment for bounded memory in long-running processes (package READMEs).

## Gemma Prohibited Use Policy

EmbeddingGemma 2 weights are Apache-2.0. The model card also states:

> Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

The sentence appears in the description and README of `Starlings.Models.EmbeddingGemma2` and of its adapter packages, and in the hosting repositories' notices.

## Hosting

Unchanged since 0.1.0-preview.6. The files Starling builds are downloadable without a token, each pinned by commit and by SHA-256 in the packages' locks:

| Repository | Revision | Files |
| --- | --- | --- |
| `Starlinghq/gliner2-base-v1-onnx` | `c02c3350a0dcaf0bd6d2878043cc6a4ba950dccb` | GLiNER2 graph |
| `Starlinghq/embeddinggemma-2-onnx` | `92a02ae2887a91180e611776dd9c286dc5fa35da` | EmbeddingGemma 2 text graph |
| `Starlinghq/embeddinggemma-2-vision-onnx` | `1a22d40bf395c7f0a42388df72c883b661a7b3d6` | EmbeddingGemma 2 vision and multimodal text graphs |
| `Starlinghq/qwen3.5-0.8b-ortgenai` | `95a690c1774f144957cb7b76986a9126e8b1bb5f` | Qwen 3.5 GenAI builds, tokenizer, chat template |

## Evidence

- Release drill: the `release` workflow's candidate job (`tools/release-drill-ai.sh --no-model`) packs every package from a clean clone, checks each nuspec and the per-profile maturity statements, and installs the packages into clean consumers outside the checkout.
- Gates: `docs/catalog/README.md` (generated), with links to `docs/evidence/`.
- Regression: the qualification workflow on the release tag (CPU release gates and soaks; `cuda (runpod)` on the RTX 3090).
