# ai-dotnet release candidate 0.1.0-preview.5

Named `ai-*` rather than `e5-*` because the set now covers e5, GLiNER2, Qwen 3.5 and EmbeddingGemma 2; earlier records (`e5-0.1.0-preview.1` to `.4`) keep their names.

**Publishing.** The `release` workflow publishes this candidate from the tag `ai-v0.1.0-preview.5`, after a maintainer approves the `release` environment: the 13 packages of the publishable set below, through NuGet trusted publishing, plus a GitHub release with this record as notes. Until that run, nothing is on nuget.org.

**FFmpeg adapter held.** `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` 0.1.0-preview.5 is built and drilled locally, but it is **not part of the publishable set**. It stays out of every published release until the legal review of the FFmpeg adapter is done (decision D2, Mark, 2026-10-12). The review covers dynamic loading of the user's own LGPL FFmpeg, shipping no FFmpeg binaries, rejecting GPL and nonfree builds unless the caller opts in, leaving codec patent responsibility with the deployer, and FFmpeg.AutoGen 8.1.0 under MIT (`ai-dotnet/THIRD-PARTY-NOTICES.md`). No other package depends on it, so holding it back leaves the rest of the set whole.

**Maturity.** The packages carry the **Preview** label (plan 14). Capabilities have their own statuses: `multilingual-e5-small/onnx-fp32-cpu` is the only profile at maturity `preview` (qualified by hand in [0.1.0-preview.1](e5-0.1.0-preview.1.md) and [0.1.0-preview.2](e5-0.1.0-preview.2.md)). Every other profile is `experimental` (`model-pipeline/registry/*/*/manifest.yaml`).

This candidate exists because ai-dotnet source changed after 0.1.0-preview.4 (`74da978`): the EmbeddingGemma 2 packages arrived, with text, image and video embeddings, in nine commits under `ai-dotnet/src`. The tokenizer had no source or profile change after its 0.1.0-preview.4 candidate. Its only change is a README line recording the last drill run, so it stays at 0.1.0-preview.4, and every ai-dotnet package still pins it exactly at `[0.1.0-preview.4]`.

## Packages

All packages target `net10.0`, are licensed MIT OR Apache-2.0 and have SourceLink to the commit. Platform statuses come from the package READMEs and the capability table in `ai-dotnet/README.md`, where every capability not listed as experimental is untested or unsupported.

| Package | Version | Set | Change since 0.1.0-preview.4 | Platform statuses |
| --- | --- | --- | --- | --- |
| `Starlings.Tokenizers` | 0.3.0-preview.1 | dependency (published from starlinghq/tokenizers) | now the published package, pinned exactly: the qualified 0.1.0-preview.4 behavior for these profiles, every conformance fixture unchanged | Linux x64 experimental for all five profiles; Windows, macOS, ARM64 untested |
| `Starlings.Runtime.Onnx` | 0.1.0-preview.5 | publishable | version only | Linux x64 CPU and CUDA (through the model profiles) experimental; Windows, macOS, ARM64, Native AOT untested |
| `Starlings.Acquisition` | 0.1.0-preview.5 | publishable | version only | Linux x64 experimental; Windows, macOS untested |
| `Starlings.Models.E5` | 0.1.0-preview.5 | publishable | version only | `onnx-fp32-cpu` Linux x64 **preview** (qualified); `onnx-fp32-cuda` experimental; Windows, macOS, ARM64, Native AOT untested |
| `Starlings.Models.E5.Extensions.AI` | 0.1.0-preview.5 | publishable | version only, README link | Linux x64 CPU experimental; over CUDA untested |
| `Starlings.Models.E5.MLNet` | 0.1.0-preview.5 | publishable | version only, README link | Linux x64 CPU pipeline composition experimental; over CUDA untested; save, cancellation, training unsupported |
| `Starlings.Models.Gliner2` | 0.1.0-preview.5 | publishable | version only | `gliner2-base-v1/onnx-fp32-cpu` Linux x64 CPU and CUDA (`use_tf32=0`) experimental; other platforms untested |
| `Starlings.Generation.OnnxGenAI` | 0.1.0-preview.5 | publishable | version only | Linux x64 CPU and CUDA experimental; natives for other RIDs exist but were not run (untested) |
| `Starlings.Models.Qwen35` | 0.1.0-preview.5 | publishable | version only | `ortgenai-fp32x` CPU, `ortgenai-fp32x` CUDA (TF32 off), `ortgenai-fp16x` CUDA, all Linux x64 experimental; Windows, macOS, ARM64, Native AOT untested |
| `Starlings.Models.Qwen35.Extensions.AI` | 0.1.0-preview.5 | publishable | version only | Linux x64 CPU and CUDA (fp32x) experimental; fp16x untested |
| `Starlings.Models.EmbeddingGemma2` | 0.1.0-preview.5 | publishable | **new** | Profiles `onnx-fp32-{cpu,cuda}`, `image-fp32-{cpu,cuda}`, `video-fp32-{cpu,cuda}`: Linux x64 experimental. Inputs of 2,050 to 8,192 tokens, the 18 task prompts, image budgets other than 280, variable frame rate, interlaced and HDR video, Windows, macOS, ARM64, Native AOT: untested. Audio, timestamps, inputs above 8,192 tokens: unsupported |
| `Starlings.Models.EmbeddingGemma2.Extensions.AI` | 0.1.0-preview.5 | publishable | **new** | Linux x64 CPU experimental (text, image and video generators); over CUDA untested |
| `Starlings.Models.EmbeddingGemma2.MLNet` | 0.1.0-preview.5 | publishable | **new** | Linux x64 CPU experimental (text column); over CUDA untested; save unsupported; video column unsupported |
| `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` | 0.1.0-preview.5 | publishable (adapter) | **new** | PNG and JPEG on Linux x64 experimental (decoded RGB equal to PIL on the 28 fixture images); GIF, BMP, WebP untested; Windows and macOS natives untested; 16-bit images unsupported |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` | 0.1.0-preview.5 | **held: built and drilled, not publishable until the FFmpeg legal review (D2)** | **new** | Linux x64 on the user's FFmpeg 8.x experimental (every frame of the 22 fixture videos equal to torchcodec 0.17.0's on FFmpeg n8.1.3); ARM64, non-AVX2 CPUs, Windows, macOS untested; FFmpeg majors other than 8 unsupported |

**Publishable set (14 packages):** the tokenizer at 0.1.0-preview.4 and the 13 ai-dotnet packages above at 0.1.0-preview.5, every row except `Video.FFmpeg`.

### Adapters are voluntary

Adapter packages are the Extensions.AI and ML.NET integrations and the SkiaSharp and FFmpeg decoders. An application references them only when it wants them. No core package references an adapter. The core packages are `Tokenizers`, `Runtime.Onnx`, `Acquisition`, `Models.E5`, `Models.Gliner2`, `Generation.OnnxGenAI`, `Models.Qwen35` and `Models.EmbeddingGemma2`. Two checks confirm this:

- The drilled nuspecs ([`rc-drill.log`](ai-0.1.0-preview.5/rc-drill.log), section 4). `Models.EmbeddingGemma2` depends only on `Acquisition [0.1.0-preview.5]`, `Runtime.Onnx [0.1.0-preview.5]` and `Tokenizers [0.1.0-preview.4]`, and the other core packages likewise depend on no adapter. The dependencies point one way: adapter to core. `Imaging.SkiaSharp` depends on `Models.EmbeddingGemma2` and SkiaSharp 4.153.1. `Video.FFmpeg` depends on `Models.EmbeddingGemma2` and `FFmpeg.AutoGen [8.1.0]`.
- The project files. No `ProjectReference` under `ai-dotnet/src` points to an adapter project. CI admits FFmpeg.AutoGen only in `Starlings.Models.EmbeddingGemma2.Video.FFmpeg.csproj`, SkiaSharp only in the imaging adapter and Microsoft.ML only in the ML.NET adapters. The drill found no FFmpeg library in any package, package cache or build output.

The `EmbeddingGemma2Search` sample references both decoders to show the image and video steps. Samples are not packages and are not published.

## Gemma Prohibited Use Policy

EmbeddingGemma 2 weights are Apache-2.0 (model card front matter and its license link; `model-pipeline/admissions/embeddinggemma-2.md`). The card also states:

> Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

The sentence appears in the description and README of `Starlings.Models.EmbeddingGemma2` and in the READMEs of its Extensions.AI, ML.NET, SkiaSharp and FFmpeg packages. It applies to every deployment that uses these packages with the EmbeddingGemma 2 weights. The packages ship asset locks, not weights. The Starling ONNX graphs (`embeddinggemma-2-onnx-r1`, `embeddinggemma-2-vision-onnx-r1`) are not published yet, and the drill seeded them locally.

## What changed since 0.1.0-preview.4

| Package | Change |
| --- | --- |
| `Starlings.Models.EmbeddingGemma2` 0.1.0-preview.5 (new) | `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b` on ONNX Runtime 1.30.0. Text embeddings from the owned tokenizer (profile `gemma-bpe-embeddinggemma2`) or token ids: explicit named prompts, mean pooling including prompt tokens, L2, Matryoshka 768/512/256/128, inputs above 8,192 tokens rejected. Image and interleaved text-and-image embeddings from decoded RGB. Video and interleaved text, image and video embeddings from decoded frames (1 fps, at most 32 frames, 140 tokens per frame). Asset locks for the text, vision and multimodal text graphs. |
| `Starlings.Models.EmbeddingGemma2.Extensions.AI` 0.1.0-preview.5 (new) | `IEmbeddingGenerator<string, ...>` for text; `IEmbeddingGenerator<DataContent, ...>` and `IEmbeddingGenerator<IEnumerable<AIContent>, ...>` with image and video decoder hooks; `Dimensions` mapped to the Matryoshka sizes; DI. |
| `Starlings.Models.EmbeddingGemma2.MLNet` 0.1.0-preview.5 (new) | ML.NET text-column transform; save unsupported. |
| `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` 0.1.0-preview.5 (new) | PIL-compatible PNG and JPEG decoding into `EmbeddingGemma2Image`; EXIF orientation opt-in; 50 MP limit. |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` 0.1.0-preview.5 (new, **held**) | torchcodec-equivalent decoding on the user's FFmpeg 8.x through FFmpeg.AutoGen 8.1.0; ships no FFmpeg binaries. |
| `Runtime.Onnx`, `Acquisition`, `Models.E5`, `Models.E5.Extensions.AI`, `Models.E5.MLNet`, `Models.Gliner2`, `Generation.OnnxGenAI`, `Models.Qwen35`, `Models.Qwen35.Extensions.AI` 0.1.0-preview.5 | Version only, with an unchanged exact tokenizer dependency `[0.1.0-preview.4]`. No source change: `git diff 74da978 -- ai-dotnet/src` touches only the five EmbeddingGemma 2 directories. The `Models.E5`, `Models.E5.Extensions.AI` and `Models.E5.MLNet` READMEs link to this record. |
| `Starlings.Tokenizers` 0.3.0-preview.1 | The published tokenizer from starlinghq/tokenizers (renamed from the planning-era package; output unchanged). Restored from nuget.org with the bytes the lock files record. |

The version wiring after the bump:

- `ai-dotnet/Directory.Build.props`: `VersionSuffix` is `preview.5`, and the release notes point here.
- The four samples set `StarlingsVersion` to `0.1.0-preview.5`.
- Lock files are refreshed. Only the project-reference ranges of ai-dotnet projects changed, from `[0.1.0-preview.4, )` to `[0.1.0-preview.5, )`; the tokenizer entries are unchanged.
- The registry manifests' `processing.tokenizer.version` stays `0.1.0-preview.4` in all 12 manifests, because it names the tokenizer, not the ai-dotnet set. No manifest changed.
- The drill scripts read both versions from MSBuild and needed no change.

## Checks run for this candidate

Every check ran at commit `d981fa3` on 2026-10-12: Intel i7-5820K, Linux x64, CPU only, .NET SDK 10.0.100 (10.0.401 outside the clone), runtime 10.0.12, with `DOTNET_PROCESSOR_COUNT=4` (shared machine). Later commits on the branch add only this record and its logs.

| Check | Result |
| --- | --- |
| `tools/release-drill-tokenizers.sh` | **PASS.** Clean clone. Five tokenizer assets required (e5, GLiNER2, Kev, EmbeddingGemma 2, Qwen 3.5). Locked restore: 1,558 tests passed, 0 skipped. Pack 0.1.0-preview.4 with no dependencies. The consumer, built from the local feed, matched exactly: encode 49/49, decode 49/49, truncation 2/2, padded batch 6/6. [`tokenizer-drill.log`](ai-0.1.0-preview.5/tokenizer-drill.log) |
| `tools/release-drill-ai.sh --seed-model-from <e5 snapshot> --seed-gliner2-from <dir> --seed-embeddinggemma2-from <drill-seed-video> --ffmpeg-dir <FFmpeg n8.1.3 LGPL lib> --seed-qwen35-from <fp32x bundle>` | **PASS with every model step** (`seed, GLiNER2 run, EmbeddingGemma 2 run with FFmpeg, Qwen3.5 run`); details in the rows below. [`rc-drill.log`](ai-0.1.0-preview.5/rc-drill.log) |
| Drill: packing and consumer install | 15 packages packed from a clean clone with locked restore. Nuspec and exact dependency checks passed (`Starlings.Tokenizers [0.1.0-preview.4]` in every package that uses it). Five consumers outside the checkout: E5Search, Gliner2Ner, QwenChat, EmbeddingGemma2Search and AiConsumer. Their restores held only packages, with sha512 equal to the packed files. No foreign tokenizer, no GPU natives and no FFmpeg library appeared in any cache or output. |
| Drill: e5 (offline) | Owned tokenizer ids and masks exact 46/46. Typed API, batch, Extensions.AI and ML.NET vs protected: max abs 2.33e-7 (bound 1e-5); batch vs own single rows 0. |
| Drill: GLiNER2 (offline) | Expected entities and offsets; 618-token input rejected (> 512). |
| Drill: EmbeddingGemma 2 text (offline) | Expected top document for 4/4 queries; Extensions.AI equal to the typed API (max abs 0); Matryoshka 256 keeps the same top document 4/4; 9,002-token input rejected (> 8,192). |
| Drill: EmbeddingGemma 2 images (offline) | Image lock verified. Expected top photo for 5/5 queries. Extensions.AI images equal to the typed API (max abs 0). |
| Drill: EmbeddingGemma 2 video (offline, user FFmpeg n8.1.3-17-gc389840894 LGPL) | Sampled frames as expected on 3 clips. Expected top clip for 3/3 queries. Extensions.AI video equal to the typed API (max abs 0). Frames without timing rejected. |
| Drill: Qwen 3.5 (offline, fp32x CPU) | "The capital of France is Paris." Stopped at `<|im_end|>`, usage 24 + 7 tokens. `IChatClient` gave the same answer and usage. |
| `dotnet test --solution ai-dotnet/AI.slnx -c Release` (tokenizer assets present, model graphs absent; run with `--max-parallel-test-modules 4`) | 692 tests: 574 passed, 118 skipped (model files absent or opt-in), 0 failed |
| `dotnet test --solution tokenizers-dotnet/Tokenizers.slnx` | 1,558 passed, 0 skipped, 0 failed |
| `tools/check-tokenizer-independence.sh` | ok (exit 0) |
| `model-pipeline`: `python3 -m pytest`, `ruff check .`, `ruff format --check .`, `model-pipeline validate registry` | 317 passed; all checks passed; 103 files already formatted; 12 manifests checked, 0 invalid |

Package SHA-256 of the drilled files is in `rc-drill.log`, section 4. Package bytes are not reproducible (random part names, timestamps), so a publish must use the drilled files or re-drill.

**Not run for this candidate:**

- `model-pipeline evaluate` (G1 and G2) with the models. The EmbeddingGemma 2 manifests keep G1 and G2 at `not-run` until the protected release set exists (decisions D4 and D5, work item W7).
- `dotnet test` with the real model graphs.
- CUDA checks of any kind.
- `EmbeddingGemma2.MLNet` was packed and inspected by the drill but not run there; its unit tests ran in the solution test run.

A note on the test command: the first attempt passed `-m:4` to `dotnet test --solution`. Microsoft.Testing.Platform forwarded it to the test modules, which ran zero tests (exit code 5). The run above caps parallelism with `--max-parallel-test-modules 4` instead.

## Known issues

- The EmbeddingGemma 2 and GLiNER2 graphs and the Qwen 3.5 hosting repository are not public yet (release assets unpublished, hosting repository private). A clean install can acquire the tokenizer and config files but not the graphs.
- The other known issues are those of the earlier records ([0.1.0-preview.2](e5-0.1.0-preview.2.md#known-issues)) and the per-package READMEs.

## Repository move (2026-10-13)

The candidate moved to `starlinghq/starling` as a single commit (`a20f19f`). The tokenizer is now the published `Starlings.Tokenizers` 0.3.0-preview.1, and the GLiNER2, EmbeddingGemma 2 and Qwen 3.5 files are hosted in the Starlinghq organization on Hugging Face, pinned by commit (private during the preview; byte-identical to the earlier files).

Rerun on `a20f19f` (Linux x64): `tools/release-drill-ai.sh --no-model` passed. 14 packages packed and inspected (13 publishable plus the held FFmpeg adapter); five consumers outside the checkout restored the packed packages and `Starlings.Tokenizers` from nuget.org with the locked bytes; e5 tokenizer ids exact 46/46. ai-dotnet tests: 593 passed, 118 skipped (GPU and opt-in), 0 failed, the same outcomes as before the move. model-pipeline: 459 passed; 12 manifests valid.
