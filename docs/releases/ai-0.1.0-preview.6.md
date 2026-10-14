# ai-dotnet release candidate 0.1.0-preview.6

**Supersedes 0.1.0-preview.5, which was never published.** The preview.5 record also contradicts itself on the tokenizer pin: its summary says every package pins `Starlings.Tokenizers [0.1.0-preview.4]`, but the code it describes already pinned `[0.3.0-preview.1]`. This candidate states the actual pin and carries the production-readiness work merged since (`docs/plan/production-readiness.md`).

**Publishing.** The `release` workflow publishes this candidate from the tag `ai-v0.1.0-preview.6` on `main`, after a maintainer approves the `release` environment: the 13 packages of the publishable set below, through NuGet trusted publishing, plus a GitHub release with this record as notes. `Starlings.Tokenizers` is not part of this publish; it is already on nuget.org (0.3.0-preview.1, published from starlinghq/tokenizers).

**FFmpeg adapters held.** `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` (FFmpeg 8) and the new `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` are built and drilled but **not publishable** until the legal review of the FFmpeg adapters (decision D2). Both are in `HELD` in `.github/workflows/release.yml`. No other package depends on them.

**Maturity.** The packages carry the **Preview** label (plan 14). `multilingual-e5-small/onnx-fp32-cpu` is the only profile at maturity `preview`; every other profile is `experimental`. No gate became `passed` and no maturity changed in this candidate: the new evidence below is development evidence, and recording gate results is a maintainer act (M10). The generated catalog (`docs/catalog/`) lists every profile's gates and what blocks Stable.

## Packages

All packages target `net10.0`, are licensed MIT OR Apache-2.0 and have SourceLink to the commit.

| Package | Version | Set | Change since 0.1.0-preview.5 |
| --- | --- | --- | --- |
| `Starlings.Tokenizers` | 0.3.0-preview.1 | dependency, already on nuget.org | unchanged; every package that uses it pins `[0.3.0-preview.1]` |
| `Starlings.Runtime.Onnx` | 0.1.0-preview.6 | publishable | AOT-compatible (`IsAotCompatible`) |
| `Starlings.Acquisition` | 0.1.0-preview.6 | publishable | AOT-compatible |
| `Starlings.Models.E5` | 0.1.0-preview.6 | publishable | AOT-compatible; operational tests |
| `Starlings.Models.E5.Extensions.AI` | 0.1.0-preview.6 | publishable | AOT-compatible |
| `Starlings.Models.E5.MLNet` | 0.1.0-preview.6 | publishable | parallel cursor-set fix for any scheduler (found on Windows) |
| `Starlings.Models.Gliner2` | 0.1.0-preview.6 | publishable | graph hosting public; `mlnet` capability `unsupported`; AOT-compatible |
| `Starlings.Generation.OnnxGenAI` | 0.1.0-preview.6 | publishable | AOT-compatible |
| `Starlings.Models.Qwen35` | 0.1.0-preview.6 | publishable | hosting public (no token); AOT-compatible |
| `Starlings.Models.Qwen35.Extensions.AI` | 0.1.0-preview.6 | publishable | AOT-compatible |
| `Starlings.Models.EmbeddingGemma2` | 0.1.0-preview.6 | publishable | text and vision graph hosting public; AOT-compatible |
| `Starlings.Models.EmbeddingGemma2.Extensions.AI` | 0.1.0-preview.6 | publishable | AOT-compatible |
| `Starlings.Models.EmbeddingGemma2.MLNet` | 0.1.0-preview.6 | publishable | parallel cursor-set fix |
| `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` | 0.1.0-preview.6 | publishable (adapter) | version only |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` | 0.1.0-preview.6 | **held (D2)** | source shared with the FFmpeg 9 adapter; not-found message names each OS's library files |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` | 0.1.0-preview.6 | **held (D2)** | **new**: bindings generated from the FFmpeg n9.0.1 headers; every frame of the 22 fixture videos equal to torchcodec on FFmpeg 9.0.1 (a GPL distribution build, loaded with `AllowNonLgplBuild`; no LGPL FFmpeg 9 build tested) |

**Publishable set (13 packages):** every row at 0.1.0-preview.6 except the two FFmpeg adapters.

ML.NET under Native AOT is unsupported (Microsoft.ML emits code at run time); every other package published and ran under Native AOT with results equal to JIT (`docs/evidence/aot/`).

## Interoperability

The libraries integrate with the .NET AI ecosystem through `Microsoft.Extensions.AI.Abstractions` only: `IEmbeddingGenerator<string, Embedding<float>>` (e5, EmbeddingGemma 2), multimodal `IEmbeddingGenerator` (EmbeddingGemma 2) and `IChatClient` (Qwen 3.5). CI rejects any cloud SDK or AI provider package (`Azure.*`, `Microsoft.Azure.*`, `OpenAI`, `Microsoft.Extensions.AI.OpenAI`, `.AzureAIInference`, `.Ollama`) in any library, directly or transitively. `Qwen35ChatClient` rejects tools and structured output (`NotSupportedException`), which is the main gap for code ported from other `IChatClient` providers.

## Gemma Prohibited Use Policy

EmbeddingGemma 2 weights are Apache-2.0. The model card also states:

> Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

The sentence appears in the description and README of `Starlings.Models.EmbeddingGemma2` and of its adapter packages, and in the hosting repositories' notices.

## Hosting

The files Starling builds are downloadable without a token since 2026-10-14, each pinned by commit and by SHA-256 in the packages' locks:

| Repository | Revision | Files |
| --- | --- | --- |
| `Starlinghq/gliner2-base-v1-onnx` | `c02c3350a0dcaf0bd6d2878043cc6a4ba950dccb` | GLiNER2 graph |
| `Starlinghq/embeddinggemma-2-onnx` | `92a02ae2887a91180e611776dd9c286dc5fa35da` | EmbeddingGemma 2 text graph |
| `Starlinghq/embeddinggemma-2-vision-onnx` | `1a22d40bf395c7f0a42388df72c883b661a7b3d6` | EmbeddingGemma 2 vision and multimodal text graphs |
| `Starlinghq/qwen3.5-0.8b-ortgenai` | `95a690c1774f144957cb7b76986a9126e8b1bb5f` | Qwen 3.5 GenAI builds, tokenizer, chat template |

## What else changed since 0.1.0-preview.5

Integrated from draft PRs #3 to #12 and #14:

| Area | Change |
| --- | --- |
| Plan | `docs/plan/production-readiness.md`: per-model gate status, workstreams, maintainer actions (#3) |
| Catalog | `model-pipeline catalog`: catalog and compatibility matrix generated from the manifests, checked in CI (#4) |
| Gate records | G0 to G7 recorded in every manifest; G1/G2 rerun at tokenizer 0.3.0-preview.1 on CPU development fixtures, all passed with unchanged errors; GLiNER2 CUDA manifest (#6) |
| G3 evaluators | retrieval, NER span F1 and the Qwen generation task in `model-pipeline evaluate`; no budgets yet (#5) |
| G4 | `Starlings.Soak.Tool` and per-model operational tests (#9) |
| G5 | GLiNER2, EmbeddingGemma 2 and Qwen benchmark harnesses on a shared core (#7) |
| G6 | Native AOT check, ML.NET and public-hosting drill steps, Windows drill script (#10) |
| Qwen | 85-case chat-template corpus (proposed), long-context evidence to 16,391 tokens, upstream report draft (#12) |
| Docs | support policy, Stable contracts, maintained-update drill design, upstream change detector (#8) |
| EmbeddingGemma 2 | FFmpeg 9 adapter, CUDA fidelity tool, long-input fixture proposal (#14) |
| CI | Windows x64 job, weekly qualification workflow, SDK pinned exactly (`rollForward: disable`) (#11, #10) |

## Checks run for this candidate

Run on 2026-10-14: Intel i7-5820K, Linux x64, CPU only, .NET SDK 10.0.100 (pinned exactly), shared machine. The drill built commit `5204eea`; the commits after it change two capability statuses (from the drill's own evidence) and documentation only. The `release` workflow packs and drills the tagged commit again before publishing, so the published bytes come from that run.

| Check | Result |
| --- | --- |
| `tools/release-drill-ai.sh --seed-model-from <e5> --seed-gliner2-from <GLiNER2> --seed-embeddinggemma2-from <EmbeddingGemma 2> --seed-qwen35-from <Qwen fp32x> --public-hosting --aot` | **PASS** (`seed, GLiNER2 run, EmbeddingGemma 2 run, Qwen3.5 run, public hosting: GLiNER2, EmbeddingGemma 2, Qwen 3.5, AOT check`). [`rc-drill.log`](ai-0.1.0-preview.6/rc-drill.log) |
| Drill: packing and consumers | 15 packages packed from a clean clone with locked restore and an empty package cache; nuspecs and exact dependencies checked (`Starlings.Tokenizers [0.3.0-preview.1]`); consumers outside the checkout restored only the packed packages and nuget.org's tokenizer. |
| Drill: public hosting (no token, no seed, empty cache) | GLiNER2 graph (829,284,599 bytes), EmbeddingGemma 2 text graph (1,084,327,317 bytes) and the Qwen 3.5 fp32x bundle (3,077,177,344-byte data file) downloaded anonymously, every SHA-256 verified, each sample then ran offline: GLiNER2 expected entities; EmbeddingGemma 2 expected top document 4/4 and Matryoshka 256 same top 4/4; Qwen 3.5 answered and stopped at `<|im_end|>` (24 + 7 tokens). |
| Drill: Native AOT | Every core and Extensions.AI package: 0 trim or AOT warnings, AOT output equal to JIT (max abs 0). ML.NET adapters: AOT unsupported, as expected. The EmbeddingGemma 2 core-only AOT target was skipped for a missing seed file; its Extensions.AI target ran and matched. |
| `dotnet test --solution ai-dotnet/AI.slnx -c Release --max-parallel-test-modules 2`, model weights present | 1,031 tests: 988 passed, 43 skipped, 0 failed. Skips: 39 video and image fixture tests (no FFmpeg libraries or fixture media on this machine) and 4 opt-in network tests. |
| `model-pipeline`: pytest, ruff check, ruff format, `validate registry`, `catalog --check` | 539 passed; clean; 142 files formatted; 13 manifests valid; catalog up to date |
| `tools/check-boundary.py` | ok |
| CI library dependency boundaries, including the new vendor-neutral rule | no library lock file contains a cloud SDK or AI provider package |

**Not run for this candidate:** CUDA of any kind; Windows beyond CI; `model-pipeline evaluate` against P-030 release sets (none exist yet); the one-hour soak.

## Known issues

- **Qwen 3.5 on CPU, unchunked prefill.** By default a request computes the whole prompt in one native call: it cannot be cancelled until that call returns (3.8 to 6 s observed for about 330 tokens), and it allocates about 1 MB per prompt token (about 16 GB at 16k tokens). Set `PrefillChunkTokens` to bound both.
- **Qwen 3.5 cached versus uncached logits** grow with context: 4.1e-4 at 16,391 tokens, above the protected 2e-4 bound (selected tokens still equal). Long-context use above about 8k tokens is not covered by the fidelity claim.
- **e5 soak screening.** A 5-minute development soak flagged 33 to 44 MiB of growth that a longer repeat did not reproduce; the one-hour run on a quiet machine is pending (M6).
- **Memory after unload.** glibc keeps freed native memory after a model is disposed (GLiNER2, EmbeddingGemma 2 image, Qwen); loaded memory stays flat. `malloc_trim` returns about 90%.
- **Windows** is untested beyond CI's build and weight-free tests; macOS and ARM64 are untested.
- The earlier records' known issues and the per-package READMEs still apply.
