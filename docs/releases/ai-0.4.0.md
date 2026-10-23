# ai-dotnet release 0.4.0

**Starlings.Models.Bert reaches nuget.org.** This release follows `0.3.0` (published 2026-10-19) and carries the work merged since (#128 to #146). It adds one package to the publishable set, `Starlings.Models.Bert`: the BERT route's seven embedding and reranking profiles, now including the multilingual `BAAI/bge-m3` embedder (#128) and the `BAAI/bge-reranker-v2-m3` cross-encoder reranker (#146, closes #99). All seven profiles are **Experimental**. The package now carries its profiles: `BertModelProfile.BuiltIn(name)` works from the package alone, with no repository checkout.

**Publishing.** The `release` workflow publishes this release from the tag `ai-v0.4.0` on `main`, after a maintainer approves the `release` environment. It publishes the 18 packages of the publishable set below through NuGet trusted publishing, and creates a GitHub release with this record as notes.

**Tokenizer.** Every package that tokenizes now pins `Starlings.Tokenizers [0.3.2]` (0.3.0 in the previous release). #128 raised the pin to 0.3.1, which adds the lstrip/rstrip added-token options bge-m3 needs. #135 raised it to 0.3.2, which adds ByteLevel `use_regex` for Whisper. No pinned asset of an existing profile declares either option, so their outputs are unchanged. Every manifest's `processing.tokenizer.version` moved with the pin.

**Maturity.** Maturity belongs to profiles, not packages, and each package README has a generated maturity table. Every Stable profile of 0.3.0 is still Stable with the same contract: no gate, tolerance or `protected/` file changed. The new profiles are Experimental: no gate has run on them, and their capabilities are at most Experimental. They get registry manifests in this release candidate so the package README can state their maturity. Each manifest has every gate `not-run`. Their tolerances are the candidate's predeclared development policy (`qualification.tolerances` points at the candidate `profile.json`, as for QwenChat). Their references are proposed and not adopted into `protected/`. The BERT route has no admission record. The models were admitted to the onboarding route in the PRs below.

| Profile | Maturity | Admitted to the route |
| --- | --- | --- |
| `all-minilm-l6-v2/onnx-fp32-cpu` | Experimental | initial five-model onboarding (`model-pipeline/FIVE_MODELS.md`) |
| `bge-small-en-v1.5/onnx-fp32-cpu` | Experimental | initial five-model onboarding |
| `ms-marco-minilm-l6-v2/onnx-fp32-cpu` | Experimental | initial five-model onboarding |
| `nomic-embed-text-v1.5/onnx-fp32-cpu` | Experimental | #120 |
| `bge-base-en-v1.5/onnx-fp32-cpu` | Experimental | #121 |
| `bge-m3/onnx-fp32-cpu` | Experimental | #128 |
| `bge-reranker-v2-m3/onnx-fp32-cpu` | Experimental | #146 (closes #99) |

Candidate contracts: `model-pipeline/candidates/{minilm,bge-small,ms-marco,nomic-embed,bge-base,bge-m3,bge-reranker-v2-m3}/profile.json`. Registry manifests: `model-pipeline/registry/<family>/onnx-fp32-cpu/manifest.yaml`. The Stable profiles are listed in [`ai-0.2.0.md`](ai-0.2.0.md), and the Kev and QwenChat profiles in [`ai-0.3.0.md`](ai-0.3.0.md). The [catalog](../catalog/README.md) links every gate to its evidence.

## Packages

All packages target `net10.0`, are licensed MIT OR Apache-2.0 and have SourceLink to the commit. Sibling and tokenizer dependencies are exact (`[version]`).

| Package | Version | Change since 0.3.0 |
| --- | --- | --- |
| `Starlings.Tokenizers` | 0.3.2 | dependency, raised from 0.3.0 (#128, #135); outputs of existing profiles unchanged |
| `Starlings.Runtime.Onnx` | 0.4.0 | README maturity table lists the BERT profiles |
| `Starlings.Acquisition` | 0.4.0 | README maturity table lists the BERT profiles |
| `Starlings.Models.E5` | 0.4.0 | tokenizer 0.3.2 (README contract row); outputs unchanged |
| `Starlings.Models.E5.Extensions.AI` | 0.4.0 | rebuilt, no change |
| `Starlings.Models.E5.MLNet` | 0.4.0 | rebuilt, no change |
| `Starlings.Models.Gliner2` | 0.4.0 | tokenizer 0.3.2 (README contract row); outputs unchanged |
| `Starlings.Generation.OnnxGenAI` | 0.4.0 | tokenizer dependency 0.3.2; no code change |
| `Starlings.Models.Qwen35` | 0.4.0 | tokenizer 0.3.2 (README contract row); outputs unchanged |
| `Starlings.Models.Qwen35.Extensions.AI` | 0.4.0 | rebuilt, no change |
| `Starlings.Models.EmbeddingGemma2` | 0.4.0 | tokenizer 0.3.2 (README contract rows, capability comment); outputs unchanged |
| `Starlings.Models.EmbeddingGemma2.Extensions.AI` | 0.4.0 | rebuilt, no change |
| `Starlings.Models.EmbeddingGemma2.MLNet` | 0.4.0 | rebuilt, no change |
| `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` | 0.4.0 | rebuilt, no change |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` | 0.4.0 | rebuilt, no change |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` | 0.4.0 | rebuilt, no change |
| `Starlings.Models.Kev` | 0.4.0 | tokenizer dependency 0.3.2; no code change |
| `Starlings.Models.QwenChat` | 0.4.0 | tokenizer dependency 0.3.2; no code change |
| `Starlings.Models.Bert` | 0.4.0 | **first publish**. BERT-family text embeddings and paired-input reranking on ONNX Runtime 1.30.0 CPU fp32, with owned tokenization (WordPiece; XLM-R Unigram for bge-m3 and the reranker). Seven Experimental profiles: all-MiniLM-L6-v2, bge-small-en-v1.5, bge-base-en-v1.5 (#121), bge-m3 (#128), nomic-embed-text-v1.5 (#120), and the rerankers ms-marco-MiniLM-L6-v2 and bge-reranker-v2-m3 (#146). New API in this release: `BertModelProfile.BuiltIn(name)` and `BertModelProfile.BuiltInIds` parse the candidate contracts embedded in the package; the embedded files are linked from `model-pipeline/candidates/*/profile.json` at build time, so they cannot drift. Since 0.3.0 on `main`: a data-driven per-repository spec (output name, segment ids, XLM-R geometry; #128). `RuntimeLock` also includes `onnx/model.onnx_data` when the profile pins it (#146). Opt-in batching options as in #126. Model files are downloaded explicitly (`ModelAssetStore.DownloadAsync(profile.RuntimeLock)`); inference never downloads |

**Publishable set (18 packages):** every row at 0.4.0. Nothing is held (`HELD` is empty in `.github/workflows/release.yml`).

**Not published:** `Starlings.Models.Whisper` (`openai/whisper-large-v3-turbo` speech-to-text, #135, #136, #141, #142) is packable, and it builds and tests on `main`. It is not in the release drill's package set. Its profile `whisper-large-v3-turbo/onnx-fp16-cpu` has no registry manifest, so the package README has no generated per-profile maturity statement and its description lacks the per-profile line, and the drill requires both. It also has no admission record. Publishing it needs its own change: an Experimental registry manifest, the README maturity block, and the drill and publish entries, as Kev and QwenChat had.

ML.NET under Native AOT is unsupported (Microsoft.ML emits code at run time).

## Upgrading from 0.3.0

- Raise every `Starlings.*` reference to `0.4.0` together. The tokenizer comes along as `0.3.2`.
- No default changed, and Stable profiles produce the same outputs.
- `Starlings.Models.Bert`: add `Microsoft.ML.OnnxRuntime` 1.30.0 (the CPU natives) to the application. Pick a profile with `BertModelProfile.BuiltIn("bge-small")` (names in `BertModelProfile.BuiltInIds`) in place of `BertModelProfile.Load("model-pipeline/candidates/bge-small/profile.json")`. `Load` and `Parse` still work.
- If you used the BERT library from source before this release: `bge-m3`'s `RuntimeLock` now has **3 files** (`onnx/model.onnx`, `onnx/model.onnx_data`, `tokenizer.json`), not 2, because its graph keeps its weights in external data (#146). Download `RuntimeLock` again, or `Create` reports the missing `onnx/model.onnx_data`. `bge-reranker-v2-m3` has the same 3 files. The other five profiles still need 2.
- Reranker scores (ms-marco and bge-reranker-v2-m3) are raw logits, not probabilities. Apply a sigmoid yourself for bge-reranker-v2-m3's 0-1 mapping. nomic-embed-text-v1.5 vectors are not unit length.

## Hosting

The files Starling builds can be downloaded without a token. Each is pinned by commit and by SHA-256 in the packages' locks. New since 0.3.0:

| Repository | Revision | Files |
| --- | --- | --- |
| `Starlinghq/bge-reranker-v2-m3-onnx` | `047d4622e93d347828457e8a13bccd997fb6a969` | bge-reranker-v2-m3 fp32 ONNX export (`onnx/model.onnx`, `onnx/model.onnx_data`), the original `BAAI/bge-reranker-v2-m3` files at `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` unchanged (tokenizer included), `SHA256SUMS` |

The other six BERT profiles download from their public upstream repositories at the revisions their candidate contracts pin. These are `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-m3`, `cross-encoder/ms-marco-MiniLM-L6-v2` and `nomic-ai/nomic-embed-text-v1.5`, each of which ships `onnx/model.onnx`.

Unchanged since 0.3.0: the Kev and QwenChat repositories at the revisions in [`ai-0.3.0.md`](ai-0.3.0.md#hosting), and the others at the revisions in [`ai-0.2.0.md`](ai-0.2.0.md#hosting).

## Evidence

- Release drill: the `release` workflow's candidate job runs `tools/release-drill-ai.sh --no-model`. It packs every package from a clean clone, checks each nuspec and the per-profile maturity statements, and installs the packages into clean consumers outside the checkout. New in this release, `tools/release-drill/bert-consumer` restores `Starlings.Models.Bert` alone from the local feed. Offline, it checks that the seven built-in profiles parse from the package with the expected ids, tasks and runtime locks. It also checks that `Create` verifies an empty snapshot before anything loads.
- BERT route: onboarding route `bert-onnx-cpu-v1` (`model-pipeline/recipes/*-cpu.yaml`), development parity against fresh framework references on the proposed corpora `model-pipeline/corpora/five-models/`. The test `BuiltInProfileTests` checks that each built-in profile equals `Load` of the repository file byte for byte.
- Gates: `docs/catalog/README.md` (generated), with links to `docs/evidence/`.
- Regression: the qualification workflow on the release tag (CPU release gates and soaks; `cuda (runpod)` on the RTX 3090).
