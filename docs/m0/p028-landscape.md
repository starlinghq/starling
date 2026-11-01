# P-028 landscape comparison: .NET encoder and tokenizer options

Status: draft for review. Covers the section 3.1 candidates of `docs/plan/fast-faithful-ai-dotnet.md` plus notable packages found while surveying. The first named consumer, the second half of P-028, is recorded since 2026-11-01: starlinghq/roost and starlinghq/corvid (#208; section 6, item 1).

## 1. Method and date

Checked on 2026-10-05.

- Versions, publish dates, listing state and license expressions come from the NuGet v3 registration API (`https://api.nuget.org/v3/registration5-gz-semver2/<id>/index.json`). Dates are the `published` field, UTC, truncated to the day. Unlisted versions (shown by NuGet as published 1900-01-01) are excluded.
- Target frameworks, dependencies and the source commit recorded by the package come from the `.nuspec` in the flat container. Native assets come from listing the `.nupkg` contents.
- Source claims come from `git clone` of each repository at the SHA given below. Statements about code describe that SHA only.
- github.com web pages, api.github.com and huggingface.co were not reachable. Facts that come only from web search are marked **(unverified)**. Text from upstream READMEs is reported as the project's own claim, not as a verified result.
- "Not found" means not found in the cloned source at the given SHA. It does not mean the feature is absent from every branch or fork.

## 2. Comparison table

Package versions and dates are from NuGet. "Stable" is the latest version without a prerelease label. "Prerelease" is the latest prerelease version.

| Option | Package ID | Latest stable | Latest prerelease | License | Repo and SHA checked | TFMs (latest) | Native dependency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic Kernel ONNX connector | Microsoft.SemanticKernel.Connectors.Onnx | none (prerelease only) | 1.80.1-alpha, 2026-09-03 | MIT | microsoft/semantic-kernel @ 9974625ddc1e1d3f55e095420fcabcc9329f7f34 | net10.0, net8.0, netstandard2.0 | ONNX Runtime 1.23.2, ONNX Runtime GenAI 0.11.4 |
| SmartComponents LocalEmbeddings | SmartComponents.LocalEmbeddings | none (prerelease only) | 0.1.0-preview10148, 2024-04-04 | none in nuspec; repo LICENSE is MIT | dotnet/smartcomponents @ bd1c9e756ad45d7f8318440ed28fe7fc08c371da | net6.0, net8.0 | ONNX Runtime 1.17.0 |
| FastBertTokenizer | FastBertTokenizer | 1.2.6, 2026-09-14 | 1.1.30-alpha, 2025-03-03 | MIT | georg-jung/FastBertTokenizer @ 31f7cb322ff29a915fc7b4236dd4144a2b3f1ca6 (package records 47041c920c55a15e63417c97862026554f006c6a) | net10.0, net8.0, netstandard2.0 | none (managed) |
| ErgoX.TokenX | ErgoX.TokenX.HuggingFace | none (0.x only) | 0.22.1, 2025-11-01 (only version) | Apache-2.0 (nuspec); no LICENSE file in repo | ergosumx/tokenx @ 2a18748cd6cbec7cd1c1ab5930fe412d139fd71c | net8.0 | Rust `tokenizers` 0.22.1 via C FFI; in-package binaries for linux-x64 and win-x64; separate .Runtime.iOS and .Runtime.Android packages |
| Tokenizers.DotNet | Tokenizers.DotNet (+ Tokenizers.DotNet.runtime.&lt;rid&gt;) | 1.4.1, 2026-04-16 | none | MIT | sappho192/Tokenizers.DotNet @ b8eaa77d72bf9d9c768d75599bfa6a602e467b93 | net6.0, netstandard2.0 | Rust `tokenizers` 0.22.2; runtime packages for win-x64, win-arm64, linux-x64, linux-arm64, osx-x64, osx-arm64 |
| Tokenizers.HuggingFace | Tokenizers.HuggingFace | 3.23.1, 2026-05-12 | 2.21.4-rc.0, 2025-08-30 | Apache-2.0 | IgnaciodelaTorreArias/Tokenizers.HuggingFace.DotNet @ f566bdcf51afb746a815cbfb9e8239d3a4a379d8 | net6.0 | Rust `tokenizers` 0.23.1 via protobuf over C ABI; in-package binaries for 6 RIDs |
| Tokenizers.NET | Tokenizers.NET | none (alpha only) | 1.1.1-alpha.64-2d85cb3, 2025-06-21 | none in nuspec; no LICENSE file in repo | budgetdevv/Tokenizers.NET @ a5463916bffffc8791114034be06c1b7db9b402c | net9.0 | Rust `tokenizers` 0.20.0; in-package binaries for 6 RIDs |
| Microsoft.ML.Tokenizers | Microsoft.ML.Tokenizers | 2.0.0, 2025-11-11 | 3.0.0-preview.26457.2, 2026-09-09 | MIT | dotnet/machinelearning @ 5a44dfdbf20d1b64211da313dfbaa9ffbf63ea5d | net8.0, netstandard2.0 | none (managed) |
| ElBruno LocalEmbeddings | ElBruno.LocalEmbeddings | 1.6.3, 2026-09-28 | 1.0.9-preview, 2026-02-14 | MIT | elbruno/elbruno.localembeddings @ bddc8120cf1689297587af7cc1b38bec94c998f6 | net10.0, net8.0 | ONNX Runtime 1.24.4 |
| Curiosity SentenceTransformers | SentenceTransformers | 26.8.3937, 2026-08-13 | none | MIT | curiosity-ai/sentence-transformers-sharp @ 16bb432014187d4c0bb924f65557d24c2d3628d3 | net10.0 | ONNX Runtime 1.27.1 |

Coverage and evidence, at the SHAs above:

| Option | Algorithms | `tokenizer.json` | Model coverage | Offsets | Batching | Published evidence | Release cadence (NuGet) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SK ONNX connector | WordPiece via FastBertTokenizer 1.0.28 | No; loads `vocab.txt` | BERT-style ONNX encoders; pooling Mean, Max, MeanSquareRootTokensLength; optional L2 normalize | Not exposed | API takes a list; inputs run one at a time | Integration tests compare to sentence-transformers Python output for bge-micro-v2 (HF revision f09f671) and all-MiniLM-L6-v2 at 8e-7 tolerance; no benchmarks found | 92 alpha versions since 2024-04-03; 1 to 2 per month in 2026 |
| SmartComponents LocalEmbeddings | WordPiece via FastBertTokenizer 0.4.67 (published package) | No; `vocab.txt` | BERT-style ONNX models with `input_ids`, `attention_mask`, `token_type_ids`; default bge-micro-v2 downloaded at build | Not exposed | Not found | Unit tests for embedding types and nearest-neighbour search; no tokenizer conformance or benchmarks found | 11 previews 2024-03-12 to 2024-04-04; none since |
| FastBertTokenizer | WordPiece only; no pair encoding (README) | Yes, for BERT WordPiece configurations; also `vocab.txt` | BERT family (tests use bert-base-uncased, bert-base-multilingual-cased, bert-base-chinese, BAAI bge-small-en) | Range per batch item for stride/chunking; no per-token offsets found | Batch enumerators, parallel and async | Tests compare token IDs to Rust `tokenizers` 0.23 on 15,000 Simple English Wikipedia articles with documented known differences; BenchmarkDotNet results published from a CI run with a cross-library table | 7 listed versions 2023-09 to 2026-09; the last two stable gaps were about 4.5 and 28.5 months |
| ErgoX.TokenX | All Rust `tokenizers` models (wrapper) | Yes (Rust loader) | Any HF `tokenizer.json`; chat templates (README) | Yes (FFI exposes offsets) | Not determined | README claims 2,500+ Python parity cases verified by SHA-256; test project and Python generators present; not run here | One version, 2025-11-01. Repository archived 2026-02-25 **(unverified)** |
| Tokenizers.DotNet | All Rust `tokenizers` models (wrapper) | Yes (Rust loader) | Any HF `tokenizer.json`; hub download helper | Not exposed (`Encode` returns `uint[]`) | Not exposed | API, FFI, thread-safety and lifetime tests; no parity corpus found | 16 versions 2024-06 to 2026-04; roughly every 2 to 6 months since 2025 |
| Tokenizers.HuggingFace | All Rust `tokenizers` models; normalizers, pre-tokenizers and training exposed | Yes (Rust loader) | Any HF `tokenizer.json` | Yes (pre-tokenizer splits with offset type and referential) | Not determined | No test project found in the .NET repo | 9 versions 2025-05 to 2026-05; major version tracks the binding, minor/patch track `tokenizers` |
| Tokenizers.NET | All Rust `tokenizers` models (wrapper) | Yes (Rust loader) | Any HF `tokenizer.json` | Not determined | Batch buffers configured at construction | Encode/decode tests; no parity corpus or published benchmarks found | Alpha builds per commit; last 2025-06-21 |
| Microsoft.ML.Tokenizers | BPE, WordPiece/BERT, SentencePiece BPE and Unigram, Tiktoken, CodeGen, EnglishRoberta, Llama, Phi2 | Partial: per-model factories; a Unigram factory from `tokenizer.json` was added in #7625 (2026-07-16, first shipped in 3.0.0-preview.26457.2); no generic pipeline loader found | Model name tables for OpenAI encodings through gpt-6 on main; Llama, Phi, DeepSeek-style ByteLevel BPE | Yes (`EncodedToken.Offset`, a `Range` over the input string) | Not found as a dedicated API | Unit tests with reference token files under `test/Microsoft.ML.Tokenizers.Tests`; no tokenizer benchmarks found in the repo | Stable once a year in November with the .NET cycle; previews in between (see section 4) |
| ElBruno LocalEmbeddings | WordPiece via Microsoft.ML.Tokenizers 2.0.0 `BertTokenizer` | No; `vocab.txt` | Sentence-transformer ONNX models from the HF Hub; Harrier and image models in companion packages | Not exposed | Yes, with batch size auto-tuning | Unit tests; BenchmarkDotNet project including tokenizer benchmarks and a committed `performance-baseline.json`; no Python parity found in the core tests | 32 versions since 2026-02-12, several per month |
| Curiosity SentenceTransformers | BERT tokenizers derived from BERTTokenizers (README); BPE helpers | Model packages ship `tokenizer.json` resources | Fixed set: MiniLM, ArcticXs, Harrier, Qwen3, with models embedded in packages | Not determined | Yes (chunking, parallel execution) | Opt-in end-to-end test reproducing the Harrier model-card similarity matrix; benchmark projects present | 95 versions since 2024-04-30, date-based version numbers, bursts of releases |

## 3. Per-option notes

### 3.1 Semantic Kernel ONNX connector (Microsoft.SemanticKernel.Connectors.Onnx)

- Every published version carries an `-alpha` suffix; `VersionSuffix` is `alpha` in the project file. The connector also contains ONNX Runtime GenAI chat completion.
- `BertOnnxTextEmbeddingGenerationService` is marked `[Obsolete]` in favor of `AddBertOnnxEmbeddingGenerator`, which registers a Microsoft.Extensions.AI `IEmbeddingGenerator<string, Embedding<float>>`.
- `BertOnnxOptions` exposes casing, maximum tokens (default 512), CLS/SEP/UNK/PAD strings, Unicode normalization form (default FormD), pooling mode and normalization. Tokenizer configuration is set by the caller, not read from model assets.
- The 1.80.1-alpha package pins FastBertTokenizer 1.0.28 (2024-04-30) and ONNX Runtime 1.23.2. The newest of each on NuGet are 1.2.6 and 1.30.0.
- The integration tests download assets from huggingface.co at test time; the bge-micro-v2 URLs are pinned to revision `f09f671`.

### 3.2 SmartComponents.LocalEmbeddings

- The last NuGet release is 0.1.0-preview10148 (2024-04-04). The original repository (dotnet-smartcomponents/smartcomponents) now states that it moved to dotnet/smartcomponents.
- At dotnet/smartcomponents `bd1c9e7` (2025-01-28), `LocalEmbedder` wraps the Semantic Kernel `BertOnnxTextEmbeddingGenerationService` and the version prefix is 0.2.0. No 0.2.0 package is on NuGet, so the published package and the current source differ: the published package references FastBertTokenizer 0.4.67 directly.
- The README presents the components as sample implementations. The package downloads a model at build time; it does not contain one.

### 3.3 FastBertTokenizer

- Managed, Native AOT compatible (README and an AOT test app in the repo), no native dependency. Semantic Kernel and SmartComponents depend on it.
- Scope is WordPiece. The README states that pair encoding with `token_type_ids` for cross-encoders is not supported.
- Evidence is the strongest of the managed options surveyed: a differential test against Rust `tokenizers` (crate version 0.23 in `src/HuggingfaceTokenizer/RustLib`) over a Wikipedia corpus, with known divergences commented in the test (for example Unicode-version differences), plus published BenchmarkDotNet results from a named CI run and hardware.
- Release cadence is slow but recent: 1.0.28 (2024-04-30), 1.1.30-alpha (2025-03-03), 1.2.6 (2026-09-14). The 1.2 release commits add reproducible packages and release provenance.

### 3.4 ErgoX.TokenX (ErgoX.TokenX.HuggingFace)

- The section 3.1 name "ErgoX.TokenX" is not a NuGet ID. The published IDs are `ErgoX.TokenX.HuggingFace`, `ErgoX.TokenX.HuggingFace.Runtime.iOS` and `ErgoX.TokenX.HuggingFace.Runtime.Android`, each with one version, 0.22.1, published 2025-11-01.
- The nuspec repository element points to `vecrax/tokenizers` at `d9518b13579526b694a4c5267db7c67597283fe7`; that repository could not be cloned (credentials requested, so it is private, renamed or deleted). The source checked is ergosumx/tokenx, whose last commit is 2025-11-01.
- Web search reports the repository as archived on 2026-02-25 **(unverified)**.
- In the main package the native files sit at `runtimes/<rid>/native/<rid>/native/`, and include static libraries (`.a`, `.lib`) and a `.pdb`; the package is 38.9 MB. Only linux-x64 and win-x64 binaries are in it. Whether NuGet's runtime asset resolution picks up the nested path was not tested.
- The README states 2,500+ Python-to-.NET parity tests with SHA-256 verification, chat templates, offsets, truncation and padding. The README also compares itself with Microsoft.ML.Tokenizers; that comparison is the project's own claim and was not reproduced.

### 3.5 Tokenizers.DotNet

- Thin binding: `Tokenizer(path)`, `Encode(string) -> uint[]`, `Decode(uint[])`, plus a hub download helper. No offsets, attention masks, pairs or batch API in the managed surface.
- Native code ships in per-RID runtime packages that the consumer installs separately. macOS packages first appeared in 1.4.0 (2026-02-27). From 1.4.x the managed assembly is strong-name signed (README).
- Tests cover API use, FFI, thread safety and object lifetime. No token-ID parity corpus was found.
- FastBertTokenizer's published benchmark includes this package as a comparison point.

### 3.6 Tokenizers.HuggingFace

- Binds the Rust crate through a protobuf message layer (`tokenizers_proto` submodule at `14fbb05980a4f2ea64032782e4cd431181e118d9`, crate `tokenizers = "0.23.1"`). Depends on Google.Protobuf.
- Exposes normalizers, pre-tokenizers, encode/decode, loading from file and training. Pre-tokenizer splits can return offsets with a chosen offset type.
- No test project was found in the .NET repository.

### 3.7 Tokenizers.NET

- Listed versions are all alpha builds named by commit; the last is 2025-06-21. Targets net9.0 only.
- No license is declared in the nuspec and no LICENSE file is in the repository; reuse terms are therefore unclear.
- The README states zero GC allocations; no published benchmark was found to check it.

### 3.8 Microsoft.ML.Tokenizers

- Managed, MIT, no native dependency. 3.0.0 previews removed the Google.Protobuf dependency (#7587, 2026-03-11).
- Tokenizers are constructed per algorithm or per model (`BertTokenizer.Create`, `TiktokenTokenizer.CreateForModel`, `LlamaTokenizer.Create` and similar). A generic loader that interprets an arbitrary `tokenizer.json` pipeline was not found in `src/Microsoft.ML.Tokenizers` at `5a44dfd`.
- Model-name mappings continue on main: gpt-5.1 (#7556), gpt-5.2 (#7571), gpt-5.3 (#7579), gpt-5.4 (#7591), gpt-5.5 and gpt-5.6 (#7661), gpt-6 (#7693). Each ships only when the next package is published.
- The `release/5.0` branch, which contains the 2.0.0 source commit efefa92, does not contain the gpt-5.1 commit; its last tokenizer-related commit is #7494 (2025-08-11).
- ElBruno.LocalEmbeddings is a downstream consumer (BertTokenizer from `vocab.txt`).

### 3.9 Other ONNX Runtime based embedding packages

- **ElBruno.LocalEmbeddings** is the most actively released embedding package found: 32 versions since February 2026, Microsoft.Extensions.AI integration, NPU companion packages, OpenTelemetry instrumentation, and a BenchmarkDotNet project with a committed baseline. Model files are fetched from the Hugging Face Hub at run time.
- **SentenceTransformers** (Curiosity) embeds model files in per-model packages and includes a pure managed encoder path for Harrier. Its strongest parity evidence is an opt-in test that reproduces a model-card score matrix.
- **Microsoft.ML.OnnxRuntimeGenAI** (latest 0.17.1, 2026-09-29) includes its own tokenizer for generation models. It is in the generation lane, not the encoder lane, and is covered by plan section 6.4.
- Older packages found and not surveyed further: BERTTokenizers (last 1.2.0, 2022-09-09) and AllMiniLmL6V2Sharp (last 0.0.3, 2024-12-06).

## 4. Microsoft.ML.Tokenizers release-table verification

Plan section 6 table against NuGet registration data and the dotnet/machinelearning history (clone of main at `5a44dfdbf20d1b64211da313dfbaa9ffbf63ea5d`, 2026-10-01).

| Plan row | Plan value | Verified value | Result |
| --- | --- | --- | --- |
| Stable 1.0.0 | 2024-11-14 | 1.0.0 published 2024-11-14 (source commit 972c994) | Matches |
| Servicing 1.0.1, 1.0.2, 1.0.3 | 2025-01-15, 2025-02-26, 2025-10-28 | 2025-01-15, 2025-02-26, 2025-10-28 | Matches |
| Stable 2.0.0 | 2025-11-11 | 2.0.0 published 2025-11-11 (source commit efefa92, which does not contain #7556) | Matches |
| PR #7556 opened | 2025-11-18 | PR head commits authored 2025-11-18 (63a3bca, aace8b8); merged to main as de4eba2 on 2025-11-19 | Matches; add the merge date |
| Next package of any kind, 3.0.0-preview.26160.2 | 2026-03-12 | 3.0.0-preview.26160.2 published 2026-03-12 (source commit def7a4a, which contains #7556) | Matches |
| Request for HF BPE format support, issue #6901 | Opened late 2023; milestone ML.NET 5.0 | Not checkable: github.com issue pages and the API are blocked. No commit on any branch references #6901. | Not verified |
| "No stable release after 2.0.0" (checked 2026-10-04) | No stable after 2.0.0 | Still true on 2026-10-05 | Matches |

Rows to add or correct:

| Event | Date | Ref |
| --- | --- | --- |
| PR #7556 merged to main (commit de4eba2d38c0561ddb32851e12dd55850e5fc8a7) | 2025-11-19 | git history |
| First package containing #7556: 3.0.0-preview.26160.2 (source commit def7a4a0341070ae4f2b00ec3062069b7d2307b3) | 2026-03-12 | S14, nuspec |
| 3.0.0-preview.26457.2 (source commit 2fc13bab635827acdb4fb4ba3bef2b78673cd215; includes gpt-5.5/5.6, gpt-6 mappings and the Unigram `tokenizer.json` factory) | 2026-09-09 | S14, nuspec |
| No 2.0.x servicing release; `release/5.0` lacks #7556 | as of 2026-10-05 | S14, git history |

Effect on the plan's conclusion: the merged fix waited 113 days (2025-11-19 to 2026-03-12) for a preview package and has no stable package 320 days after merge. The plan's "about four months" is consistent with the preview gap. The sentence "the listing was read through a search index" can now be replaced by the registration API and git evidence above.

## 5. Gaps this project fills

These are planned properties of this organization's work (plan sections 6, 10 and 11), not delivered results. Each states the difference from what the surveyed options publish today.

1. **`tokenizer.json` interpretation in managed code.** Among managed options, FastBertTokenizer reads `tokenizer.json` for WordPiece only and Microsoft.ML.Tokenizers uses per-model factories. Full `tokenizer.json` support exists today only through Rust bindings, which bring native binaries per RID. ADR-15 plans a managed interpreter for the components the pilot profiles use, with unsupported operators failing explicitly.
2. **Conformance evidence published per model profile.** FastBertTokenizer has a corpus-level differential test for BERT; Semantic Kernel compares embeddings for two models; ErgoX claims a Python parity suite. None publishes a per-profile record tying tokenizer version, asset revision and reference revision to a protected corpus that covers offsets, special tokens, pairs, truncation and padding. Plan section 6.2 requires that record and the protected corpus.
3. **Declared offset semantics.** Microsoft.ML.Tokenizers returns UTF-16 `Range` offsets; the Rust bindings expose the crate's offsets; the embedding packages expose none. The plan requires declaring the unit (UTF-8 bytes, scalars or UTF-16) and testing conversion on non-ASCII text.
4. **Release cadence decoupled from the .NET cycle.** Microsoft.ML.Tokenizers ships stable once a year; the Rust bindings ship when their maintainers rebuild against a new crate; the embedding packages pin tokenizer and ONNX Runtime versions that lag current releases (Semantic Kernel pins FastBertTokenizer 1.0.28 and ONNX Runtime 1.23.2). The plan commits to an independent tokenizer release lane with a release drill (P-025) proving a fix can ship without an upstream release.
5. **Benchmark method.** FastBertTokenizer and ElBruno.LocalEmbeddings publish BenchmarkDotNet numbers. The plan adds the C# versus Python same-backend comparison as the headline, allocation counts, cold and warm results, sample counts and confidence intervals, and publication of unfavorable results (section 11).
6. **Model bundle identity.** The embedding packages either take files from the caller or download from the Hub at a moving revision (SmartComponents' documented example uses `resolve/main`). The plan pins every asset by revision and records tokenizer components in the bundle (sections 6 and 9).

Where the surveyed options already do well and this project should not claim a difference: FastBertTokenizer's WordPiece speed and differential testing, Rust bindings' breadth of `tokenizer.json` coverage, and the Microsoft.Extensions.AI integration already shipped by Semantic Kernel and ElBruno.LocalEmbeddings.

## 6. Open items

1. **First named consumer.** Settled 2026-11-01 (#208, decided 2026-11-01 by the maintainer's delegate (agent), revisitable): starlinghq/roost and starlinghq/corvid, which consume the published `0.3.0` packages from nuget.org (pinned in each one's `Directory.Packages.props`), are the first named consumers (`docs/kickoff.md`). They are the organization's own projects, not independent outside demand. Outside demand stays scored from observable signals as plan section 3.1 says, and an outside consumer is added if one agrees; candidates can be drawn from the downstream users above, for example projects that use the Semantic Kernel ONNX connector or FastBertTokenizer for BERT embeddings, since the P0 encoder is in the same lane.
2. **Issue #6901 status and milestone.** Not verifiable from this environment. Re-check from a network that reaches github.com.
3. **ErgoX.TokenX archive status and source repository.** Archive date is search-only; the nuspec source repository `vecrax/tokenizers` is not reachable.
4. **Measurements not run.** No benchmark or parity test from any surveyed project was executed here. Comparisons in section 2 restate what each project publishes. Running FastBertTokenizer's and the bindings' tests against the P0 encoder's tokenizer would give a like-for-like baseline once huggingface.co is reachable.
5. **Update plan section 6.** Carry the corrected rows from section 4 into the next plan revision and replace the search-index caveat with the sources used here.
6. **Re-verify before publishing.** Versions in this note will drift; re-run the registration queries when P-028 closes, since the plan requires versions verified at that point.
