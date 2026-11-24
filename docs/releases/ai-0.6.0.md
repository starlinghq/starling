# ai-dotnet release 0.6.0

**OCR joins the publishable set, the tokenizer moves to 0.3.6, and two BERT profiles, an in-process memory opt-in and safer Qwen3.5 tool results.** This release follows `0.5.0` (published 2026-11-03 from tag `ai-v0.5.0`) and carries everything merged to `main` since. The publishable set grows from 27 to **28 packages**: `Starlings.Models.PaddleOcr` (PP-OCRv5 mobile OCR, Experimental) is new. No package leaves it. Nothing is removed from the API and nothing breaks at compile time in a published package. Two behaviour changes need a look: `Qwen35ChatClient` escapes tool results by default, and `Starlings.Models.Bert` runs with ONNX Runtime's CPU arena off. Caches of two hosted bundles download one file again (QwenChat r2, Qwen3.5 int4 r3). See [Upgrading](#upgrading-from-050).

**Publishing.** This is a release candidate. The `release` workflow publishes it from the tag `ai-v0.6.0` on `main`, after a maintainer approves the `release` environment, through NuGet trusted publishing, and creates a GitHub release with this record as notes. Until then nothing in this record is on nuget.org.

**Tokenizer.** Every package that tokenizes pins `Starlings.Tokenizers [0.3.6]` (0.3.4 in `0.5.0`). See [Tokenizer](#tokenizer).

**Maturity.** Maturity belongs to profiles, not packages, and each package README has a generated maturity table. Every Stable profile of `0.5.0` is still Stable with the same contract: no gate, tolerance or `protected/` file changed. The one contract change is a relaxation: `gliner2-base-v1/onnx-fp32-cpu` no longer requires `MALLOC_MMAP_THRESHOLD_` on Linux (#178). New registry profiles, all Experimental with every gate not run:

| Profile | Maturity | Package | Admission |
| --- | --- | --- | --- |
| `ppocrv5-mobile/onnx-fp32-cpu` | Experimental | `Starlings.Models.PaddleOcr` (**new, published**) | `model-pipeline/admissions/ppocrv5-mobile.md` (Mark Walls, 2026-11-23) |
| `paraphrase-multilingual-minilm-l12-v2/onnx-fp32-cpu` | Experimental | `Starlings.Models.Bert` (published) | #182 |
| `gte-reranker-modernbert-base/onnx-fp32-cpu` | Experimental | `Starlings.Models.Bert` (published) | #101, #102 |

The [catalog](../catalog/README.md) links every gate to its evidence.

## Packages

All packages target `net10.0` and have SourceLink to the commit. All are licensed MIT OR Apache-2.0, except `Starlings.Models.Lfm3`, whose NuGet license is the LFM Open License v1.0 file (commercial use licensed only below USD 10 million annual revenue; see [`ai-0.4.0.md`](ai-0.4.0.md#new-package-starlingsmodelslfm3-lfm3-17b-experimental-217)). Sibling and tokenizer dependencies are exact (`[version]`).

| Package | Version | Change since 0.5.0 |
| --- | --- | --- |
| `Starlings.Tokenizers` | 0.3.6 | dependency: raised from 0.3.4 (0.3.5 adds pre-tokenizer support and three profiles; 0.3.6 declares AOT compatibility); outputs of existing profiles unchanged |
| `Starlings.Runtime.Onnx` | 0.6.0 | `OnnxSessionOptions.ApplyGlibcMmapThreshold`, `GlibcMallocThreshold`, `OnnxSessionDiagnostics.GlibcMmapThresholdDiagnostic` (opt-in, #178); README maturity table lists `ppocrv5-mobile/onnx-fp32-cpu` |
| `Starlings.Acquisition` | 0.6.0 | README maturity table lists the new BERT and PP-OCRv5 profiles |
| `Starlings.Models.Abstractions` | 0.6.0 | `ModelProfileCapabilities.Generation` (`ProfileGeneration`, Experimental, #177); README links the reranking sample |
| `Starlings.Models.E5` | 0.6.0 | `E5EncoderOptions.MaxSequenceLength` bounded to 2 to 512 (throws `ArgumentOutOfRangeException` outside) |
| `Starlings.Models.E5.Extensions.AI` | 0.6.0 | the `MaxSequenceLength` bound, through `E5EmbeddingGenerator.Create` |
| `Starlings.Models.E5.MLNet` | 0.6.0 | the `MaxSequenceLength` bound, through the transform loaded from files |
| `Starlings.Runtime.OnnxWeb` | 0.6.0 | rebuilt, no change |
| `Starlings.Runtime.OnnxWeb.WebGpu` | 0.6.0 | rebuilt, no change |
| `Starlings.Models.E5.Browser` | 0.6.0 | the `MaxSequenceLength` bound, checked before any download |
| `Starlings.Models.Gliner2` | 0.6.0 | CPU profile drops the `MALLOC_MMAP_THRESHOLD_` deployment requirement (#178; CUDA keeps it); tokenizer 0.3.6 |
| `Starlings.Generation.Abstractions` | 0.6.0 | `ChatRouteContract`, `ChatBundleFile`, `ChatModelDefinition` (Experimental, #109) |
| `Starlings.Generation.OnnxGenAI` | 0.6.0 | `GenAIChatModel`, a generic text chat loader (Experimental, #109) |
| `Starlings.Models.Qwen35` | 0.6.0 | tokenizer 0.3.6; no API change |
| `Starlings.Models.Qwen35.OnnxGenAI` | 0.6.0 | int4 graph at hosting revision r3 (#294; **an r2 int4 cache downloads one file again**); `structured_output` experimental on int4 |
| `Starlings.Models.Qwen35.Extensions.AI` | 0.6.0 | **behaviour change**: tool results escaped by default (`Qwen35ChatClientOptions.EscapeToolResults`, `Qwen35ChatRequestMapper.EscapeToolResult`, new `MapMessages` overload) |
| `Starlings.Models.EmbeddingGemma2` | 0.6.0 | tokenizer 0.3.6; README memory notes (#178) |
| `Starlings.Models.EmbeddingGemma2.Extensions.AI` | 0.6.0 | rebuilt, no change |
| `Starlings.Models.EmbeddingGemma2.MLNet` | 0.6.0 | rebuilt, no change |
| `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` | 0.6.0 | rebuilt, no change |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` | 0.6.0 | README: the LGPL rebuilds the gates use (#277) |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` | 0.6.0 | README: the LGPL rebuilds the gates use (#277) |
| `Starlings.Models.Kev` | 0.6.0 | tokenizer 0.3.6; no API change |
| `Starlings.Models.Kev.Onnx` | 0.6.0 | README links the in-process `MALLOC_MMAP_THRESHOLD_` opt-in (the requirement stays, #178) |
| `Starlings.Models.QwenChat` | 0.6.0 | tokenizer 0.3.6; no API change |
| `Starlings.Models.QwenChat.OnnxGenAI` | 0.6.0 | hosting revision r2 (#278; **new `model.onnx` digests, an r1 cache downloads one file again**) |
| `Starlings.Models.Bert` | 0.6.0 | built-in profiles `paraphrase-multilingual` and `gte-reranker-modernbert-base` (Experimental, #182, #101); `BertReranker.Create(maxTokens)`; **behaviour change**: CPU arena off on every profile (`Starlings.Models.Bert.CpuMemoryArena` switch) |
| `Starlings.Models.Lfm3` | 0.6.0 | generation fields in its capability record (#177); license unchanged (LFM Open License v1.0) |
| `Starlings.Models.PaddleOcr` | 0.6.0 | **new in the publishable set** (PP-OCRv5 mobile OCR, Experimental, #223) |

README-only edits (the "star Starling on GitHub" line, tokenizer 0.3.6 in the tokenizing packages' contract tables) are not listed above.

**Publishable set (28 packages):** every row at 0.6.0. Nothing is held (`HELD` is empty in `.github/workflows/release.yml`). `Starlings.Models.PaddleOcr` is in the release drills (`tools/release-drill-ai.sh`, `.ps1`; dependencies `Starlings.Acquisition`, `Starlings.Models.Abstractions`, `Starlings.Runtime.Onnx`, no tokenizer), the Native AOT smoke list and the publish order.

**Not published:**

- `Starlings.Models.Qwen3Embedding` and `Starlings.Models.Qwen3Embedding.Extensions.AI`: source only, pending the reranker decision (#144, #100), as in [`ai-0.5.0.md`](ai-0.5.0.md#packages).
- `Starlings.Models.Whisper`: packable, not in the drill's package set (its description lacks the per-profile maturity line), for the reasons in [`ai-0.4.0.md`](ai-0.4.0.md#packages).
- `Starlings.Guardrails`, `Starlings.Guardrails.Extensions.AI` and `Starlings.Models.Kev.Guardrails` (new, Experimental): packable, not in the drill's package set.
- `Starlings.Models.Phi4` (new, #109), `Starlings.Interventions`, `Starlings.Models.Lfm3.Interventions` and `Starlings.Models.Clover`: source only.

## Changes merged since 0.5.0

Everything below was merged to `main` after `0.5.0` (moved here from `docs/releases/unreleased.md`, with the guardrails, memory opt-in, Whisper detection and evidence entries that were merged without one).

### New package: Starlings.Models.PaddleOcr (PP-OCRv5 mobile OCR, Experimental; #223, #345)

`Starlings.Models.PaddleOcr` joins the publishable set. It runs PP-OCRv5 mobile OCR on ONNX Runtime 1.30.0, CPU: the official PaddlePaddle ONNX exports of the DB text detector (`PaddlePaddle/PP-OCRv5_mobile_det_onnx` at `e6f4fa85`) and the CTC line recognizer (`PaddlePaddle/PP-OCRv5_mobile_rec_onnx` at `ed152b8b`), both Apache-2.0, with owned C# pre- and post-processing and no OpenCV. It was packable but held out of the 0.5.0 release set. Mark admitted it at Experimental on 2026-11-23 (`model-pipeline/admissions/ppocrv5-mobile.md`). Its dependencies are `Starlings.Acquisition`, `Starlings.Runtime.Onnx` and `Starlings.Models.Abstractions`. It needs no tokenizer: the recognizer's dictionary comes from its pinned `inference.yml`. The release drills, the Native AOT smoke list and the publish order include it.

Registry profile `ppocrv5-mobile/onnx-fp32-cpu`, Experimental, every gate not run. The evidence is the onboarding route's development run in #228: 487 checks passed on ten rendered images, English and Simplified Chinese. The manifest schema gains the task `ocr` and, for OCR profiles only, `processing.dictionary` in place of `processing.tokenizer`. `PaddleOcrCapabilities` adds `batch`, `streaming`, `persistence` and `mlnet`, all unsupported, the rows every manifest states. No other status changes. Not covered: CUDA, Windows, Traditional Chinese and Japanese, and text-line orientation.

### Tokenizer

Every package that tokenizes now pins `Starlings.Tokenizers [0.3.6]` (0.3.4 in `0.5.0`), and every manifest's `processing.tokenizer.version` moved with the pin. `0.3.5` adds the `WhitespaceSplit` pre-tokenizer, the o200k `Split` regex and three experimental profiles (`phi4-bpe-chat`, `xlmr-unigram-paraphrase-multilingual`, `modernbert-bpe-gte-reranker`); of these, `xlmr-unigram-paraphrase-multilingual` and `modernbert-bpe-gte-reranker` are used by a package, by the two BERT profiles below. Outputs of existing profiles are unchanged: the conformance fixtures they have pass unchanged, the profiles without fixtures reach no code that changes their output, and no model was rerun (`docs/support-policy.md`, "Tokenizer bumps", option B). A tokenizer with an `rstrip` added token and, in the same matching pass, an added token that starts with whitespace now fails at load instead of diverging from upstream; no pinned asset has that combination.

`0.3.6` changes no tokenizer code: the package now declares Native AOT and trimming compatibility (`IsAotCompatible`), and the tokenizer test suite passes as a Native AOT binary. Every profile's conformance fixtures pass unchanged, so this step also rests on option B and no model was rerun. An application published with Native AOT loads every pinned tokenizer faster than Python's `tokenizers` (starlinghq/tokenizers `docs/benchmarks/tokenizer-aot-2026-11-14.md`).

### Starlings.Runtime.Onnx: in-process `MALLOC_MMAP_THRESHOLD_` opt-in; GLiNER2 CPU no longer needs the variable (#178)

- **New, opt-in, off by default.** `OnnxSessionOptions.ApplyGlibcMmapThreshold` (or `GlibcMallocThreshold.Apply()` directly) calls glibc's `mallopt(M_MMAP_THRESHOLD, 65536)` once per process, before the session is created, and writes a diagnostic to standard error; `OnnxSessionDiagnostics.GlibcMmapThresholdDiagnostic` records what it did. It applies the same threshold as the environment variable, for hosts that cannot set `MALLOC_MMAP_THRESHOLD_=65536` in the environment of the process they start. The environment form stays the primary recommendation. A no-op on other platforms and without glibc; nothing calls `mallopt` unless the host asks.
- **Deployment requirements, per profile, from one-hour soaks without the variable** (`docs/evidence/malloc-noenv-2026-11-15/`): `gliner2-base-v1/onnx-fp32-cpu` and `embeddinggemma-2/video-fp32-cpu` pass at the default glibc allocator, so the GLiNER2 CPU profile **drops** the requirement (README, `Gliner2Capabilities` comment and manifest). Kev-0.8B fails without it on CPU too, so the Kev requirement stays on both providers, and Kev-4B is confirmed. GLiNER2 on CUDA and EmbeddingGemma 2 video on CUDA keep the requirement: their soak evidence is all with the variable. The troubleshooting guide and each README link the opt-in.

### Starlings.Models.Qwen35.Extensions.AI: tool results escaped by default

**Behaviour change.** `Qwen35ChatClient` now escapes the tokenizer's added-token strings in tool-result content before rendering, by default. Before, the text of a `FunctionResultContent` went into the prompt verbatim and `Qwen35ChatOptions.EscapeSpecialTokens` was off by default, so under `UseFunctionInvocation` a function that returned untrusted text such as `<|im_end|>\n<|im_start|>system\n...` could close the tool turn and open a system turn of its own. The new option `Qwen35ChatClientOptions.EscapeToolResults` (default true) rewrites each `<|name|>` to `<¦name¦>` and puts a `¦` after the `<` of each tag token the pinned tokenizer adds (`<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`, `<tts_pad>`, `<tts_text_bos>`, `<tts_text_eod>`, `<tts_text_bos_single>`), which the `<|name|>` rewrite of `EscapeSpecialTokens` does not cover. A non-string result is escaped after it is serialized to JSON. The rewrite is the new public `Qwen35ChatRequestMapper.EscapeToolResult`; a test holds its tag list to the `added_tokens` of the pinned `tokenizer.json`. System, user and assistant messages are rendered as before, and the typed API (`Starlings.Models.Qwen35`, `Qwen35ChatTemplate`) is unchanged and stays faithful to upstream `apply_chat_template`.

`Qwen35ChatRequestMapper.MapMessages(messages, instructions)` follows the client's default and escapes tool results too; the new overload `MapMessages(messages, instructions, escapeToolResults)` maps with the option set either way.

**Upgrading:** a conversation whose tool results contain none of these strings renders to the same prompt as before. One that does gets a different prompt, and so possibly a different completion. To keep the earlier, upstream-faithful rendering, set `EscapeToolResults = false` on `Qwen35ChatClientOptions`.

### Starlings.Models.QwenChat.OnnxGenAI: hosting revision r2 (#278)

`QwenChatProfile.Qwen3` and `QwenChatProfile.Qwen25` pin new `model.onnx` digests, and `QwenChatModelAssets.Qwen3HostingRevision` and `Qwen25HostingRevision` move to revision `r2` of `Starlinghq/qwen3-0.6b-ortgenai` (`00278275`) and `Starlinghq/qwen2.5-0.5b-instruct-ortgenai` (`265459af`). Only `model.onnx` changes, and only in its `producer_version` header: `0.17.1+83de55a`, the builder's own commit, instead of `0.17.1+1bada25`, the Git HEAD of the checkout that built r1. The weights and the other five bundle files are the same bytes, the sizes are the same, and the maturity of both profiles stays Experimental.

`build_bundle.py` hides Git from the builder since #283, so a local build with the hash-locked builder gives the pinned digests again: two builds per profile on 2026-11-04, one from an environment inside a Git work tree and one from outside any, on one machine. With the new pins the onboarding `dotnet` stage, which failed with exit 70 in qualification, passes for both profiles, and development parity passes on all 14 corpus rows. Evidence: `docs/evidence/qwenchat-hosting-r2-2026-11-04/README.md`.

**Upgrading:** r1 stays hosted at its commits (`c24cd59c`, `6b6957c8`), so packages up to 0.5.0 keep working unchanged. After upgrading the package, a cache acquired from r1 fails `QwenChatModelAssets.OpenLocal` with `ModelAssetIntegrityException` on `model.onnx`, at the default SHA-256 verification; call `DownloadAsync` once, which replaces that one 150 to 250 KB file. A cache opened with `CacheVerification` set to size only passes `OpenLocal`, because the size is unchanged, and `QwenChatModel.Load` then refuses the bundle, so run `DownloadAsync` with SHA-256 verification. An r1 bundle kept in a directory of its own is refused by `Load`.

Both revisions use one cache path, because the path is keyed on the upstream revision. So an app on 0.5.0 and a newer app that share a cache directory invalidate each other, and rolling back hits the same thing: each `DownloadAsync` replaces `model.onnx` with its own revision, and the other package then fails as above until it downloads again. Give the two apps separate cache directories.

The same stamp exposure in the Qwen3.5 int4 graph (`c98b9fc`) is fixed by hosting revision r3 (below).

### Starlings.Models.Qwen35.OnnxGenAI: hosting revision r3 for the int4 graph (#294)

`build_ortgenai_bundle.py` now hides Git from the builder the way `build_bundle.py` has since #283, and refuses a graph whose `producer_version` is not the pinned wheel commit `0.17.1+83de55a`. Hosting revision r3 of `Starlinghq/qwen3.5-0.8b-ortgenai` (`4651f7e6fd8c0421c4e248b5c82a1d7f25374177`, tag `r3`) replaces `ortgenai-int4/model.onnx` with a rebuild from the pinned snapshot at that stamp; its digest is the M0 build's `766e49f3…`. Every other file of every variant keeps its r2 bytes, r2 (`22477cdb…`) stays reachable, and the `ortgenai-int4-cpu` manifest, `Qwen35ModelAssets` and the asset tests pin r3. The fp32x and fp16x graphs were assessed: their stamps are part of digests that reproduce byte-for-byte on this machine, so they carry no checkout-HEAD exposure and their hosting pins stay at r2.

**Upgrading:** packages up to 0.5.0 pin r2 by commit and keep working. A cache acquired from r2 fails `Qwen35ModelAssets.OpenLocal` on `ortgenai-int4/model.onnx` (SHA-256, 364,388 bytes); one `DownloadAsync` of the int4 lock replaces it. The same shared-cache-path caveat as the QwenChat r2 applies between r2 and r3 of this repository.

### Qwen3.5 int4 guided decoding

`structured_output` on the Experimental profile `qwen3.5-0.8b/ortgenai-int4-cpu` moves from untested to experimental. A regex, a JSON schema and the required tool grammar were followed on int4 on CPU, through the typed API and the `IChatClient`, on the hosted r2 files ([evidence](../evidence/qwen35-int4-guided-2026-11-04/README.md)). No code path changed. `tool_calling` and `extensions_ai` stay untested on int4, and the profile still has no tolerances or task-quality floor.

### BERT route: paraphrase-multilingual-MiniLM-L12-v2, Experimental (#182)

- `Starlings.Models.Bert` adds an eighth built-in profile, `BertModelProfile.BuiltIn("paraphrase-multilingual")` (`paraphrase-multilingual-minilm-l12-v2/onnx-fp32-cpu`): `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at `e8f8c211`, Apache-2.0, XLM-R Unigram tokenizer with BERT segment ids, mean pooling, no L2 (normalize the vectors before cosine or dot-product comparison), 128 tokens, 384 wide, CPU fp32. Experimental with every gate `not-run`: the evidence is a development comparison on the 54-case proposed corpus, with token ids exact and the pooled embedding within 7.6e-7 of the PyTorch reference (bound 1e-5). This is the first end-to-end .NET run of the `WhitespaceSplit` pre-tokenizer and the bare `Precompiled` normalizer in a model, and its tokenizer profile `xlmr-unigram-paraphrase-multilingual` is one of the two 0.3.5 profiles a package uses; its fixtures are proposed. Windows is untested. The profile needs `Starlings.Tokenizers` 0.3.5, so the published `0.5.0` package does not have it.
- `model-pipeline scaffold` takes `--token-type-ids` / `--no-token-type-ids` to state whether the graph takes `token_type_ids`; without either flag an existing entry keeps its value, and a value that contradicts the entry is refused.

### BERT route: gte-reranker-modernbert-base, Experimental (#101, #102)

- `Starlings.Models.Bert` adds a ninth built-in profile, `BertModelProfile.BuiltIn("gte-reranker-modernbert-base")` (`gte-reranker-modernbert-base/onnx-fp32-cpu`): `Alibaba-NLP/gte-reranker-modernbert-base` at `f7481e60`, Apache-2.0, the first ModernBERT model. Two-input graph as published upstream, BPE tokenizer profile `modernbert-bpe-gte-reranker`, raw logits through `IReranker`, 8,192 tokens. Onboarding evidence stops at 1,468 tokens. Experimental; the references are proposed and no gate has run.
- `BertReranker.Create` takes `maxTokens` (Experimental, off by default), for all three reranker profiles: a limit per pair below the profile's, set once and applied on every call of that reranker. Text longer than it is truncated, longest side first from the right, with the ids of transformers `truncation=True, max_length=<limit>`; a longer pre-tokenized row given to `ScoreTokens` is refused with `ArgumentException`. This is how the profile treats its own limit. `BertReranker.MaxTokens` reports the limit in effect.
- **Behavior change on all nine BERT profiles:** `Starlings.Models.Bert` now runs with ONNX Runtime's CPU arena off. On .NET, one 8,192-token gte-reranker-modernbert-base pair at four threads kept 18.4 GiB resident from its second run on with the arena; without it the peak is 9.6 GiB and resident memory returns to 0.9 GiB after each run. Scores are unchanged. The cost is latency, and on .NET it is small. With processes alternating the setting and no CI jobs on the machine, one 8,192-token pair took about 6 percent longer without the arena, a 2,048-token pair at two threads about 3 percent, and 8 rows of about 60 tokens on bge-base showed no difference. That is two processes of each setting on one CPU, for gte-reranker-modernbert-base and bge-base. A Python probe on a busier machine showed more: 54 to 56 s against 45 to 53 s at 8,192 tokens, and 5.2 s against 4.0 s at 2,048. This package does not use `DedicatedRunThreads`, unlike EmbeddingGemma 2. Without the arena, each run's buffers are freed into the calling thread's glibc heap, which can keep up to 64 MiB resident per heap, so a host calling from a rotating thread pool can spread that memory over more heaps. That effect was measured on EmbeddingGemma 2, not here. The Experimental `AppContext` switch `Starlings.Models.Bert.CpuMemoryArena` turns the arena back on for the process. Evidence: `docs/evidence/gte-reranker-modernbert-onboarding-2026-11-04/`.
- `BertSession` replaces the truncation and Fixed padding of 8000 that this `tokenizer.json` declares with no padding and the profile's limit, as it already did for every profile; `ModernBertTokenizerTests` now pins that.
- `model-pipeline inspect` suggests `bert-onnx-cpu-v1` for a `modernbert` config with a BPE tokenizer, and no longer reports the GPT-2 split or `lstrip` added tokens as unsupported, which `Starlings.Tokenizers` has implemented since 0.3.2 and 0.3.1.

### Starlings.Models.E5: `MaxSequenceLength` bounded to 2 to 512

`E5EncoderOptions.MaxSequenceLength` now has to be between 2 and `E5Encoder.MaxSupportedSequenceLength` (512); outside that range `E5Encoder`, `E5Encoder.Create`, `E5TextEmbedder.Create`, `E5EmbeddingGenerator.Create`, the ML.NET transform loaded from files, `E5BrowserEncoder` and `E5BrowserEmbedder.LoadAsync` throw `ArgumentOutOfRangeException` before the model is loaded (the browser loader before any download). The value is also the tokenizer's truncation length, so a larger one such as 1024 let text tokenize to more positions than the XLM-R position table holds, and the sequence reached the graph; a value of 1 passed the options and then failed in `E5Tokenizer` with a less direct error. The default (512) and every value in range behave as before, and no model output changes.

### Generation fields in the capability record

`ModelProfileCapabilities.Generation` (`ProfileGeneration`, new and Experimental API) carries tool-call mode, structured output, state reuse, caching style and max output tokens for the seven generation profiles, generated from the new manifest section `behavior.generation` (#177). The fields are informational until a release note promotes them (`docs/kickoff.md`, "Capability record fields"). No capability status changes. The validator and the contract tests hold each value to its capability row, so `untested` appears where a row is untested or absent: tool calls on Qwen3.5 int4, structured output on the QwenChat profiles and max output tokens on LFM3. An `untested` value also blocks Stable, so these appear as `generation.<field>` in the catalog's Stable readiness column; those profiles are all Experimental, and no Stable profile has one.

### Phi-4-mini-instruct: `genai-chat-cpu-v1` route, admitted at Experimental (#109)

A data-driven GenAI chat onboarding route, `genai-chat-cpu-v1`, with `microsoft/Phi-4-mini-instruct` as its one candidate, admitted at maturity Experimental by Mark on 2026-11-04. New experimental API for a package consumer: `GenAIChatModel` in `Starlings.Generation.OnnxGenAI`, a generic text chat loader, and `ChatRouteContract`, `ChatBundleFile` and `ChatModelDefinition` in `Starlings.Generation.Abstractions`. Existing API is unchanged, and those packages carry no model data. `Starlings.Models.Phi4` and the tool `Starlings.GenAIChat.Fidelity` are source only: no Phi-4 package, registry manifest or hosted bundle ships. Development parity passed on the 14 corpus rows, the longest 366 tokens, on CPU fp32 with Linux x64; protected G1/G2 did not run, and context from 4,096 tokens up stays `untested`. The route's design note is accepted for the route shape and the generic loader; hosting a bundle stays open. Evidence: `docs/evidence/phi4-mini-onboarding-2026-11-04/README.md`

### Guardrails packages: source only, not published in this release (Experimental; #335, #336, #341)

New `Starlings.Guardrails` (optional checks of prompts and responses: `IContentGuard`, the model-free `PatternGuard` with canaries, `CascadeGuard`, `GuardLevels` Light, Balanced and Strict), `Starlings.Guardrails.Extensions.AI` (middleware that runs a guard around any `IChatClient`) and `Starlings.Models.Kev.Guardrails` (a zero-shot model guard on the Kev decision models). All three are Experimental: detection quality is measured on small hand-written sets only (`docs/evidence/guardrails-2026-11-16/`, `docs/evidence/guardrails-l1b3rt4s-2026-11-16/`), and no level is a reliable defense on its own. They are packable but **not in the publishable set**: they are not in the release drill's package set. Build them from this repository.

### Starlings.Models.Whisper: a language is detected, not guessed (source only; #332, #333, #344)

Up to `0.5.0`, `WhisperOptions.Language = null` put `<|startoflm|>` in the prompt's language slot, a prompt the model was not trained on and not how transformers detects the language, with `language_detection` untested. #333 first made a null language throw; #344 then implemented detection the way transformers' `detect_language` does it. A null `Language` now means **detect**: `WhisperModel.DetectLanguage` runs one decoder step over `<|startoftranscript|>`, masks every logit outside the language-token range and takes the argmax, and `WhisperTranscriber` detects from the first 30 s window and reports the code on `WhisperTranscription.DetectedLanguage` (also `DetectLanguage(float[] waveform)` and `DetectLanguageMel`). An unknown language code still throws `WhisperUnsupportedLanguageException`. `WhisperModel.Prompt` takes a non-nullable `language` with no default and throws the same for null. `language_detection` is `experimental` in `WhisperCapabilities.All` and the `whisper-large-v3-turbo/onnx-fp16-cpu` manifest; detection is reference-verified for English only (the golden detection token of the fp32 reference), and a non-English reference clip is the follow-up. Explicit-language transcription is unchanged. The package is source only, so no published package changes.

### Catalog: `owner` key removed

`docs/catalog/catalog.json` loses the `owner` key of every profile, and `docs/catalog/README.md` loses its Owner column. `maintenance.owner` is gone from the manifest schema and from every registry manifest, and a manifest that still has it now fails validation: the maintainer removed the named owner and named consumer requirements on 2026-11-04 (`docs/kickoff.md`). The catalog format stays `starlings-catalog/1`, so a reader of `catalog.json` that uses `owner` must stop. No package, API, capability status or model output changes.

### Reranking sample

New sample `ai-dotnet/samples/E5RerankSearch`: multilingual-e5-small retrieves the top 5 from an in-memory `Microsoft.Extensions.VectorData` collection and `bge-reranker-v2-m3` reorders them with `Reranking.RerankAsync`, on the published 0.5.0 packages, with its lock file and recorded output. The helpers and the reranker profile stay Experimental: gates G0 to G7 have not run for `bge-reranker-v2-m3/onnx-fp32-cpu`. No code changes in any package; the `Starlings.Models.Abstractions` README links the sample, which ships with that package's next release.

### `dotnet new` template: local RAG (#306)

New template `ai-dotnet/templates/LocalRag` (#306): `dotnet new install ai-dotnet/templates/LocalRag`, then `dotnet new starling-rag -o MyApp` scaffolds the LocalRag console app outside the repository, on the published `0.5.0` packages (pin a different version with `--starlings-version`). It is the LocalRag sample verbatim — EmbeddingGemma 2 retrieval with the document/query prompt prefixes, Qwen3.5 answers through the `IChatClient` adapter, the persisted index cache — plus the same two demo documents. The native runtime is the application's choice as always: exactly one of the CPU natives (default) or the CUDA natives (`--native-runtime cuda`). CI installs the template, scaffolds it and builds the scaffold. The template is not a NuGet package and is not part of the publishable set; publishing it as a template package (versioning with the packages, the drill's `dotnet new` step) is maintainer scope and stays open on the issue.

### Qualification: CUDA regression runs locally; model tests fail when a module passes nothing

- **CUDA regression moves off RunPod.** The qualification workflow's `cuda (runpod)` job, its dispatch inputs `cuda_soak_duration` and `skip_cuda`, and `tools/ci/runpod.py` and `tools/ci/runpod-cuda-pod.sh` are removed; no workflow uses a GPU or the `RUNPOD_API_KEY` secret any more. CUDA qualification is now `tools/cuda-qualification-local.sh --gpu <index> <out dir>`, run by the maintainer on their RTX 3090 from a clean checkout before every `ai-v*` tag. It does the same work (the CUDA test projects, now including Kev's, G1, G2 and G3 of every Stable CUDA profile on its release sets, a soak smoke test per Stable CUDA profile) into the same output layout (`summary.md`, `summary.tsv`, `status.txt`, `environment.txt`, `exit-code`, per-module and per-profile results); the maintainer records `summary.md` and `environment.txt` in `docs/evidence/gpu-regression-<date>/`. It runs one step at a time, waits on `agent-budget` when memory is critical, and never sets `RUNNER_ENVIRONMENT`. The support policy, the workflow and tools READMEs, the benchmark protocol, the CUDA manifests' notes, the catalog and the package READMEs say so.
- **A skipped module is no longer green.** The `model-tests` job now runs `tools/ci/check-trx.py` on the TRX reports and fails when a test module with tests passed none of them (`tools/ci/trx-allow-zero.txt` lists exemptions with a reason; it is empty). The local CUDA script uses the same check.
- **More EmbeddingGemma 2 tests run in `model-tests`.** `tools/ci/eg2-test-inputs.sh` prepares the fixture image set (regenerated by `make_eg2_images.py` and checked against the protected `images.json`), the pinned FFmpeg 8.1 and 9.0 LGPL builds and the four release videos, and the job sets `STARLING_EMBEDDINGGEMMA2_IMAGES`, `STARLING_FFMPEG_DIR`, `STARLING_FFMPEG9_DIR` and `STARLING_EMBEDDINGGEMMA2_RELEASE_VIDEOS`. The image parity and preprocessing tests and the release-video decoder test now run instead of skipping. The 22 synthetic fixture videos still have no reproducible source, so the tests that need them still skip. No package, API or model output changes.

### Evidence and registry upkeep

- **CUDA concurrency on the RTX 3090 (#257, #342).** The four concurrent-call CUDA tests (e5 text, EmbeddingGemma 2 text, image and video) pass in 8 of 8 processes on an RTX 3090, every vector bit for bit equal to the sequential run (`docs/evidence/cuda-concurrency-2026-11-21/`). No status changes.
- **0.5.0 regression on the RTX 3090.** A local rerun of the Stable CUDA profiles and `embeddinggemma-2/video-fp32-cpu` on the `ai-v0.5.0` tag and on `main` found no output regression (`docs/evidence/regression-0.5.0-rtx3090-2026-11-06/`). Regression evidence, not a qualification round.
- **FFmpeg LGPL rebuilds (#277).** The pinned FFmpeg autobuilds the release gates used were deleted upstream; the gates now use LGPL rebuilds (`docs/evidence/embeddinggemma2-video-ffmpeg-rebuild-2026-11-04/`). The FFmpeg adapter READMEs state their limits (AVX2 measured, AVX-512 not).
- **LFM3 export drift recorded (#279, #343).** `LiquidAI/LFM3-1.7B-ONNX` moved to `57c125df`, changing `chat_template.jinja` among other files. The manifest records it as the latest known revision with the drift as a known issue; the pin, the C# port and the evidence stay on the pinned template.

## Upgrading from 0.5.0

- Raise every ai-dotnet `Starlings.*` reference to `0.6.0` together. `Starlings.Tokenizers` comes in at `0.3.6` (exact); an application that references the tokenizer itself raises it too, or NuGet warns with NU1608.
- Stable profiles produce the same outputs, and no default of a Stable profile changed.
- **`Qwen35ChatClient` escapes tool results by default (behaviour change).** A conversation whose tool results contain no added-token string renders as before; one that does gets a different prompt. Set `Qwen35ChatClientOptions.EscapeToolResults = false` to keep the earlier, upstream-faithful rendering. `Qwen35ChatRequestMapper.MapMessages(messages, instructions)` follows the new default.
- **`Starlings.Models.Bert` runs with ONNX Runtime's CPU arena off (behaviour change).** Scores are unchanged; peak and retained memory fall, latency rises a little. The `AppContext` switch `Starlings.Models.Bert.CpuMemoryArena` turns the arena back on for the process.
- **QwenChat caches (hosting r2).** After upgrading, a cache acquired from r1 fails `QwenChatModelAssets.OpenLocal` with `ModelAssetIntegrityException` on `model.onnx` at the default SHA-256 verification; run `DownloadAsync` once (it replaces that one 150 to 250 KB file). With size-only verification `OpenLocal` passes and `QwenChatModel.Load` refuses the bundle, so download with SHA-256 verification. r1 stays hosted for older packages, but both revisions share one cache path: give apps on different versions separate cache directories.
- **Qwen3.5 int4 cache (hosting r3).** A cache acquired from r2 fails `Qwen35ModelAssets.OpenLocal` on `ortgenai-int4/model.onnx`; one `DownloadAsync` of the int4 lock replaces it (364,388 bytes). fp32x and fp16x are unchanged at r2. The same shared-cache-path caveat applies.
- **`E5EncoderOptions.MaxSequenceLength`** outside 2 to 512 now throws `ArgumentOutOfRangeException` when the encoder, embedder, generator, ML.NET transform or browser loader is created. The default and every value in range behave as before.
- **GLiNER2 on Linux CPU** no longer needs `MALLOC_MMAP_THRESHOLD_=65536`; keeping it is harmless. GLiNER2 on CUDA, EmbeddingGemma 2 video on CUDA and Kev still need it. Hosts that cannot set the environment can use `OnnxSessionOptions.ApplyGlibcMmapThreshold` (opt-in).
- Nothing is removed. The members marked `[Obsolete]` in `0.4.0` are still present and behave as before; see [`BREAKING.md`](BREAKING.md#pending-removals).

## Hosting

The files Starling builds can be downloaded without a token. Each is pinned by commit and by SHA-256 in the packages' locks. New since 0.5.0:

| Repository | Revision | Files |
| --- | --- | --- |
| `Starlinghq/qwen3-0.6b-ortgenai` | `002782754045f2f8f15acf238d888455642181ab` (tag `r2`; r1 `c24cd59c` stays hosted) | `model.onnx` with the builder's own `producer_version`; the other bundle files unchanged |
| `Starlinghq/qwen2.5-0.5b-instruct-ortgenai` | `265459af18b9dac105d3c7a528d22cd926cfe8f7` (tag `r2`; r1 `6b6957c8` stays hosted) | as above |
| `Starlinghq/qwen3.5-0.8b-ortgenai` | `4651f7e6fd8c0421c4e248b5c82a1d7f25374177` (tag `r3`; r2 `22477cdb` stays reachable) | `ortgenai-int4/model.onnx` rebuilt at the pinned wheel stamp; every other file at its r2 bytes |
| `Starlinghq/ffmpeg-lgpl-rebuilds` (dataset) | `caf396f176296fc12620c18a344a31f628a4613a` | LGPL FFmpeg 8 and 9 shared builds with their source, for the release gates only; no package downloads them |

`Starlings.Models.PaddleOcr` downloads the official PaddlePaddle exports directly from `PaddlePaddle/PP-OCRv5_mobile_det_onnx` (`e6f4fa85`) and `PaddlePaddle/PP-OCRv5_mobile_rec_onnx` (`ed152b8b`); Starling hosts nothing for it. Unchanged since 0.5.0: the repositories in [`ai-0.5.0.md`](ai-0.5.0.md#hosting) and the records it links.

## Evidence

- Release drill: the `release` workflow's candidate job runs `tools/release-drill-ai.sh --no-model`. It packs every package of the set from a clean clone, checks each nuspec (version, license, readme, repository commit, exact dependencies, including `Starlings.Models.PaddleOcr`'s) and the per-profile maturity statements, and installs the packages into clean consumers outside the checkout. It passed on this candidate's commit.
- New evidence since 0.5.0: `docs/evidence/gte-reranker-modernbert-onboarding-2026-11-04/`, `docs/evidence/phi4-mini-onboarding-2026-11-04/`, `docs/evidence/qwen35-int4-guided-2026-11-04/`, `docs/evidence/qwenchat-hosting-r2-2026-11-04/`, `docs/evidence/kev-cuda-graph-spike-2026-11-04/`, `docs/evidence/embeddinggemma2-video-ffmpeg-rebuild-2026-11-04/`, `docs/evidence/regression-0.5.0-rtx3090-2026-11-06/`, `docs/evidence/malloc-noenv-2026-11-15/`, `docs/evidence/guardrails-2026-11-16/`, `docs/evidence/guardrails-l1b3rt4s-2026-11-16/` (unpublished packages), `docs/evidence/cuda-concurrency-2026-11-21/`. The PP-OCRv5 development run is recorded in #228.
- Gates: `docs/catalog/README.md` (generated), with links to `docs/evidence/`.
- Regression: the qualification workflow on the release tag (CPU release gates and soaks), and the maintainer's local CUDA qualification (`tools/cuda-qualification-local.sh`) on the RTX 3090 before the tag, recorded in `docs/evidence/gpu-regression-<date>/`.
