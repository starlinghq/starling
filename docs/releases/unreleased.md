# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Build-time native package checks, CUDA recipe, Native AOT smoke check (backlog E1)

**Native package checks (new, in the packages).** `Starlings.Runtime.Onnx` and `Starlings.Generation.OnnxGenAI` pack MSBuild targets in `buildTransitive/`, so every executable or test project that consumes them, directly or through a model package, checks its restored package graph:

| Code | Level | Condition |
| --- | --- | --- |
| `STARLINGS001` | warning | `Starlings.Runtime.Onnx` without an ONNX Runtime native package |
| `STARLINGS002` | error | `Microsoft.ML.OnnxRuntime` and a `Microsoft.ML.OnnxRuntime.Gpu*` package together |
| `STARLINGS003` | warning | `Starlings.Generation.OnnxGenAI` without a GenAI native package |
| `STARLINGS004` | error | `Microsoft.ML.OnnxRuntimeGenAI` and `Microsoft.ML.OnnxRuntimeGenAI.Cuda` together |

Class libraries and browser projects are not checked. Opt-out properties: `StarlingsOnnxRuntimeNativeCheck`, `StarlingsOnnxRuntimeGenAINativeCheck`. No dependency of any package changed: the libraries still reference only the managed bindings, and no native package is referenced under `ai-dotnet/src/`.

**Guide.** [docs/guides/native-runtimes-and-cuda.md](../guides/native-runtimes-and-cuda.md): the native packages per runtime, the checks, the `-p:UseCuda=true` build recipe (one project, a CUDA lock file), Linux-only GPU dependencies (the CUDA libraries are the machine's; prefer `Microsoft.ML.OnnxRuntime.Gpu.Linux`; publish with a runtime identifier when `Microsoft.ML.OnnxRuntimeGenAI.Cuda` brings the Windows GPU natives) and the Native AOT status per package.

**Native AOT smoke check (tooling, not packed).** `tools/aot-smoke-ai.sh` publishes a consumer per package (`tools/release-drill/aot-smoke`) with `-p:PublishAot=true`, the whole package assembly rooted, no model files, and records which packages are AOT-clean. The release drill runs it with `--aot` or `--aot-smoke`. Run record: [docs/evidence/aot-smoke/](../evidence/aot-smoke/README.md).

**Release drill.** Checks that both packages carry their `buildTransitive` targets, that no sample or checker triggers a native package check, and builds four throwaway consumers (`tools/release-drill/native-check`) that must trigger `STARLINGS001` to `STARLINGS004`; the two-native cases restore the GPU packages (about 420 MB) into the drill's isolated cache.

**Not provided.** Runtime convenience packages (`Starlings.Runtime.Cpu`, `.Cuda`, GenAI equivalents) and per-family umbrella packages: CI forbids native package references under `ai-dotnet/src/`, and the checks above cover the failure they would prevent (see the guide).

## Onboarding: one BERT contract table, scaffold, route suggestions (backlog E2)

**`Starlings.Models.Bert`.** The per-repository contract table is no longer a hand-written `switch`: the package embeds `model-pipeline/routes/bert-onnx-cpu-v1/contracts.json`, the file the Python allow-list (`onboarding_bert.py`) reads, and validates it on first use with the same rules. New: `BertModelProfile.SupportedRepositories`. The seven contracts are unchanged value for value, and `Parse` accepts and rejects the same documents as before.

**model-pipeline.** `model-pipeline scaffold <candidate> [--write]` generates a BERT-route candidate's recipe, contract table entry, `tools/fetch-tokenizer-assets.sh` line, package README row, .NET contract-test row and a C# runtime lock. `model-pipeline inspect` now suggests `bert-onnx-cpu-v1` (with the observed task, pooling and normalization) or `qwen-genai-cpu-v1` as well as `e5-onnx-cpu-v1`. `model-pipeline/Dockerfile` describes a contributor image with the hash-locked reference environments; no CI job builds it and it is untested.

**Capability statuses.** None changed. No profile's contract, gate, tolerance or `protected/` file changed.

## Starlings.Models.Abstractions and the encoder de-duplication (backlog A2, D3)

**New package `Starlings.Models.Abstractions`** (references only `Starlings.Acquisition`; no ONNX Runtime). `Starlings.Models.E5`, `Starlings.Models.EmbeddingGemma2`, `Starlings.Models.Gliner2` and `Starlings.Models.Bert` reference it. Its types, all additive:

- `CapabilityStatus`: one status enum for every package.
- `ModelProfileCapabilities` (with `ProfileMaturity`, `TruncationPolicy`, `ProfilePlatform`): the machine-readable capability record of a profile, generated from its registry manifest by `model-pipeline catalog` into each package's `ProfileCapabilities.g.cs`. Contract fields: profile id, maturity, max input tokens, truncation policy, dimension, qualified providers. Informational: tasks, platform rows, stop token ids, the manifest's capabilities map. `docs/catalog/catalog.json` gains the same `behavior` fields and `qualified_providers` per profile. `model-pipeline catalog --check` (CI) fails when a generated file is stale, and each package's contract tests compare every record with its manifest field for field.
- `ITextEmbedder`: implemented by `E5TextEmbedder`, `EmbeddingGemma2TextEmbedder` (new overload `EmbedBatch(texts, tokenCounts, cancellationToken)` with no prompt; new `Dimension` and `MaxSequenceLength`) and `BertTextEmbedder` (new overload with the prompt `none`; new `MaxSequenceLength`). `E5TextEmbedder` gains `Dimension` and `MaxSequenceLength`.
- `IReranker`: implemented by `BertReranker` (new `Score(query, documents, cancellationToken)`, the raw logits of `ScoreBatch`).
- `IModelAssets<TFiles>`: `E5ModelAssets.Assets`, `EmbeddingGemma2ModelAssets.Assets` and `Gliner2ModelAssets.Assets` expose the static lock and acquisition members as an instance.
- `InputTooLongException` (an `ArgumentException` with `InputIndex`, `TokenCount`, `MaxTokens`): now the base of `EmbeddingGemma2InputTooLongException` and `Gliner2InputTooLongException`. `E5Encoder` throws it (instead of a plain `ArgumentException`, same message) for an id sequence above `MaxSequenceLength`.

**New members on the capability classes.** `E5Capabilities`, `EmbeddingGemma2Capabilities` and `Gliner2Capabilities` gain `Statuses` (the package-wide table in the shared enum) and `ProfileCapabilities` (the generated record of every profile the package serves, Stable or not).

**Obsolete, removed in the next minor version.** Each still works and returns the same values as before:

| Obsolete member | Replacement |
| --- | --- |
| `Starlings.Models.E5.CapabilityStatus`, `EmbeddingGemma2CapabilityStatus`, `Gliner2CapabilityStatus` | `Starlings.Models.CapabilityStatus` (same names and order) |
| `E5Capabilities.All`, `EmbeddingGemma2Capabilities.All`, `Gliner2Capabilities.All` | `XCapabilities.Statuses` |
| `E5Capabilities.Profiles`, `EmbeddingGemma2Capabilities.Profiles`, `Gliner2Capabilities.Profiles` | `XCapabilities.ProfileCapabilities[profile].Capabilities` |
| `EmbeddingGemma2InputTooLongException.MaximumTokens` | `MaxTokens` (the shared name; GLiNER2 already used it) |

**Encoder de-duplication, outputs unchanged.** Right-padded packing, mean and CLS pooling with L2 normalization (including the Matryoshka truncation) and the chunk run moved into one internal `EncoderCore` in `Starlings.Runtime.Onnx`, used by `E5Encoder`, `EmbeddingGemma2Encoder` (text and the multimodal text path) and the BERT route. The arithmetic and its order are unchanged. On CPU, embeddings and reranker scores from the moved code are bit for bit those of the code it replaces for e5, EmbeddingGemma 2 (768 and 128 dimensions, prompts, token budget with length sorting) and four BERT-route profiles (mean pooling, CLS pooling, an unnormalized embedder and a reranker). The tolerance-based CPU parity tests of these packages, the EmbeddingGemma 2 image tests included, pass before and after. The ML.NET transform internals (data view, cursors, bindings, row mapper) and the request and result handling of the two text `IEmbeddingGenerator`s are shared source now instead of two copies; public types and behavior are unchanged.

**Capability statuses.** None changed. No Stable profile's contract, gate, tolerance or `protected/` file changed.

## Guided decoding through ONNX Runtime GenAI guidance (backlog B1, #139; code path only)

All additive, opt-in and experimental. Unguided requests are unchanged: without a guidance option the decode loop calls GenAI as before, and no Stable profile's contract, capability status, gate, tolerance or `protected/` file changed.

- **`Starlings.Generation.OnnxGenAI`.** New `GenerationGuidance` (`JsonSchema`, `Lark`, `Regex`; `Kind`, `Data`) and `GenerationGuidanceKind`. `GenerationOptions.Guidance` and `GenAISequenceOptions.Guidance` pass it to `GeneratorParams.SetGuidance`, which ONNX Runtime GenAI 0.17.1 already exposes, with llguidance compiled into the CPU and CUDA linux-x64 natives. GenAI builds the constraint's token table from `tokenizer.json` and `tokenizer_config.json` in the model directory. `GenAIModel.GuidanceFiles`, `GuidanceDirectory` (honors `model.tokenizer_dir`) and `GuidanceMissingFiles` report whether they are there. A guided request on a directory without them fails with `NotSupportedException` naming the missing files, before any native call. Guidance cannot be combined with `ReturnLogprobs`.
- **Fast-forward tokens.** `GenerationGuidance.WithFastForward(maxTokensPerStep)` turns on GenAI's forced tokens, so the grammar's fixed text is appended without a decode step per token. GenAI appends forced tokens without checking the sequence's maximum length, so the decode loop sizes the sequence for `MaxNewTokens` plus that bound. The context-length check counts the bound, and a step that appends more fails the request. Forced tokens are completion tokens: they are returned in order before the step's selected token and go through the same token limit (a completion is cut at `MaxNewTokens`), stop ids, detokenization and stop sequences. New `GenAISequence.GenerateNextTokens()` returns every token a step appended. Off by default.
- **`Starlings.Models.Qwen35`.** `Qwen35ChatOptions.Guidance` passes a constraint through. New `Qwen35ToolGrammar.Build` and `BuildLark` (with `Qwen35ToolGrammarOptions`: `Mode` `Required` or `Auto`, `AllowMultipleCalls`, `MaxValueLength` (default 1,024 characters), `FastForward`) write an llguidance Lark grammar for the pinned template's `<tool_call><function=...><parameter=...>` format: only offered tools and parameters, required parameters present, and values bounded by their `enum`, `integer`, `number` or `boolean` schema. Free-text values are bounded in length and cannot contain `<`. `Qwen35ModelInfo.GuidanceMissingFiles` reports the loaded directory's missing guidance files.
- **`Starlings.Models.Qwen35.Extensions.AI`.** When the model directory has the guidance files, `ChatToolMode.RequireAny` and `RequireSpecific` map to the tool grammar instead of throwing: all offered functions, or the named one; one call with `AllowMultipleToolCalls = false`, one or more otherwise. A JSON `ResponseFormat` maps to `json_schema`: its schema, or `{"type": "object"}` without one. Without the files, as with the published bundle, these requests still fail with `NotSupportedException`, whose message now names the missing files. Also rejected: JSON with offered tools, guidance with `enable_thinking` or with log probabilities, and a required function that is not offered. New `Qwen35ChatClientOptions.GuidanceFastForward` (default false) builds the tool grammar with fast-forward tokens.
- **Capabilities.** New `Qwen35Capabilities.All` keys `guidance.json_schema`, `guidance.lark` and `guidance.regex`: experimental. `structured_output`, `extensions_ai.tool_mode.required` and `extensions_ai.structured_output` stay unsupported, because the published bundle's variant directories have neither guidance file.
- **Evidence.** Scripted sequences in CI: the grammar builder, the grammar against calls the template renders and against calls outside the schemas, the fast-forward decode loop, and the `IChatClient` mapping. On the real fp32x model on CPU with the two files added to a temporary copy of the variant directory (local, one machine): a regex, a JSON schema and the required tool-call grammar were followed, with and without fast-forward, and unguided greedy ids still equal the reference. See `docs/evidence/qwen35-guidance-2026-10-26/`. No latency qualification.
- **Not in this change (maintainer).** The bundle change of B1 part 2: both tokenizer files in each GenAI variant directory, which means a new hosting revision and new asset locks. Until then, guided requests need a directory the user prepares.

## Upgrading

- An executable or test project that references a Starlings model package without a native package now gets warning `STARLINGS001` or `STARLINGS003` at build time (an error under `TreatWarningsAsErrors`). Add the native package, or set the opt-out property if the application deploys its own native library.
- A project with both a CPU and a GPU native package of one runtime now fails to build (`STARLINGS002`, `STARLINGS004`). Such a project shipped two copies of one native library, of which the loader picked one; keep one package.
- `Starlings.Models.Abstractions` breaks nothing: every former member is still there, marked `[Obsolete]`. Projects that treat warnings as errors see CS0618 until they move to the replacements above.
- A file that imports both `Starlings.Models` and `Starlings.Models.E5` sees two types named `CapabilityStatus` until the obsolete one is removed: qualify it or alias it (`using Status = Starlings.Models.CapabilityStatus;`).
- Code that tested for the exact type `ArgumentException` (not `is ArgumentException`) on an over-long e5 id sequence now gets `InputTooLongException`, a subclass.
- Add `Starlings.Models.Abstractions` at the same version as the other `Starlings.*` packages; it comes in transitively with e5, EmbeddingGemma 2 and GLiNER2.
- Guided decoding breaks nothing: requests that set no guidance behave as before.
- `IChatClient` callers that caught `NotSupportedException` for `RequireAny`, `RequireSpecific` or a JSON `ResponseFormat` still get it on the published bundle; the message changed.
- `AllowMultipleToolCalls = false` is now accepted with `RequireAny` and `RequireSpecific` on a directory with the guidance files; with `ChatToolMode.Auto` it is still rejected.
