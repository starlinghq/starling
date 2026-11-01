# Breaking changes and deprecations

Every change to the public C# API of the `Starlings.*` packages that can break a consumer's build or behavior, per ai-dotnet version, newest first: members marked `[Obsolete]` (with the replacement and the planned removal), members removed (with the release that announced them), and other incompatible changes. It collects, in one place, what each release record (`ai-<version>.md`) lists under its "Upgrading" section and its release notes; the release record stays the authoritative statement for its version, and this log links to it. The rules are in the [support policy](../support-policy.md#public-api-deprecation): one minor version of `[Obsolete]` notice before a removal at `0.x`, and a removal that no earlier release record announced is a release blocker.

A model-output change is not an API change: it is a new qualified profile and a release note (support policy, profile rules). Behavior changes are listed here only where code written against the earlier version can observe them (an exception type, a message format, a default, a file a lock requires).

The release-candidate change adds a section here for its version, from `unreleased.md` and its own record, in the same change that writes `ai-<version>.md`.

## Pending removals

Marked `[Obsolete]` (a warning, `error: false`) and removed in the next minor version after the one that marked them. Each still works and returns the same values until then.

| Obsolete member | Package | Replacement | Marked in |
| --- | --- | --- | --- |
| `Starlings.Models.E5.CapabilityStatus` | `Starlings.Models.E5` | `Starlings.Models.CapabilityStatus` (same names and order) | 0.4.0 |
| `Starlings.Models.EmbeddingGemma2.EmbeddingGemma2CapabilityStatus` | `Starlings.Models.EmbeddingGemma2` | `Starlings.Models.CapabilityStatus` | 0.4.0 |
| `Starlings.Models.Gliner2.Gliner2CapabilityStatus` | `Starlings.Models.Gliner2` | `Starlings.Models.CapabilityStatus` | 0.4.0 |
| `E5Capabilities.All`, `EmbeddingGemma2Capabilities.All`, `Gliner2Capabilities.All` | e5, EmbeddingGemma 2, GLiNER2 | `XCapabilities.Statuses` | 0.4.0 |
| `E5Capabilities.Profiles`, `EmbeddingGemma2Capabilities.Profiles`, `Gliner2Capabilities.Profiles` | e5, EmbeddingGemma 2, GLiNER2 | `XCapabilities.ProfileCapabilities[profile].Capabilities` | 0.4.0 |
| `EmbeddingGemma2InputTooLongException.MaximumTokens` | `Starlings.Models.EmbeddingGemma2` | `MaxTokens` (on `Starlings.Models.InputTooLongException`) | 0.4.0 |
| `WhisperCapabilities.StableProfile` | `Starlings.Models.Whisper` (not yet published) | `WhisperCapabilities.CpuProfile` (same value) | unreleased (source only) |

The [`Starlings.Models.Abstractions` README](../../ai-dotnet/src/Starlings.Models.Abstractions/README.md#migrating-from-the-per-package-types) has the migration for the first six rows.

## Unreleased

Merged to `main` after 0.4.0; collected in [`unreleased.md`](unreleased.md). Nothing here is published yet.

- **Obsolete:** `WhisperCapabilities.StableProfile`, replaced by `WhisperCapabilities.CpuProfile` (the profile is Experimental, not Stable). `Starlings.Models.Whisper` has not been published to nuget.org, so this affects only source consumers.
- **`Starlings.Models.Whisper` (source only).** `WhisperModelAssets.Lock` adds `onnx/decoder_with_past_model_fp16.onnx`: a cache acquired earlier makes `WhisperModelAssets.OpenLocal` and `OpenDirectory` throw `ModelAssetMissingException` until `DownloadAsync` runs again, with either `DecoderMode`. KV-cache decoding is the default (`WhisperOptions.DecoderMode = WhisperDecoderMode.FullContext` keeps the previous loop). `WhisperOptions.NoTimestamps = false` makes `Transcribe` return the timestamped decode's text. `WhisperTokenizer.Decode` drops timestamp tokens.
- **Tokenizer pin** `Starlings.Tokenizers [0.3.4]` (0.3.2 in 0.4.0): additive; outputs of existing profiles are unchanged.
- No member was removed.

## 0.4.0

Record: [`ai-0.4.0.md`](ai-0.4.0.md), "Upgrading from 0.3.0".

**Breaking: the generation package split.** Generated tokens, texts and ids do not change; package references and namespaces do. Assemblies built against 0.3.0 must be recompiled: the types moved to another assembly and namespace, with no type forwarders.

- `Starlings.Models.Qwen35` is now the model definition without ONNX Runtime. In-process users reference `Starlings.Models.Qwen35.OnnxGenAI` instead (it brings `Starlings.Models.Qwen35`); `Starlings.Models.Qwen35.Extensions.AI` users change nothing.
- `Starlings.Models.QwenChat` is now the model definition. In-process users reference `Starlings.Models.QwenChat.OnnxGenAI`. `QwenChatTemplate.Render` takes a `QwenChatCheckpoint`; `Render(profile, ...)` still compiles because `QwenChatProfile` converts to its checkpoint.
- `Starlings.Models.Kev` is now the model definition (depends on `Starlings.Tokenizers` only). Users of `KevDecisionModel`, `KevOptions`, `KevState`, `KevPlacement`, `KevModelAssets`, `KevModelFiles` and `KevCapabilities` reference `Starlings.Models.Kev.Onnx`.
- `GenerationOptions`, `SamplingOptions`, `GenerationGuidance`, `GenerationResult`, `GenerationUpdate`, `GenerationFinishReason`, `GenerationUsage`, `GenerationTimings`, `TokenLogprob`, `TopTokenLogprob`, `GenerationContextLengthException`, `StopSequenceMatcher` and `IncrementalDetokenizer` moved from `Starlings.Generation.OnnxGenAI` to the namespace `Starlings.Generation` in the new `Starlings.Generation.Abstractions`: add `using Starlings.Generation;`.

**Obsolete** (removed in 0.5.0; see [Pending removals](#pending-removals)): the per-package capability status enums, `XCapabilities.All` and `XCapabilities.Profiles` of e5, EmbeddingGemma 2 and GLiNER2, and `EmbeddingGemma2InputTooLongException.MaximumTokens`. A file that imports both `Starlings.Models` and `Starlings.Models.E5` sees two types named `CapabilityStatus` until the obsolete one is removed: qualify it or alias it (`using Status = Starlings.Models.CapabilityStatus;`). Projects that treat warnings as errors see CS0618.

**Other incompatible changes:**

- `E5Encoder` throws `InputTooLongException` (an `ArgumentException` subclass, same message) for an over-long id sequence; code that tested for the exact type `ArgumentException` no longer matches.
- `ITextEmbedder` is new; a class outside these packages that implements it must implement `CountTokens(text)`, `CountTokens(texts)` and `Validate(text)`.
- `KevInputTooLongException.Message` for an over-long question starts with `question <index>:` and names every over-long question; the properties are unchanged.
- `Qwen35ChatTemplate.Render` throws `Qwen35TemplateException` (was `InvalidOperationException`) for a key holding a lone UTF-16 surrogate. `Qwen35ChatTemplate.Instance.Render` throws `ArgumentException` for an empty conversation.
- `ModelAssetLock.SchemaVersion` is 3 (a constant: rebuild to see it); compare version-2 locks with `SourcesSchemaVersion`.
- `ModelAssetStage` gains `Waiting`; a `switch` over it sees the new value.
- `ModelAssetStore.DownloadAsync` and `OpenLocal` each gain a second overload: a cref or method group without parameters is ambiguous.
- `EmbeddingGemma2MultimodalOptions.Session.KnownModelSha256` is ignored; each graph takes its own digest.
- `IChatClient` (`Starlings.Models.Qwen35.Extensions.AI`): `AllowMultipleToolCalls = false` is accepted with `RequireAny` and `RequireSpecific` when the guidance files are present; the `NotSupportedException` message for the published bundle changed.
- `Qwen35ChatModel` on CPU prefills prompts above 1,024 tokens in 1,024-token chunks unless `PrefillChunkTokens` is set (`int.MaxValue` keeps the one-call prefill).
- Build diagnostics: an executable or test project that references a model runtime package without a native package gets `STARLINGS001` or `STARLINGS003`; one with both a CPU and a GPU native package of one runtime fails with `STARLINGS002` or `STARLINGS004`.
- `Starlings.Models.Bert` (first publish; for users of the library from source): `bge-m3` and `bge-reranker-v2-m3` `RuntimeLock`s have 3 files (`onnx/model.onnx_data` added); download again.

**Removed:** nothing.

## 0.3.0

Record: [`ai-0.3.0.md`](ai-0.3.0.md), "Upgrading from 0.2.0".

No breaking change, no member marked `[Obsolete]`, nothing removed. Additions only: opt-in batching on the embedding and extraction packages (off unless set), the public stop-string matcher (#124), and the new packages `Starlings.Models.Kev` and `Starlings.Models.QwenChat`. `QwenChatOptions.MaxNewTokens` defaults to 16 (the corpus setting).

## 0.2.0

Record: [`ai-0.2.0.md`](ai-0.2.0.md), "Upgrading from 0.1.0-preview.6".

No breaking API change, no member marked `[Obsolete]`, nothing removed. The first version without a preview suffix (`--prerelease` is no longer needed). `Starlings.Tokenizers` moves from `0.3.0-preview.1` to `0.3.0` with unchanged outputs.

## Earlier previews

`ai-0.1.0-preview.5`, `ai-0.1.0-preview.6` and the `e5-0.1.0-preview.*` releases predate the deprecation policy; their records in this directory list their changes.
