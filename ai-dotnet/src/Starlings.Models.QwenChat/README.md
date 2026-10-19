# Experimental Qwen text chat

Experimental. The package is on nuget.org from release 0.3.0 (`docs/releases/ai-0.3.0.md`); keep every `Starlings.*` package at the same version. The API may still change. For the Stable Qwen3.5-0.8B chat model, use `Starlings.Models.Qwen35`.

## Maturity

<!-- catalog:maturity profiles=qwen3-0.6b/*,qwen2.5-0.5b-instruct/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `qwen3-0.6b/ortgenai-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
<!-- /catalog:maturity -->

The admitted profiles are `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` and `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`. Both run CPU fp32 with ONNX Runtime GenAI 0.17.1 and the owned tokenizer. Add `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 to the application for the native runtime.

```csharp
using Starlings.Models.QwenChat;
// Setup step, once: downloads (or verifies the cached) bundle, 2.4 GB, anonymously.
string bundleDirectory = await QwenChatModelAssets.DownloadAsync(QwenChatProfile.Qwen3);
// At startup, offline: QwenChatModelAssets.OpenLocal(QwenChatProfile.Qwen3).
using var model = QwenChatModel.Load(QwenChatProfile.Qwen3, bundleDirectory);
var result = model.Generate([new(QwenChatRole.User, "Explain gravity briefly.")],
    new QwenChatOptions { MaxNewTokens = 128 });
Console.WriteLine(result.Text);
```

`QwenChatModelAssets.DownloadAsync` fetches the profile's bundle from Starling's public hosting (see "Bundle"); no token is needed. The loader checks every bundle file against the profile's pinned digests (`QwenChatProfile.BundleFiles`), as well as the tokenizer and template identities, the architecture, the vocabulary and the EOS ids. `Load` itself never downloads anything.

These are experimental text-only profiles. Qwen3 explicitly uses `enable_thinking=false`, which differs from its upstream default. Qwen2.5 inserts the upstream default system instruction when the first message is not a system message. Tools, tool messages, multimodal input, sampling and Qwen3 thinking mode are unsupported. Each request has its own disposable generator/cache; continuing chat requires resending the whole conversation. CPU native threads are bounded to two. Cancellation is checked before prefill and between decode steps; a running native operation is not interruptible.

Both native paths overlay `model.pad_token_id=151935`, the last embedding row, after verifying neither original vocabulary nor added tokens maps to it (maximum tokenizer ids: Qwen3 151668; Qwen2.5 151664). GenAI otherwise treats literal `<|endoftext|>` id 151643 in prompt text as padding. Original configuration and graph files are never modified by this runtime overlay.

Fresh reference and parity reports remain proposals. They do not award protected G1/G2 qualification, approve releases, or change support status.

## Bundle

The bundles are hosted publicly, one Hugging Face repository per profile, at a pinned commit (tag `r1`): `Starlinghq/qwen3-0.6b-ortgenai` (`c24cd59c`) for Qwen3 (2.4 GB) and `Starlinghq/qwen2.5-0.5b-instruct-ortgenai` (`6b6957c8`) for Qwen2.5 (2.0 GB), with the original `LICENSE`, a `NOTICE` and `SHA256SUMS`. `QwenChatModelAssets.Lock(profile)` is the acquisition lock (`Starlings.Acquisition`): the six pinned files, each sourced from `HostingBaseUrl(profile)`. The digests, not the address, pin the bytes.

To build a bundle instead, `build_bundle.py --candidate <profile.json> --snapshot <original snapshot> --out <bundle>` needs the `onnxruntime-genai` 0.17.1 builder; the candidate contracts are `model-pipeline/candidates/qwen3/profile.json` and `model-pipeline/candidates/qwen25/profile.json`. `Load` requires exactly these files in the bundle directory: `genai_config.json`, `model.onnx`, `model.onnx.data`, `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`. Each file's size and SHA-256 must equal the profile's pin (`QwenChatProfile.BundleFiles`, the same digests as the registry manifest's `execution.graph.assets`) as hashed on disk. A downloaded bundle has no `bundle-lock.json`; when one is present (a local build), its entries must equal the pin too, and it must also name the profile, the original repository and revision, builder 0.17.1 at source commit `83de55a1`, fp32 and `CPUExecutionProvider`. A mismatch throws `InvalidDataException`.

The build is reproducible: on 2026-10-18 two builds from the pinned originals with the hash-locked builder (`model-pipeline/reference/generation/requirements.txt`) gave the pinned digests byte for byte, for both profiles. A builder wheel or platform outside that lock was not tested, and a bundle with other digests is refused.

## API

- `QwenChatProfile.Qwen3` (`qwen3-0.6b/ortgenai-fp32-cpu`, context 40,960 tokens) and `QwenChatProfile.Qwen25` (`qwen2.5-0.5b-instruct/ortgenai-fp32-cpu`, context 32,768 tokens); `QwenChatProfile.FromId(id)` resolves either id and throws `NotSupportedException` otherwise.
- `QwenChatModelAssets.DownloadAsync(profile, options, progress, cancellationToken)` and `OpenLocal(profile, options)` return the verified bundle directory (`ModelAcquisitionOptions`: cache directory, verification); `HostingRepository`, `HostingRevision`, `HostingBaseUrl` and `Lock` per profile.
- `QwenChatModel.Load(profile, bundleDirectory, options)`: `QwenChatModelOptions.IntraOpThreads` is 1 or 2 (default 2).
- `CreatePrompt(messages)` renders the owned template and returns `QwenChatPrompt(Text, TokenIds)`. `Generate(messages | prompt, options, cancellationToken)` returns a `GenerationResult` (`Starlings.Generation.OnnxGenAI`); `GenerateStreaming(prompt, options, cancellationToken)` yields `GenerationUpdate`s with the same stop and usage contract. Generation is greedy.
- `QwenChatOptions`: `MaxNewTokens` (default 16), `AdditionalStopTokenIds`, `StopSequences`, `PrefillChunkTokens` (default null: one prefill).
- `QwenChatMessage(Role, Content, ReasoningContent)`: roles `System`, `User`, `Assistant`. `ReasoningContent` is accepted only on assistant history; elsewhere it throws `NotSupportedException`.
- `Info` (`QwenChatModelInfo`): profile, repository, revision, tokenizer and template SHA-256, GenAI version, native package, context length, and the capability values (`TextGeneration` experimental, `Tools` and `Multimodal` unsupported, `Cache` fresh-request).
- `Tokenizer`, `Profile`, and `CreateSequence(GenAISequenceOptions)` (advanced, an independent disposable sequence).
