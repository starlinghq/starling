# Starlings.Models.QwenChat

The model definitions of the experimental Qwen text chat profiles, without a runtime: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` (rendered with `enable_thinking=false`) and `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`. No ONNX Runtime dependency (only `Starlings.Generation.Abstractions`, `Starlings.Acquisition` for the checkpoint locks, and the owned tokenizer `Starlings.Tokenizers`). To run the models on ONNX Runtime GenAI, reference [`Starlings.Models.QwenChat.OnnxGenAI`](https://github.com/starlinghq/starling/blob/main/ai-dotnet/src/Starlings.Models.QwenChat.OnnxGenAI/README.md), which brings this package. Experimental; the API may still change.

- `QwenChatTemplate.Render(checkpoint, messages, tools)`: the pinned templates rendered in C# for system, user, assistant and tool messages, tool definitions and assistant tool calls (tool calling experimental; multimodal input rejected). A `QwenChatProfile` of the runtime package converts implicitly to its checkpoint, so `Render(profile, ...)` works as before. `QwenChatTemplate.For(checkpoint)` (or `checkpoint.ChatTemplate`) is the same renderer behind `IChatTemplate`.
- `QwenChatToolCallParser`: the templates' JSON tool-call form, whole or streamed; it also implements `IToolCallParser`.
- `QwenChatMessage`, `QwenChatTool`, `QwenChatToolCall`, `QwenChatOptions` (with `ToGenerationOptions()`), `QwenChatPrompt`; conversions to and from `ChatTurn`, `ToolDefinition` and `ToolCall`.
- `QwenChatCheckpoint.Qwen3` and `QwenChatCheckpoint.Qwen25`: repository, revision, model type, tokenizer and template SHA-256, end-of-sequence ids, decoder shape from the pinned `config.json`, `Profile` (a `ModelProfile`), and `OpenTokenizer(path)`, which verifies the original `tokenizer.json` before loading it.
- `QwenChatCheckpoint.Lock`: the asset lock of each Hub checkpoint itself (`config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`; about 1.5 GB for Qwen3, 1.0 GB for Qwen2.5), for runtimes that load the original weights. The package also carries them as `locks/qwen3-0.6b.checkpoint.lock.json` and `locks/qwen2.5-0.5b-instruct.checkpoint.lock.json` (`LockFileName`; JSON lock schema version 2, read by `ModelAssetLock.Load`). Nothing downloads unless you pass a lock to `ModelAssetStore.DownloadAsync`. The locks leave out the repository's `README.md` and `LICENSE` (both checkpoints are Apache-2.0): a directory filled from a lock holds the weights without the license text, so fetch `LICENSE` from the Hub repository at the same revision if you redistribute them.

```csharp
using Starlings.Models.QwenChat;

QwenChatCheckpoint checkpoint = QwenChatCheckpoint.Qwen3;
var tokenizer = checkpoint.OpenTokenizer(tokenizerPath);          // digest and special ids checked
string text = QwenChatTemplate.Render(checkpoint, [new(QwenChatRole.User, "Explain gravity briefly.")]);
int[] promptIds = [.. tokenizer.Encode(text, addSpecialTokens: false).Ids];
```

## Maturity

<!-- catalog:maturity profiles=qwen3-0.6b/*,qwen2.5-0.5b-instruct/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `qwen3-0.6b/ortgenai-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
<!-- /catalog:maturity -->

## Attribution and license

The C# renderers adapt the `chat_template` strings of the pinned originals, Copyright 2024 Alibaba Cloud, Apache-2.0: see `THIRD-PARTY-NOTICES.md` and `LICENSE.Qwen.txt` in this package. This package's code is MIT OR Apache-2.0.
