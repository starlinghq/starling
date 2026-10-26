# Starlings.Models.Qwen35

The model definition of [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) at revision `2fc06364715b967f1860aea9cf38778875588b17`, without a runtime: what any backend needs to prompt the model and read its output. No ONNX Runtime dependency (only `Starlings.Generation.Abstractions`, `Starlings.Acquisition` and the owned tokenizer `Starlings.Tokenizers`). To run the model in-process on ONNX Runtime GenAI, reference [`Starlings.Models.Qwen35.OnnxGenAI`](https://github.com/starlinghq/starling/blob/main/ai-dotnet/src/Starlings.Models.Qwen35.OnnxGenAI/README.md), which brings this package and holds the Stable contract of the three registry profiles.

- `Qwen35ChatTemplate`: the pinned chat template rendered in C# for text and tool-calling conversations (`Qwen35Message`, `Qwen35Tool`, `Qwen35ToolCall`, `enable_thinking`); `Qwen35ChatTemplate.Instance` is the same renderer behind `IChatTemplate` over the backend-neutral `ChatTurn` and `ToolDefinition`.
- `Qwen35ToolCallParser`: the template's tool-call format, whole or streamed; `Qwen35ToolCallParser.Create(toolDefinitions, enableThinking)` is the same parser behind `IToolCallParser`.
- `Qwen35ToolGrammar`: the llguidance Lark grammar of the call format (experimental; see the runtime package for guided decoding).
- `Qwen35ChatOptions` and `Qwen35Prompt`: per-request settings and a rendered prompt.
- `Qwen35Checkpoint`: repository and revision, the special token ids (`<|endoftext|>` 248044, `<|im_start|>` 248045, `<|im_end|>` 248046), the owned tokenizer profile `qwen35-bpe-chat` and its digest, `Profile` (a `ModelProfile`: vocabulary 248,320, hidden size 1,024, 24 layers, head size 256, context 262,144, Apache-2.0) and `Lock`.
- `Qwen35Checkpoint.Lock`: the asset lock of the Hub checkpoint itself (`config.json`, the safetensors weights and their index, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`; about 1.76 GB), for runtimes that load the original weights. The package also carries it as `locks/qwen3.5-0.8b.checkpoint.lock.json` (JSON lock schema version 2, read by `ModelAssetLock.Load`). Nothing downloads unless you pass it to `ModelAssetStore.DownloadAsync`.

```csharp
using Starlings.Generation;
using Starlings.Models.Qwen35;
using Starlings.Tokenizers;

string text = Qwen35ChatTemplate.Render([Qwen35Message.User("What is the capital of France?")]);
var tokenizer = Tokenizer.FromFile(tokenizerPath);                  // the qwen35-bpe-chat profile asset
int[] promptIds = [.. tokenizer.Encode(text, addSpecialTokens: false).Ids];
// ... run promptIds on any backend; stop at Qwen35Checkpoint.Profile.EosTokenIds ...
Qwen35ParsedOutput parsed = Qwen35ToolCallParser.Parse(completionText, tools: [], enableThinking: false);
```

The types of the runtime package (`Qwen35ChatModel`, `Qwen35ModelAssets`, `Qwen35ModelOptions`, `Qwen35ModelInfo`, `Qwen35Capabilities`) keep the namespace `Starlings.Models.Qwen35`.

## Maturity

<!-- catalog:maturity profiles=qwen3.5-0.8b/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

The template renderer and the tool-call parser are part of every profile above; their statuses are in the runtime package's README and `Qwen35Capabilities`.

## Attribution and license

Qwen3.5-0.8B by the Qwen team, Alibaba Cloud (Copyright 2026 Alibaba Cloud), Apache-2.0 (`LICENSE` in the Hub repository at the pinned commit). `Qwen35ChatTemplate` is a C# port of the checkpoint's `chat_template.jinja` (Apache-2.0); `Qwen35ToolCallParser` ports transformers' argument casting (Apache-2.0). This package's code is MIT OR Apache-2.0.
