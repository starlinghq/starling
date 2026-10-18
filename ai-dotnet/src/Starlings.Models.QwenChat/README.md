# Experimental Qwen text chat

Experimental. This package is not published to nuget.org: it is not part of the 0.2.0 release (`docs/releases/ai-0.2.0.md`, "Not published"). Build it from this repository, for example with a `ProjectReference` to `ai-dotnet/src/Starlings.Models.QwenChat/Starlings.Models.QwenChat.csproj`. The API may still change. For the published, Stable Qwen3.5-0.8B chat model, use `Starlings.Models.Qwen35`.

The admitted profiles are `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` and `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`. Both run CPU fp32 with ONNX Runtime GenAI 0.17.1 and the owned tokenizer. Add `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 to the application for the native runtime.

```csharp
using Starlings.Models.QwenChat;
using var model = QwenChatModel.Load(QwenChatProfile.Qwen3, bundleDirectory);
var result = model.Generate([new(QwenChatRole.User, "Explain gravity briefly.")],
    new QwenChatOptions { MaxNewTokens = 128 });
Console.WriteLine(result.Text);
```

Build the hash-verified original snapshot with `model-pipeline/reference/qwen-chat/build_bundle.py`. The loader validates the admitted original tokenizer and template, architecture, vocabulary, EOS ids and proposed graph lock. It never downloads weights. Graph hashes are observed candidate evidence, not a published independently adopted asset lock.

These are experimental text-only profiles. Qwen3 explicitly uses `enable_thinking=false`, which differs from its upstream default. Qwen2.5 inserts the upstream default system instruction when the first message is not a system message. Tools, tool messages, multimodal input, sampling and Qwen3 thinking mode are unsupported. Each request has its own disposable generator/cache; continuing chat requires resending the whole conversation. CPU native threads are bounded to two. Cancellation is checked before prefill and between decode steps; a running native operation is not interruptible.

Both native paths overlay `model.pad_token_id=151935`, the last embedding row, after verifying neither original vocabulary nor added tokens maps to it (maximum tokenizer ids: Qwen3 151668; Qwen2.5 151664). GenAI otherwise treats literal `<|endoftext|>` id 151643 in prompt text as padding. Original configuration and graph files are never modified by this runtime overlay.

Fresh reference and parity reports remain proposals. They do not award protected G1/G2 qualification, approve releases, or change support status.

## Bundle

`build_bundle.py --candidate <profile.json> --snapshot <original snapshot> --out <bundle>` needs the `onnxruntime-genai` 0.17.1 builder; the candidate contracts are `model-pipeline/candidates/qwen3/profile.json` and `model-pipeline/candidates/qwen25/profile.json`. `Load` requires exactly these files in the bundle directory, each matching the size and SHA-256 in `bundle-lock.json`: `genai_config.json`, `model.onnx`, `model.onnx.data`, `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`. The lock must name the profile, the original repository and revision, builder 0.17.1, fp32 and `CPUExecutionProvider`. A mismatch throws `InvalidDataException`.

## API

- `QwenChatProfile.Qwen3` (`qwen3-0.6b/ortgenai-fp32-cpu`, context 40,960 tokens) and `QwenChatProfile.Qwen25` (`qwen2.5-0.5b-instruct/ortgenai-fp32-cpu`, context 32,768 tokens); `QwenChatProfile.FromId(id)` resolves either id and throws `NotSupportedException` otherwise.
- `QwenChatModel.Load(profile, bundleDirectory, options)`: `QwenChatModelOptions.IntraOpThreads` is 1 or 2 (default 2).
- `CreatePrompt(messages)` renders the owned template and returns `QwenChatPrompt(Text, TokenIds)`. `Generate(messages | prompt, options, cancellationToken)` returns a `GenerationResult` (`Starlings.Generation.OnnxGenAI`); `GenerateStreaming(prompt, options, cancellationToken)` yields `GenerationUpdate`s with the same stop and usage contract. Generation is greedy.
- `QwenChatOptions`: `MaxNewTokens` (default 16), `AdditionalStopTokenIds`, `StopSequences`, `PrefillChunkTokens` (default null: one prefill).
- `QwenChatMessage(Role, Content, ReasoningContent)`: roles `System`, `User`, `Assistant`. `ReasoningContent` is accepted only on assistant history; elsewhere it throws `NotSupportedException`.
- `Info` (`QwenChatModelInfo`): profile, repository, revision, tokenizer and template SHA-256, GenAI version, native package, context length, and the capability values (`TextGeneration` experimental, `Tools` and `Multimodal` unsupported, `Cache` fresh-request).
- `Tokenizer`, `Profile`, and `CreateSequence(GenAISequenceOptions)` (advanced, an independent disposable sequence).
