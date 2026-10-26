# Starlings.Generation.Abstractions

Backend-neutral types for Starling's chat models: everything a generation runtime and a model definition share, without ONNX Runtime. Dependencies: the owned tokenizer `Starlings.Tokenizers` and `Microsoft.Extensions.AI.Abstractions` 10.10.1 only. The ONNX Runtime GenAI backend is `Starlings.Generation.OnnxGenAI`; the model definitions are `Starlings.Models.Qwen35` and `Starlings.Models.QwenChat`. The API may still change.

Namespace `Starlings.Generation`:

- Request and result: `GenerationOptions` (`MaxNewTokens`, `StopSequences`, `AdditionalStopTokenIds`, `Sampling`, `PrefillChunkTokens`, `ReturnLogprobs`, `TopLogprobs`, `Guidance`; `Validate()` applies the decode loop's checks), `SamplingOptions`, `GenerationGuidance` (experimental: JSON Schema, Lark grammar or regex, optional fast-forward), `GenerationResult`, `GenerationUpdate`, `GenerationFinishReason`, `GenerationUsage`, `GenerationTimings`, `TokenLogprob`, `TopTokenLogprob` and `GenerationContextLengthException`.
- Text: `StopSequenceMatcher` (stop strings over streamed text, never emitting part of one) and `IncrementalDetokenizer` (UTF-8-safe streaming detokenization with the owned tokenizer).
- Chat: `ChatTurn` (`ChatTurnRole` system, user, assistant or tool; content, reasoning, `ToolCalls`), `ToolCall` (name and JSON arguments), `ToolDefinition` (the JSON a template renders), `IChatTemplate` (render turns, tools and the thinking switch to prompt text), `IToolCallParser` (split a streamed completion into `CompletionPart`s), `ParsedCompletion`.
- `ModelProfile`: a pinned checkpoint's repository and revision, owned tokenizer profile and digest, template digest, end-of-sequence ids, vocabulary size, hidden size, layers, head size, context length and licence.
- `PythonJson`: Python `json.dumps(ensure_ascii=False)`, `str()` and `repr(float)` of JSON values, as chat templates write tool data; data Python could not hold the same way throws `PythonJsonException`.
- `IGenerationSequence`: the asynchronous seam of one request's decoding state (`AppendTokensAsync`, `StepAsync`, `CopyLastLogitsAsync`, `Dispose`). `GenAISequence` in `Starlings.Generation.OnnxGenAI` implements it; other backends can.
- `GenerationChatMapping`: `ToChatFinishReason`, `ToUsageDetails` and `ToChatRole` for Microsoft.Extensions.AI adapters.

```csharp
using Starlings.Generation;
using Starlings.Models.Qwen35;

IChatTemplate template = Qwen35ChatTemplate.Instance;       // the same text as Qwen35ChatTemplate.Render
string prompt = template.Render([ChatTurn.User("What is the capital of France?")]);
IToolCallParser parser = Qwen35ToolCallParser.Create(tools: [], enableThinking: false);
ParsedCompletion parsed = ParsedCompletion.Parse(parser, completionText);
```

## Maturity

<!-- catalog:maturity profiles=qwen3.5-0.8b/*,qwen3-0.6b/*,qwen2.5-0.5b-instruct/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `qwen3-0.6b/ortgenai-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

The types moved here from `Starlings.Generation.OnnxGenAI` unchanged in behaviour; the migration from 0.3.0 is in the release notes. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#generation-backend-and-pilot-llm-p-016).
