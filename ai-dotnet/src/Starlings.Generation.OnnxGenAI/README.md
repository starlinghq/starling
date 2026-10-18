# Starlings.Generation.OnnxGenAI

Text generation on [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) 0.17.1 behind a Starling adapter, with an owned decode loop. Prompt ids come from the owned tokenizer (`Starlings.Tokenizers`); GenAI's tokenizer and template engine are not used. The API may still change.

- `GenAIModel.Load(directory, options)`: explicit provider (CPU, or CUDA with TF32 off by default), no silent fallback, config overlays (threads, pad id) recorded in `AppliedOverlays`, reference-counted disposal, package version check. On CUDA the native calls of concurrent sequences are serialized by default (`SerializeNativeCalls`): concurrent GenAI 0.17.1 generators on the CUDA provider produced wrong tokens.
- `TextGenerator`: greedy or seeded sampling, `MaxNewTokens`, stop token ids (never part of the completion) and stop strings (text cut before the match), streaming with UTF-8-safe incremental detokenization (`IncrementalDetokenizer`), optional chunked prefill, cancellation checked before the prefill, between prefill chunks and between decode steps (not inside a step), usage (`PromptTokens`, `CompletionTokens` excluding the stop token) and stage timings. Synchronous by design: each step is blocking native work.
- `StopSequenceMatcher`: the stop-string matcher `TextGenerator` uses, for applications with their own decode loop. Push each detokenized piece and emit what `Push` returns; text that could still start a stop string is held back, so no emitted piece contains part of one. When `Matched` is set, the text is cut before the match and the request ends; otherwise emit `Flush()` at the end. Ordinal comparison; the earliest match wins, the longest at the same start. One instance per request.
- Advanced: `GenAISequence` (append ids, step, copy the last logits).

Most applications use it through a model package such as `Starlings.Models.Qwen35`.

## Use

```csharp
using Starlings.Generation.OnnxGenAI;
using Starlings.Tokenizers;

using var model = GenAIModel.Load(modelDirectory, new GenAIModelOptions { Provider = GenAIExecutionProvider.Cpu });
var tokenizer = Tokenizer.FromFile(tokenizerPath);
var generator = new TextGenerator(model, tokenizer);
var promptIds = tokenizer.Encode(promptText, addSpecialTokens: false).Ids;   // the model package renders the template
GenerationResult result = generator.Generate(promptIds, new GenerationOptions { MaxNewTokens = 256 });
Console.WriteLine($"{result.Text} ({result.FinishReason}, {result.Usage.CompletionTokens} tokens)");
```

- `GenAIModelOptions`: `Provider` (default `Cpu`), `CudaDeviceId` (default 0), `UseTf32` (default false), `IntraOpThreads` (null: the CPU quota or affinity count in a constrained container, else the GenAI default), `PadTokenId` (null keeps the model's), `ConfigOverlays` (advanced JSON overlays on `genai_config.json`), `SerializeNativeCalls` (null: serialized on CUDA, concurrent on CPU). The loaded `GenAIModel` reports `ModelType`, `ContextLength`, `VocabularySize`, `EosTokenIds`, `PadTokenId`, `AppliedOverlays`, `ProviderOptions` and `SerializesNativeCalls`.
- `GenerationOptions`: `MaxNewTokens` (default 512, at least 1), `StopSequences` (non-empty strings), `AdditionalStopTokenIds`, `Sampling` (null: greedy), `PrefillChunkTokens` (null: one prefill).
- `SamplingOptions`: `Temperature` (default 1.0, greater than 0), `TopK` (default 50), `TopP` (default 1.0, in (0, 1]), `Seed` (required, non-negative). The same seed, prompt, settings, model and device reproduce the same tokens.
- `GenerationResult`: `Text`, `TokenIds`, `FinishReason` (`Stop`, `StopSequence` or `Length`; cancellation throws instead), `StopTokenId`, `StopSequence`, `Usage` (`PromptTokens`, `CompletionTokens`, `TotalTokens`), `Timings` (`Prefill`, `Decode`). `GenerateStreaming` yields `GenerationUpdate`s (`Text`, `TokenIds`); the last has `IsFinal` and `Result`.
- `GenAIRuntimeInfo`: `PinnedVersion` (`0.17.1`), and `ManagedVersion` and `NativePackage` read from the application's `.deps.json` (null in single-file or AOT hosts).
- Exceptions: `GenAIRuntimeUnavailableException` (no native library, or packages other than 0.17.1), `GenAIProviderUnavailableException` (the provider cannot be used; no fallback), `GenAIModelException` (GenAI rejected the directory, its configuration or a step), `GenerationContextLengthException` (prompt plus `MaxNewTokens` exceed the context length; `PromptTokens`, `MaxNewTokens`, `ContextLength`).

## Maturity

<!-- catalog:maturity profiles=qwen3.5-0.8b/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Native runtime

The package references only `Microsoft.ML.OnnxRuntimeGenAI.Managed` (exactly 0.17.1; it brings `Microsoft.Extensions.AI.Abstractions`). The application references exactly one native package of the same version:

| Deployment | Package | Notes |
| --- | --- | --- |
| CPU | `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 | Brings `Microsoft.ML.OnnxRuntime` 1.30.0. Natives for linux-x64, linux-arm64, win-x64, win-arm64, osx-arm64; exercised on linux-x64 only. |
| CUDA | `Microsoft.ML.OnnxRuntimeGenAI.Cuda` 0.17.1 | Brings `Microsoft.ML.OnnxRuntime.Gpu` 1.30.0. linux-x64 and win-x64 natives (CUDA 13 build); the CUDA runtime and cuBLAS are not shipped. Exercised on linux-x64 only. |

The model directory must come from the onnxruntime-genai builder of the same version. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#generation-backend-and-pilot-llm-p-016).
