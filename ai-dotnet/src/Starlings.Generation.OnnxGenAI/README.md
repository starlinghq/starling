# Starlings.Generation.OnnxGenAI

Text generation on [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) 0.17.1 behind a Starling adapter, with an owned decode loop. Prompt ids come from the owned tokenizer (`Starlings.Tokenizers`); GenAI's tokenizer and template engine are not used. The API may still change.

- `GenAIModel.Load(directory, options)`: explicit provider (CPU, or CUDA with TF32 off by default), no silent fallback, config overlays (threads, pad id) recorded in `AppliedOverlays`, reference-counted disposal, package version check. On CUDA the native calls of concurrent sequences are serialized by default (`SerializeNativeCalls`): concurrent GenAI 0.17.1 generators on the CUDA provider produced wrong tokens.
- `TextGenerator`: greedy or seeded sampling, `MaxNewTokens`, stop token ids (never part of the completion) and stop strings (text cut before the match), streaming with UTF-8-safe incremental detokenization (`IncrementalDetokenizer`), optional chunked prefill, cancellation checked before the prefill, between prefill chunks and between decode steps (not inside a step), usage (`PromptTokens`, `CompletionTokens` excluding the stop token) and stage timings. Synchronous by design: each step is blocking native work.
- `StopSequenceMatcher`: the stop-string matcher `TextGenerator` uses, for applications with their own decode loop. Push each detokenized piece and emit what `Push` returns; text that could still start a stop string is held back, so no emitted piece contains part of one. When `Matched` is set, the text is cut before the match and the request ends; otherwise emit `Flush()` at the end. Ordinal comparison; the earliest match wins, the longest at the same start. One instance per request.
- Experimental, opt-in: guided decoding (`GenerationOptions.Guidance`): a JSON Schema, Lark grammar or regex constraint passed to GenAI's llguidance integration (`GeneratorParams.SetGuidance`), optionally with fast-forward tokens.
- Advanced: `GenAISequence` (append ids, step, step with forced tokens, copy the last logits); it also implements the asynchronous `IGenerationSequence` seam of `Starlings.Generation.Abstractions` (each call completes synchronously).

The request and result types (`GenerationOptions`, `SamplingOptions`, `GenerationGuidance`, `GenerationResult`, `GenerationUpdate`, `GenerationUsage`, `TokenLogprob`, `GenerationContextLengthException`), `StopSequenceMatcher` and `IncrementalDetokenizer` are in `Starlings.Generation.Abstractions` (namespace `Starlings.Generation`), which this package references; they moved there unchanged in behaviour. Code in the namespace `Starlings.Generation.OnnxGenAI` sees them without a `using`; other code adds `using Starlings.Generation;`.

Most applications use it through a model package such as `Starlings.Models.Qwen35.OnnxGenAI`.

## Use

```csharp
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Tokenizers;

using var model = GenAIModel.Load(modelDirectory, new GenAIModelOptions { Provider = GenAIExecutionProvider.Cpu });
var tokenizer = Tokenizer.FromFile(tokenizerPath);
var generator = new TextGenerator(model, tokenizer);
var promptIds = tokenizer.Encode(promptText, addSpecialTokens: false).Ids;   // the model package renders the template
GenerationResult result = generator.Generate(promptIds, new GenerationOptions { MaxNewTokens = 256 });
Console.WriteLine($"{result.Text} ({result.FinishReason}, {result.Usage.CompletionTokens} tokens)");
```

- `GenAIModelOptions`: `Provider` (default `Cpu`), `CudaDeviceId` (default 0), `UseTf32` (default false), `IntraOpThreads` (null: the CPU quota or affinity count in a constrained container, else the GenAI default), `PadTokenId` (null keeps the model's), `ConfigOverlays` (advanced JSON overlays on `genai_config.json`), `SerializeNativeCalls` (null: serialized on CUDA, concurrent on CPU). The loaded `GenAIModel` reports `ModelType`, `ContextLength`, `VocabularySize`, `EosTokenIds`, `PadTokenId`, `AppliedOverlays`, `ProviderOptions`, `SerializesNativeCalls`, and `GuidanceDirectory` and `GuidanceMissingFiles` (guided decoding).
- `GenerationOptions`: `MaxNewTokens` (default 512, at least 1), `StopSequences` (non-empty strings), `AdditionalStopTokenIds`, `Sampling` (null: greedy), `PrefillChunkTokens` (null: one prefill), `ReturnLogprobs` and `TopLogprobs` (opt-in per-token log probabilities, top 0 to 20; experimental), `Guidance` (null: unconstrained; experimental; not combinable with `ReturnLogprobs`).
- `SamplingOptions`: `Temperature` (default 1.0, greater than 0), `TopK` (default 50), `TopP` (default 1.0, in (0, 1]), `Seed` (required, non-negative). The same seed, prompt, settings, model and device reproduce the same tokens.
- `GenerationResult`: `Text`, `TokenIds`, `FinishReason` (`Stop`, `StopSequence` or `Length`; cancellation throws instead), `StopTokenId`, `StopSequence`, `Usage` (`PromptTokens`, `CompletionTokens`, `TotalTokens`), `Timings` (`Prefill`, `Decode`), `Logprobs` (with `ReturnLogprobs`). `GenerateStreaming` yields `GenerationUpdate`s (`Text`, `TokenIds`, `Logprobs`); the last has `IsFinal` and `Result`.
- `TokenLogprob` (`TokenId`, `Text`, `Logprob`, `TopLogprobs` of `TopTokenLogprob`): one per completion token (the stop token has none), the natural-log probability from the log-softmax of the raw logits of the step that selected the token (before temperature, top-k and top-p). After a step, ONNX Runtime GenAI 0.17.1 reports the logits that selected the returned token, because it runs the model on that token only at the next step; the decode loop reads them right after the step. Each step then copies one logits row (vocabulary size x 4 bytes, about 1 MB for Qwen3.5).
- Guided decoding (experimental, opt-in): `GenerationGuidance.JsonSchema(schema)`, `Lark(grammar)` or `Regex(pattern)` as `GenerationOptions.Guidance`. GenAI masks the tokens the constraint does not allow before it selects each one, and allows only an end-of-sequence id once the constraint is complete. GenAI builds the token table from `tokenizer.json` and `tokenizer_config.json` in the model directory (`GenAIModel.GuidanceFiles`; `model.tokenizer_dir` when set). `GenAIModel.GuidanceMissingFiles` lists the ones missing, and a guided request on such a directory fails with `NotSupportedException` before any native call. `WithFastForward(maxTokensPerStep)` turns on GenAI's forced tokens: when the constraint leaves one continuation, GenAI appends it without a decode step per token. The loop returns those tokens in order, before the step's selected token, through the same token limit, stop ids, detokenization and stop sequences. GenAI does not check the sequence's maximum length for forced tokens, so the loop sizes the sequence for `MaxNewTokens` plus `maxTokensPerStep` (counted in the context-length check), and a step that appends more fails the request. The bound is the caller's to derive from the constraint. Without `Guidance` the loop is unchanged.
- Lifetime: dispose `GenAIModel`, `GenAISequence` and streaming enumerators (`foreach` does). As a safety net, a sequence or model that becomes unreachable undisposed (for example an abandoned `GenerateStreaming` enumerator) is released after a garbage collection: GenAI's own finalizers on these objects are suppressed, and the adapter's finalizer queues the same release as `Dispose` (generator under the native-call lock, then the model reference) to a thread-pool thread, never the finalizer thread. An abandoned request therefore no longer pins the model, but its native memory stays allocated until a collection finds it.
- `GenAIRuntimeInfo`: `PinnedVersion` (`0.17.1`), and `ManagedVersion` and `NativePackage` read from the application's `.deps.json` (null in single-file or AOT hosts).
- Exceptions: `GenAIRuntimeUnavailableException` (no native library, or packages other than 0.17.1), `GenAIProviderUnavailableException` (the provider cannot be used; no fallback), `GenAIModelException` (GenAI rejected the directory, its configuration or a step), `GenerationContextLengthException` (prompt plus `MaxNewTokens` exceed the context length; `PromptTokens`, `MaxNewTokens`, `ContextLength`).

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

## Native runtime

The package references `Microsoft.ML.OnnxRuntimeGenAI.Managed` (exactly 0.17.1; it asks for `Microsoft.Extensions.AI.Abstractions` 9.8.0 or later, which `Starlings.Generation.Abstractions` lifts to 10.10.1), `Starlings.Generation.Abstractions` and the owned tokenizer. The application references exactly one native package of the same version:

| Deployment | Package | Notes |
| --- | --- | --- |
| CPU | `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 | Brings `Microsoft.ML.OnnxRuntime` 1.30.0. Natives for linux-x64, linux-arm64, win-x64, win-arm64, osx-arm64; exercised on linux-x64 only. |
| CUDA | `Microsoft.ML.OnnxRuntimeGenAI.Cuda` 0.17.1 | Brings `Microsoft.ML.OnnxRuntime.Gpu` 1.30.0. linux-x64 and win-x64 natives (CUDA 13 build); the CUDA runtime and cuBLAS are not shipped. Exercised on linux-x64 only. |

The model directory must come from the onnxruntime-genai builder of the same version. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#generation-backend-and-pilot-llm-p-016).
