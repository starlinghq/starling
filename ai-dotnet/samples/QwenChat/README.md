# QwenChat sample

Streaming console chat with `Qwen/Qwen3.5-0.8B`, using only the published packages (`Starlings.Models.Qwen35`, which brings `Generation.OnnxGenAI`, `Acquisition` and `Tokenizers`, and `Starlings.Models.Qwen35.Extensions.AI` for the optional `IChatClient` path) and the ONNX Runtime GenAI CPU natives (`Microsoft.ML.OnnxRuntimeGenAI` 0.17.1). Requires the .NET 10 SDK. The profiles `qwen3.5-0.8b/ortgenai-fp32x-cpu`, `ortgenai-fp32x-cuda` and `ortgenai-fp16x-cuda` are Stable; maturity is per profile (the `Starlings.Models.Qwen35` README). This sample is not the experimental `Starlings.Models.QwenChat` package, which is not published.

1. `download`: explicit acquisition of the pinned files (`chat_template.jinja`, the runtime `tokenizer.json` and the ONNX Runtime GenAI build that Starling produces from the pinned commit) from the Hugging Face hosting repository `Starlinghq/qwen3.5-0.8b-ortgenai` at its pinned commit. The only step that uses the network.
2. `chat`: offline. Verifies the files, loads the model and runs an interactive chat: the answer streams token by token, Ctrl+C stops it at the next decode step, `/reset` clears the conversation, an empty line or `/exit` quits. Each turn sends the whole conversation (the chat template re-renders it); the finish reason, token usage and decode speed are printed after each answer.
3. `ask`: offline, one streamed answer to the question on the command line (default: "What is the capital of France? Answer in one sentence."). It takes the same options as `chat`.

```sh
dotnet run -- download [--cache <dir>] [--variant fp32x|fp16x]     # fp32x about 3.1 GB, fp16x about 1.6 GB
dotnet run -- chat [--cache <dir> | --snapshot <dir>] [--variant fp32x|fp16x] [--cuda] [--thinking] [--max-new 512] [--chat-client]
dotnet run -- ask  [same options as chat] "What is the capital of France?"
```

Without `--cache` the files go to `$STARLING_MODELS` or the user cache directory (`~/.cache/starlings/models` on Linux). The default variant is fp32x.

`--chat-client` streams through the Microsoft.Extensions.AI `IChatClient` adapter (`Starlings.Models.Qwen35.Extensions.AI`, `Qwen35ChatClient`) instead of the typed API: the same tokens, with the finish reason and usage read from the last `ChatResponseUpdate`. In short:

```csharp
using IChatClient client = new Qwen35ChatClient(model);                      // wraps the loaded model; owns nothing
await foreach (ChatResponseUpdate update in client.GetStreamingResponseAsync("What is the capital of France?", new ChatOptions { MaxOutputTokens = 64 }))
{
    Console.Write(update.Text);                                              // FinishReason and UsageContent on the last update
}
```

`--thinking` sets the template's `enable_thinking`: the model writes its reasoning and `</think>` before the answer, all of it printed. `--cuda` needs `Microsoft.ML.OnnxRuntimeGenAI.Cuda` instead of the CPU package in `QwenChat.csproj` (reference exactly one) and the CUDA 13 runtime; on CUDA use `--variant fp16x` (the fp32x graph runs there but is slow). On CPU only fp32x is supported (about 6 tokens/s decode on a 6-core AVX2 machine).

The hosting repository is public: `download` needs no token. Files already in the snapshot directory (`<cache>/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17/`, layout `chat_template.jinja`, `ortgenai-fp32x/`, `transformers-5.18.0/tokenizer.json`) are verified and reused; `--snapshot <dir>` uses a directory in that layout. The files can also be built with `model-pipeline/reference/generation/build_ortgenai_bundle.py`; the lock checks every SHA-256.

The sample does not demonstrate tool calling (experimental in 0.2.0); for the `ChatOptions.Tools` and `UseFunctionInvocation` pattern, see the `Starlings.Models.Qwen35.Extensions.AI` README.

The sample uses `PackageReference`s only, no project references: `Starlings.*` 0.3.0 (the `StarlingsVersion` property) restores from nuget.org. `tools/release-drill-ai.sh` also packs the packages into a local feed and builds this sample from a copy outside the repository, and, when given a CUDA device, a copy on `Microsoft.ML.OnnxRuntimeGenAI.Cuda`.
