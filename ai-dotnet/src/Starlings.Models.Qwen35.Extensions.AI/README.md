# Starlings.Models.Qwen35.Extensions.AI

**Preview.** [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) `IChatClient` over the `Qwen/Qwen3.5-0.8B` chat model in `Starlings.Models.Qwen35`: the same template, tokenizer, decode loop and tokens, usable with Microsoft.Extensions.AI middleware and dependency injection. The API may still change.

## Install

```sh
dotnet add package Starlings.Models.Qwen35.Extensions.AI --prerelease
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.17.1        # CPU natives; or Microsoft.ML.OnnxRuntimeGenAI.Cuda (exactly one)
```

## Use

```csharp
using Starlings.Models.Qwen35;
using Starlings.Models.Qwen35.Extensions.AI;
using Microsoft.Extensions.AI;

Qwen35ModelFiles files = Qwen35ModelAssets.OpenLocal(Qwen35Variant.Fp32x);   // offline, verified, never downloads
using IChatClient client = Qwen35ChatClient.Create(files);

ChatResponse response = await client.GetResponseAsync("What is the capital of France?", new ChatOptions { MaxOutputTokens = 128 });
Console.WriteLine($"{response.Text} [{response.FinishReason}, {response.Usage?.InputTokenCount} + {response.Usage?.OutputTokenCount} tokens]");

await foreach (ChatResponseUpdate update in client.GetStreamingResponseAsync("Why is the sky blue?"))
{
    Console.Write(update.Text);
}

// Or with DI: services.AddQwen35ChatClient(files);
```

### Tool calling (experimental)

```csharp
var getWeather = AIFunctionFactory.Create((string city) => $"18 C and sunny in {city}.", "get_weather", "Get the current weather for a city.");
using IChatClient agent = new ChatClientBuilder(client).UseFunctionInvocation().Build();
ChatResponse answer = await agent.GetResponseAsync("What is the weather in Paris today?", new ChatOptions { Tools = [getWeather] });
Console.WriteLine(answer.Text);   // on the 0.8B model, greedy: "Today in Paris, the weather is 18 degrees Celsius and sunny."
```

- `ChatOptions.Tools`: each `AIFunctionDeclaration` (any `AIFunction`) is rendered into the template's tools block as `{"type": "function", "function": {"name", "description", "parameters": JsonSchema}}`. Other tool types and duplicate names are rejected.
- `ToolMode`: `Auto` (or unset) and `None` are supported; `None` renders no tools and parses nothing. `RequireAny`, `RequireSpecific` and `AllowMultipleToolCalls = false` are rejected with `NotSupportedException`: the template cannot require a call and decoding is not constrained.
- History: `FunctionCallContent` in assistant messages is rendered as the template's `<tool_call>` blocks after the message text; each `FunctionResultContent` of a tool message becomes one template tool message (strings as is, other results as compact JSON), so several results render as consecutive `<tool_response>` blocks.
- Responses: tool-call blocks become `FunctionCallContent` (a new `CallId`, `Name`, `Arguments` as `JsonElement` values cast with the tool's schema; `RawRepresentation` is the `Qwen35ToolCall`), and the finish reason is `ToolCalls`. Text before a call is `TextContent`. A malformed or unfinished block is returned as text, verbatim, with finish reason `Stop` or `Length`; the client never throws on model output. With thinking enabled, the reasoning up to `</think>` stays text and is not parsed.
- Streaming: text deltas are emitted as usual; text that could start a tool call is held back, and each `FunctionCallContent` is emitted once its block is complete, so no `<tool_call>` markup leaks for a valid call. The updates reassemble (`ToChatResponse`) to the non-streamed response.
- A 0.8B model does not always call tools when it should, and may produce a malformed call or call an undeclared function; check calls before running them. Structured output (`ResponseFormat` JSON) stays unsupported.

### Messages and options

- Roles system, user, assistant and tool. Text content; `TextReasoningContent` and `FunctionCallContent` in assistant messages; `FunctionResultContent` in tool messages. Images and other content are rejected with `NotSupportedException`.
- `MaxOutputTokens`, `StopSequences`, `Temperature`/`TopP`/`TopK` with an explicit `Seed` (greedy without them; sampling without a seed is rejected), and `AdditionalProperties["enable_thinking"]`. Every other option that would change the result is rejected instead of being ignored.
- Streaming updates are complete-character text deltas; they concatenate to the non-streamed text, and the last update carries the finish reason (`Stop`, `Length` or `ToolCalls`) and usage (stop token excluded).
- Each request runs on a dedicated background thread (ONNX Runtime GenAI steps are blocking native calls). Cancellation takes effect between prefill chunks and decode steps.

Status: Linux x64 CPU and CUDA experimental. See the [documentation and evidence](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#microsoftextensionsai-chat-client-p-017).
