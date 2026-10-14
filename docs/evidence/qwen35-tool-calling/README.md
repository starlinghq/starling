# Qwen3.5-0.8B tool calling: evidence (2026-10-14)

Scope: tool calling in `Starlings.Models.Qwen35` and `Starlings.Models.Qwen35.Extensions.AI`. This covers the template's tools block, assistant tool calls and tool messages in the owned C# renderer, the parser of the template's tool-call format, and the `IChatClient` mapping (`AIFunction` tools, `FunctionCallContent`, `FunctionResultContent`, `ToolMode`, `FunctionInvokingChatClient`). Status: **experimental** (`Qwen35Capabilities`: `tool_calling`, `template.tools`, `template.tool_calls`, `template.tool_messages`, `tool_call_parsing`, `tool_call_parsing.streaming`, `extensions_ai.tools`, `extensions_ai.function_invocation`; manifests: `tool_calling`).

Machine: the M0 workstation (Linux x64, i7-5820K), CPU only, shared with other work.

## Source of truth

The pinned `chat_template.jinja` (`273d8e0e...`, `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`) defines the format, and it is treated as data. Its tool instructions ask for:

```
<tool_call>
<function=NAME>
<parameter=ARG>
VALUE
</parameter>
</function>
</tool_call>
```

History is rendered the same way. Argument values that are objects or arrays use transformers' `tojson`, and other values use Python `str()`. Tools are written with `tojson` (Python `json.dumps(ensure_ascii=False)`).

## Template (upstream renderer only)

`model-pipeline/reference/generation/make_template_corpus.py` now has 41 tool cases in five categories (`tools`, `tool-calls`, `tool-results`, `tool-thinking`, `tool-multi-turn`). They were rendered by transformers 5.18.0 `apply_chat_template` with jinja2 3.1.6, tokenizers 0.23.2 and markupsafe 3.0.4 on Python 3.12.14, with the snapshot's `tokenizer.json` (`5f9e4d49...`). The C# code produced nothing in the corpus. Before the change, regenerating the existing 85-case corpus was byte-identical to the committed file. The corpus is `model-pipeline/proposed/qwen3.5-0.8b/template/` (121 cases; see its `PROPOSED.md`).

| Check | Result |
| --- | --- |
| `TemplateCorpusTests` (typed API): rendering and owned-tokenizer ids | 105/105 renderings equal, 40 of them with tools, calls or results; 105/105 `prompt_ids` equal; 8/8 template errors with equal messages (1 with tools); 8 unsupported cases rejected |
| `TemplateCorpusChatClientTests` (`IChatClient`, tools as `AIFunctionDeclaration`, calls as `FunctionCallContent`, results as `FunctionResultContent`) | 106/106 prompts equal `prompt_ids`; 8 errors; 7 `NotSupportedException` |
| `TemplateTests` on `prompts-v1` | 22/22 renderings, the tools prompt included (it was rejected before) |

## Parser and client (scripted sequences, CI)

These tests run the real template, tokenizer, decode loop and detokenizer, with only the model's token choice scripted. The core tests are `Starlings.Models.Qwen35.Tests/ToolCallingTests` (54 tests); the client tests are `Starlings.Models.Qwen35.Extensions.AI.Tests/ToolCallingTests` (21 tests). They cover:

- one and several calls, and text before calls; argument casting with the tool's schema, as transformers 5.18.0 `_coerce` does (integer, number, boolean, null, object, array, `anyOf`, `nullable`, string-first type lists, undeclared parameters and tools kept as strings);
- malformed input returned as text, verbatim and without throwing: an unclosed block, a missing `</parameter>`, `</function>` or `<function=`, JSON-style and Python-style calls, two functions in one block, duplicate parameters, and text between tags;
- every two-way split of 12 completions, and per-character streaming, with thinking off and on, equal to the whole parse; no `<` in streamed text before a valid call; a call returned exactly when its block closes;
- a parsed completion rendered back as the model wrote it (with and without thinking);
- `json.dumps` and `repr` formatting (18 Python `repr` values), and explicit failures on duplicate keys, non-object arguments, lone surrogates, and tool calls on non-assistant messages;
- `FunctionCallContent` with `ChatFinishReason.ToolCalls`; streaming reassembled with `ToChatResponse` equal to the non-streamed response for 6 completions; `ToolMode` `None` and `Auto`; thinking; a call cut at the token limit (`Length`, text);
- `UseFunctionInvocation()` non-streaming and streaming, including two results in one tool message. The second prompt equals the typed rendering of question, call and result;
- 28 unsupported requests rejected before any generation (`RequireAny`, `RequireSpecific`, `AllowMultipleToolCalls = false`, hosted tools, duplicate names, malformed tool messages, among others).

## Real model (fp32x, CPU, greedy)

`Starlings.Models.Qwen35.Extensions.AI.Tests/RealModelTests`, with `STARLING_QWEN35_SNAPSHOT` set to the pinned bundle:

- `ToolsPromptGivesTheReferenceCallAsFunctionCallContent`: the prompts-v1 tools prompt ("What is the weather in Paris today?", one `get_weather(city)` tool) passed through `ChatOptions.Tools`. The prompt ids equal the reference. The 26 completion ids equal the transformers fp32 reference completion `<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>`. The response is one `FunctionCallContent` (`get_weather`, `city` = `Paris`) with `ToolCalls`, both streamed and non-streamed. Passed.
- `FunctionInvocationLoopCompletesOnTheRealModel`: `UseFunctionInvocation()` with `AIFunctionFactory.Create((string city) => ..., "get_weather", "Get the current weather for a city.")`. The function was invoked once with `Paris`. The middleware sent "18 C and sunny in Paris." back, and the model answered "Today in Paris, the weather is 18 degrees Celsius and sunny." (finish `Stop`). Passed, in 35 s for both tests.

This shows one prompt on one model size. It is not evidence that the 0.8B model picks tools reliably, fills arguments correctly in general, or avoids calling tools when it should not. No tool-use evaluation set exists yet.

## Not done

- A tool-use quality evaluation (dataset, metric, budget), which would be needed to qualify more than the mechanics.
- Real-model runs with thinking on, with several tools, or on CUDA (no GPU was used; the template, parser and mapping do not depend on the provider).
- Constrained decoding: `ToolMode` `RequireAny` and `RequireSpecific` are rejected, as is `AllowMultipleToolCalls = false`. Structured output stays unsupported.
- Unicode digits in numeric arguments: Python's `int()` and `float()` accept them, but the port keeps them as strings.
