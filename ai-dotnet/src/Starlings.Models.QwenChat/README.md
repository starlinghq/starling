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

These are experimental text profiles. Qwen3 explicitly uses `enable_thinking=false`, which differs from its upstream default. Qwen2.5 inserts the upstream default system instruction when the first message is not a system message. Tool calling is experimental (see "Tool calling"). Multimodal input, sampling and Qwen3 thinking mode are unsupported. Each request has its own disposable generator/cache; continuing chat requires resending the whole conversation. CPU native threads are bounded to two. Cancellation is checked before prefill and between decode steps; a running native operation is not interruptible.

Both native paths overlay `model.pad_token_id=151935`, the last embedding row, after verifying neither original vocabulary nor added tokens maps to it (maximum tokenizer ids: Qwen3 151668; Qwen2.5 151664). GenAI otherwise treats literal `<|endoftext|>` id 151643 in prompt text as padding. Original configuration and graph files are never modified by this runtime overlay.

Fresh reference and parity reports remain proposals. They do not award protected G1/G2 qualification, approve releases, or change support status.

## Tool calling (experimental)

Both templates define tools the same way: a tools block in the system turn (`# Tools`, the definitions inside `<tools></tools>`, one JSON object per line), assistant tool calls as JSON inside `<tool_call></tool_call>`, and tool results as `<tool_response>` blocks, consecutive ones in one user turn. `QwenChatTemplate` ports these sections of both pinned templates, with the instructions copied verbatim.

```csharp
using System.Text.Json;
using Starlings.Models.QwenChat;

var weather = QwenChatTool.Function("get_weather", "Current weather for a city.",
    JsonDocument.Parse("""{"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}""").RootElement);
var options = new QwenChatOptions { MaxNewTokens = 64, Tools = [weather] };
var messages = new List<QwenChatMessage> { new(QwenChatRole.User, "What is the weather in Paris right now?") };
QwenChatParsedOutput reply = QwenChatToolCallParser.Parse(model.Generate(messages, options).Text);
if (reply.ToolCalls.Count > 0)
{
    // Run each call (call.Name, call.Arguments as a JsonElement object), then send the results back.
    messages.Add(reply.ToAssistantMessage());
    messages.Add(new(QwenChatRole.Tool, """{"temperature_c": 18}"""));
    Console.WriteLine(model.Generate(messages, options).Text);
}
```

- **Definitions.** `QwenChatTool.Function(name, description, parameters)` builds `{"type": "function", "function": {...}}`; `QwenChatTool.FromJson` takes any definition object as given. Each is written as transformers' `tojson` writes it (Python `json.dumps`, keys in order, non-ASCII kept). Pass them as `QwenChatOptions.Tools` to `Generate`, or to `CreatePrompt(messages, tools)` and `QwenChatTemplate.Render(profile, messages, tools)`. An empty list renders no tools block, as upstream.
- **Calls and results.** `QwenChatMessage.ToolCalls` (assistant only) holds `QwenChatToolCall(Name, Arguments)`; `Arguments` must be a JSON object. A `QwenChatRole.Tool` message holds one result. Qwen3 keeps its think-block rules around tool calls; Qwen2.5 ignores reasoning content.
- **Parsing.** `QwenChatToolCallParser` splits a completion into text and calls, whole (`Parse`) or from streamed deltas (`Append`, `Complete`). A block must hold one strict JSON object with exactly `name` and `arguments` (an object); a closing tag inside a JSON string does not end the block. A malformed or unfinished block is returned as text, verbatim; the parser never throws on model output. Whitespace-only text around calls is dropped.
- **Rejected** with `NotSupportedException`: tool calls or reasoning on a non-assistant message, arguments that are not a JSON object (upstream raises for missing arguments, and the two templates render a JSON string of arguments differently), call names with `"`, `\` or a control character (the templates write the name unescaped into the JSON), and tool data with duplicate keys or lone surrogates. `QwenChatTool.Function` and `FromJson` reject such names with `ArgumentException`.
- **Not done.** The Qwen3.5 package's parser (`Qwen35ToolCallParser`) reads the XML parameter form of the Qwen3.5 template, which neither of these templates uses, so it is not reused. There is no `IChatClient` adapter for these profiles, no constrained decoding and no required tool choice.

Evidence (2026-10-26). The proposed template corpora (`model-pipeline/proposed/qwen3-0.6b/template/` and `model-pipeline/proposed/qwen2.5-0.5b-instruct/template/`, written by `model-pipeline/reference/qwen-chat/make_template_corpus.py` with transformers 5.18.0 `apply_chat_template`) hold 46 Qwen3 and 42 Qwen2.5 render cases; 37 and 33 of them have tools, tool calls or tool results. The owned renderer equals every rendering, and the owned tokenizer gives every `prompt_ids` (`TemplateCorpusTests`; the id check runs when `tools/fetch-tokenizer-assets.sh` has fetched the two tokenizers). The unsupported cases are rejected or cannot be expressed. Every rendered call block parses back to the same call, whole and streamed one character at a time. One greedy CPU run per profile (`docs/evidence/qwenchat-tool-calling-2026-10-26/`) produced a well-formed `get_weather` call for Paris and, after the tool result, an answer that used it. The model's choice of tools is not otherwise qualified.

## Bundle

The bundles are hosted publicly, one Hugging Face repository per profile, at a pinned commit (tag `r1`): `Starlinghq/qwen3-0.6b-ortgenai` (`c24cd59c`) for Qwen3 (2.4 GB) and `Starlinghq/qwen2.5-0.5b-instruct-ortgenai` (`6b6957c8`) for Qwen2.5 (2.0 GB), with the original `LICENSE`, a `NOTICE` and `SHA256SUMS`. `QwenChatModelAssets.Lock(profile)` is the acquisition lock (`Starlings.Acquisition`): the six pinned files, each sourced from `HostingBaseUrl(profile)`. The digests, not the address, pin the bytes.

To build a bundle instead, `build_bundle.py --candidate <profile.json> --snapshot <original snapshot> --out <bundle>` needs the `onnxruntime-genai` 0.17.1 builder; the candidate contracts are `model-pipeline/candidates/qwen3/profile.json` and `model-pipeline/candidates/qwen25/profile.json`. `Load` requires exactly these files in the bundle directory: `genai_config.json`, `model.onnx`, `model.onnx.data`, `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`. Each file's size and SHA-256 must equal the profile's pin (`QwenChatProfile.BundleFiles`, the same digests as the registry manifest's `execution.graph.assets`) as hashed on disk. A downloaded bundle has no `bundle-lock.json`; when one is present (a local build), its entries must equal the pin too, and it must also name the profile, the original repository and revision, builder 0.17.1 at source commit `83de55a1`, fp32 and `CPUExecutionProvider`. A mismatch throws `InvalidDataException`.

The build is reproducible: on 2026-10-18 two builds from the pinned originals with the hash-locked builder (`model-pipeline/reference/generation/requirements.txt`) gave the pinned digests byte for byte, for both profiles. A builder wheel or platform outside that lock was not tested, and a bundle with other digests is refused.

## API

- `QwenChatProfile.Qwen3` (`qwen3-0.6b/ortgenai-fp32-cpu`, context 40,960 tokens) and `QwenChatProfile.Qwen25` (`qwen2.5-0.5b-instruct/ortgenai-fp32-cpu`, context 32,768 tokens); `QwenChatProfile.FromId(id)` resolves either id and throws `NotSupportedException` otherwise.
- `QwenChatModelAssets.DownloadAsync(profile, options, progress, cancellationToken)` and `OpenLocal(profile, options)` return the verified bundle directory (`ModelAcquisitionOptions`: cache directory, verification); `HostingRepository`, `HostingRevision`, `HostingBaseUrl` and `Lock` per profile.
- `QwenChatModel.Load(profile, bundleDirectory, options)`: `QwenChatModelOptions.IntraOpThreads` is 1 or 2 (default 2).
- `CreatePrompt(messages)` renders the owned template and returns `QwenChatPrompt(Text, TokenIds)`. `Generate(messages | prompt, options, cancellationToken)` returns a `GenerationResult` (`Starlings.Generation.OnnxGenAI`); `GenerateStreaming(prompt, options, cancellationToken)` yields `GenerationUpdate`s with the same stop and usage contract. Generation is greedy.
- `QwenChatOptions`: `MaxNewTokens` (default 16), `AdditionalStopTokenIds`, `StopSequences`, `PrefillChunkTokens` (default null: one prefill), `Tools` (default none; experimental).
- `QwenChatMessage(Role, Content, ReasoningContent)` with `ToolCalls`: roles `System`, `User`, `Assistant` and `Tool`. `ReasoningContent` and `ToolCalls` are accepted only on assistant history; elsewhere they throw `NotSupportedException`.
- `CreatePrompt(messages, tools)`, `QwenChatTemplate.Render(profile, messages, tools)`, `QwenChatTool`, `QwenChatToolCall`, `QwenChatToolCallParser` (`QwenChatParsedOutput`, `QwenChatOutputPart`): see "Tool calling".
- `Info` (`QwenChatModelInfo`): profile, repository, revision, tokenizer and template SHA-256, GenAI version, native package, context length, and the capability values (`TextGeneration` and `Tools` experimental, `Multimodal` unsupported, `Cache` fresh-request).
- `Tokenizer`, `Profile`, and `CreateSequence(GenAISequenceOptions)` (advanced, an independent disposable sequence).
