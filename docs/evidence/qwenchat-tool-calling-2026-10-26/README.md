# QwenChat tool calling, one CPU run per profile

Development evidence for the experimental tool calling of `Starlings.Models.QwenChat`, produced on 2026-10-26 on Linux x64 (Intel Xeon at 2.80 GHz, 4 vCPUs, two native threads, the `QwenChatModelOptions` default) in a cloud container with no GPU. The bundles were downloaded anonymously from the public hosting (`QwenChatModelAssets.DownloadAsync`) and verified against the pinned digests. The package was built from branch `postponed/qwenchat-tool-calling`.

`ToolCallCheck.cs.txt` is the program (a console project with `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 and a project reference to the package). For each profile it asks "What is the weather in Paris right now?" with one `get_weather` tool, greedy, at most 64 new tokens. It parses the completion with `QwenChatToolCallParser`, repeats the request streamed through the incremental parser, then appends the parsed assistant turn and a tool result (`{"temperature_c": 18, "sky": "sunny"}`) and generates the next turn.

| Profile | Turn 1 | Streamed calls | Turn 2 | Log |
| --- | --- | ---: | --- | --- |
| `qwen3-0.6b/ortgenai-fp32-cpu` | `get_weather` with `{"city": "Paris"}`, `Stop`, 166 prompt and 19 completion tokens | 1 | "The current weather in Paris is sunny with a temperature of 18°C." (`Stop`, 16 tokens, no call) | `qwen3.log` |
| `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu` | `get_weather` with `{"city": "Paris"}`, `Stop`, 178 prompt and 19 completion tokens | 1 | "The current temperature in Paris is 18 degrees Celsius, and the sky is sunny." (`Stop`, 18 tokens, no call) | `qwen25.log` |

This is one prompt per profile. It shows that the rendered tools block, the parser and the tool-result turn work end to end on the real models. It does not qualify the models' choice of tools or their arguments, and no reference completion was compared (no transformers run of these prompts). The template renderings themselves are checked against transformers on the proposed corpora (`model-pipeline/proposed/qwen3-0.6b/template/`, `model-pipeline/proposed/qwen2.5-0.5b-instruct/template/`).
