# Guided decoding on CPU, one development run

Development evidence for the experimental guided decoding of `Starlings.Generation.OnnxGenAI` and `Starlings.Models.Qwen35` (backlog B1, #139). It was produced on 2026-10-26 on Linux x64 (Intel Xeon at 2.80 GHz, 4 vCPUs shared with other processes) in a cloud container with no GPU, with `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 (CPU). The package was built from branch `postponed/genai-guidance`.

**Model directory.** The published bundle's variant directories have neither `tokenizer.json` nor `tokenizer_config.json`, which GenAI needs to build the constraint's token table, so a guided request on them fails with `NotSupportedException`. The tests build a temporary directory (`tests/Shared/Qwen35/Qwen35GuidedDirectory.cs`) with:

- links to the three files of the fp32x variant (downloaded anonymously with `Qwen35ModelAssets.DownloadAsync` and verified against the lock);
- a copy of the bundle's `transformers-5.18.0/tokenizer.json` (SHA-256 `a5cd9732…`, the `qwen35-bpe-chat` asset);
- `tokenizer_config.json` of the checkpoint at the pinned revision `2fc06364715b967f1860aea9cf38778875588b17`, given by `STARLING_QWEN35_TOKENIZER_CONFIG` and checked against SHA-256 `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c`, the digest corvid spike S5 recorded.

**Run.** `tests/Starlings.Models.Qwen35.Tests`, class `GuidanceCpuModelTests` (greedy, thinking off), run with the xUnit in-process runner and `-showLiveOutput`. `cpu.log` is the raw output. The decode times are wall clock on a shared machine. They show the order of the fast-forward saving on this one run, not a measurement.

| Request | Constraint | Completion | Tokens | Decode |
| --- | --- | --- | ---: | ---: |
| "Write a product code." | regex `FORCED-[0-9]{3}` | `FORCED-101` | 6 | 1,063 ms |
| capital of France as JSON | JSON schema, `city` and `country` required, no other properties | `{"city": "Paris", "country": "France"}` (pretty-printed) | 19 | 3,193 ms |
| "Set a timer for 25 minutes." (two tools) | `Qwen35ToolGrammar`, one required call | `set_timer` with `minutes` 25 | 27 | 4,602 ms |
| the same | the same, fast-forward tokens | the same text | 27 | 1,526 ms |
| "What's the weather in Porto?" (two tools) | `Qwen35ToolGrammar`, one required call | `get_weather` with `city` Porto, `units` celsius | 40 | 6,810 ms |
| the same | the same, fast-forward tokens | the same text | 40 | 2,815 ms |

Each completion matches its constraint, ends with an end-of-turn token (`Stop`), and the tool calls parse with `Qwen35ToolCallParser`. With fast-forward tokens the text is the same as one token per step. The same class checks that the directory reports no missing guidance files, and that unguided greedy ids on it still equal the reference fixture (`en-short`).

`tests/Starlings.Models.Qwen35.Extensions.AI.Tests`, class `GuidedRealModelTests`, also passed on the same directory: `ChatToolMode.RequireAny` with two functions returned one `FunctionCallContent` for `get_weather` with finish reason `ToolCalls`, and a JSON schema `ResponseFormat` returned JSON with an integer `answer`.

This is one prompt per constraint. It shows that the constraint reaches GenAI's step and that the decode loop handles fast-forward steps on the real model. It does not qualify tool choice, JSON content or latency. Corvid spike S5 measured the one-token-per-step grammar on CUDA. Fast-forward tokens have no GPU measurement.
