# Proposed fixture: chat-template corpus for `Qwen/Qwen2.5-0.5B-Instruct` (QwenChat)

Status: **proposed**. Not accepted, and not under `model-pipeline/protected/`. If the maintainer accepts it, the intended destination is `model-pipeline/protected/qwen2.5-0.5b-instruct/template/`, with the files byte-identical. The tests read the protected copy when it exists and this proposal otherwise (`ai-dotnet/tests/Starlings.Models.QwenChat.Tests/TemplateCorpus.cs`).

This is a template corpus in the pattern of backlog P-029 for the Qwen2.5 profile of `Starlings.Models.QwenChat`: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`, whose `tokenizer_config.json` `chat_template` string (the bundle's `chat_template.jinja`) has SHA-256 `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`. It was written on 2026-10-26, when `QwenChatTemplate` gained the template's tool sections. 49 cases cover what the owned C# renderer claims to support and each input it rejects; 33 of the render cases have tools, tool calls or tool results.

## Reference source

`model-pipeline/reference/qwen-chat/make_template_corpus.py --profile qwen25` defines the cases and renders each one with the upstream reference renderer: transformers `apply_chat_template` (a Jinja2 sandbox) with the snapshot's own template. It checks the revision and the template digest first. It ran with transformers 5.18.0, tokenizers 0.23.2, jinja2 3.1.6 and markupsafe 3.0.4 (the versions of `model-pipeline/reference/generation/requirements.txt`, installed without torch, which rendering does not need) on Python 3.12.3. The snapshot held the Hub's `tokenizer.json` (`c0382117...`) and `tokenizer_config.json` at the pinned revision. Prompt ids come from the transformers tokenizer of that snapshot. Nothing in the corpus was produced by the C# renderer. `environment.json` records the package versions, the asset digests and the corpus digest. Regenerating the corpus gave byte-identical output (Qwen3 checked on Python 3.13.17 and 3.12.3).

```sh
python make_template_corpus.py --profile qwen25 --model-dir <snapshot> --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --out model-pipeline/proposed/qwen2.5-0.5b-instruct/template
```

The template has no thinking switch, so no template arguments are passed. It inserts its default system instruction, also in the tools block, when the first message is not a system message, and ignores `reasoning_content`.

## Contents

`template-corpus.jsonl` (format `starlings.template-corpus` version 1, as `model-pipeline/proposed/qwen3.5-0.8b/template/`) has one case per line with the fields `id`, `category`, `messages`, `tools`, `chat_template_kwargs`, `add_generation_prompt`, `owned`, `unsupported_feature`, `rendered`, `error`, `error_source` and `prompt_ids`.

| `owned` | Cases | Requirement |
| --- | ---: | --- |
| `render` | 42 | The rendering must equal `rendered` exactly, and the owned tokenizer's ids must equal `prompt_ids`. |
| `unsupported` | 7 | The input is outside the supported subset. The typed API cannot express it or rejects it with `NotSupportedException`. `rendered` or `error` records what upstream does. |

| Category | Cases | What it covers |
| --- | ---: | --- |
| `text` | 8 | Text conversations the port supported before tools: user only, system (also empty), multi-turn, a late system message, whitespace, special-token text, a `<tool_response>` user message |
| `tools` | 11 | `tools=[]` (no tools block), one tool with and without a system prompt (also empty and multiline), four tools, a nested JSON schema (`$defs`/`$ref`, `anyOf`, `enum` with null, Python float and big-int formatting such as `1e+16`, `1e-07`, `-0.0`, every character `json.dumps` escapes, non-ASCII text, `</tools>` in a description), no description, no parameters, history, a late system message |
| `tool-calls` | 12 | One call with empty, whitespace-only and non-empty text; the same without tools; the call as the last message (with and without text); two and three calls in a turn; every JSON argument type (ints, floats, big ints, Booleans, null, lists, objects, strings with escapes, markup and the closing tag); empty arguments; an undeclared function; reasoning content on a call turn |
| `tool-results` | 9 | Results with whitespace, empty and markup-like text; results followed by a user turn; two results then a user turn; results after an assistant turn without calls; a result as the first message, after the system message, and as the only message |
| `tool-multi-turn` | 2 | A weather conversation with parallel calls, results, an answer, a follow-up and a second call; two sequential tool steps |
| `unsupported` | 7 | Arguments as a JSON string, missing arguments (upstream raises), a call name with a quote, image content, an unknown role, `add_generation_prompt=False`, `reasoning_content` on a user message (upstream ignores it; the package refuses it) |

## Result (2026-10-26)

On `postponed/qwenchat-tool-calling`, all 49 cases pass (`TemplateCorpusTests` in `Starlings.Models.QwenChat.Tests`): 42 renderings equal, the owned tokenizer gives `prompt_ids` on all of them, and the 7 unsupported cases are rejected or not expressible. For every assistant turn with tool calls, the text the model would generate for that turn parses back with `QwenChatToolCallParser` to the same calls and text, whole and streamed one character at a time.
