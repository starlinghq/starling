# Proposed fixture: chat-template corpus (P-029)

Status: **proposed**. Not accepted, and not under `model-pipeline/protected/`. If the maintainer accepts it, the intended destination is `model-pipeline/protected/qwen3.5-0.8b/template/`, with the files byte-identical. The tests read the protected copy when it exists and this proposal otherwise (`ai-dotnet/tests/Shared/Qwen35/Qwen35TemplateCorpus.cs`).

This is the template corpus of backlog P-029 for `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`. Its pinned `chat_template.jinja` has SHA-256 `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`. The corpus extends the 21 text prompts of `prompts-v1.jsonl` to 121 cases, which cover what the owned C# renderer `Qwen35ChatTemplate` claims to support and each feature it rejects. 41 cases cover tool calling: tool definitions, assistant tool calls and tool results (added 2026-10-14, when the renderer gained them).

## Reference source

`model-pipeline/reference/generation/make_template_corpus.py` defines the cases and renders each one with the upstream reference renderer: transformers `apply_chat_template` (a Jinja2 sandbox) with the checkpoint's own template. It runs in the environment of `model-pipeline/reference/generation/requirements.txt` (transformers 5.18.0, tokenizers 0.23.2, jinja2 3.1.6, markupsafe 3.0.4, Python 3.12.14). Prompt ids come from the transformers tokenizer of the snapshot (`tokenizer.json` `5f9e4d49...`). Nothing in the corpus was produced by the C# renderer. `environment.json` records the package versions, the snapshot asset digests and the corpus digest. Regenerating the corpus is byte-identical.

```sh
python make_template_corpus.py --model-dir <Qwen3.5-0.8B snapshot> --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --out model-pipeline/proposed/qwen3.5-0.8b/template
```

## Contents

`template-corpus.jsonl` (format `starlings.template-corpus` version 1) has one case per line. Each case has these fields: `id`, `category`, `messages`, `tools`, `chat_template_kwargs`, `add_generation_prompt`, `owned`, `unsupported_feature`, `rendered`, `error`, `error_source` and `prompt_ids`. The `owned` field states what the owned renderer must do with the case:

| `owned` | Cases | Requirement |
| --- | ---: | --- |
| `render` | 105 | The rendering must equal `rendered` exactly, and the owned tokenizer's ids must equal `prompt_ids`. |
| `template_error` | 8 | Rendering must fail with `Qwen35TemplateException`. When the template raised the error (`error_source` `template`), the message must be the same. One case, the empty conversation, is rejected by transformers before the template runs; the owned renderer uses the template's own message. |
| `unsupported` | 8 | The feature is outside the supported subset. The typed API cannot express it or rejects it. The `IChatClient` adapter throws `NotSupportedException` before any generator is created. `rendered` or `error` records what upstream does. |

| Category | Cases | What it covers |
| --- | ---: | --- |
| `multi-turn` | 10 | Two, three and 41 turns; consecutive user or assistant turns; a conversation ending on one or two assistant turns; an assistant turn first; thinking on in a multi-turn conversation |
| `system` | 9 | Basic, empty, whitespace-only, multiline and think-tag system prompts; system-only, second, late and doubled system messages (errors) |
| `enable_thinking` | 5 | Undefined, explicit false and true, true with a system prompt, true with history |
| `history-think` | 21 | Historical `<think>` stripping before the last query: no newlines, close tag only, open tag only, two blocks, an extra close tag, a prefix, reasoning only, an empty block, uppercase lookalikes and CRLF. Also: `reasoning_content` dropped, kept, empty or overriding the tags; reasoning kept after the last query; `<tool_response>` user turns that move the last query (padded and unwrapped); all-tool-response conversations (errors) |
| `whitespace` | 11 | Empty and whitespace-only user and assistant content; every character Python's `str.strip()` removes (U+0009 to U+000D, U+001C to U+0020, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F, U+3000) and some it keeps (U+200B, U+FEFF, U+180E, U+2060); inner whitespace; CRLF; NUL and control characters |
| `unicode` | 9 | CJK, RTL (Arabic, Hebrew), combining marks, ZWJ emoji, astral characters, special-token text typed by the user, Jinja syntax in content, line separators, Cyrillic with thinking on |
| `long` | 5 | A 54,000-character user message (13,702 tokens), a 20,000-character system prompt, 8,000-character historical and kept reasoning, and 201 turns |
| `tools-empty` | 1 | `tools=[]`, which the template treats as no tools |
| `tools` | 12 | One tool; with a system prompt, an empty, whitespace-only and multiline one; four tools; a nested JSON schema (`$defs`/`$ref`, `anyOf`, `enum` with null, objects in arrays, Python float and big-int formatting such as `1e+16`, `1e-07`, `-0.0`, every character `json.dumps` escapes, non-ASCII text, `</tools>` in a description); no description; no parameters; thinking on and off; a tool prompt after history |
| `tool-calls` | 11 | One call with empty, null, whitespace-only and non-empty text; the same without tools; two and three calls in a turn; every argument type (`str()` of ints, floats, `True`, `None`; `tojson` of lists and objects; strings with newlines, padding, markup); empty and missing arguments; an undeclared function |
| `tool-results` | 9 | Results with whitespace, empty, and markup-like text; results followed by a user turn; two results then a user turn; results after an assistant turn without calls; a result as the first message and after the system message; a tool-only conversation (error) |
| `tool-thinking` | 5 | Thinking on with `reasoning_content`, with `<think>` in the content, with text after the think block; thinking off; reasoning before the last query dropped |
| `tool-multi-turn` | 4 | A weather conversation with parallel calls, results, an answer, a follow-up question and a second call; with a final answer; thinking on; thinking on with reasoning on two consecutive calls |
| `template-error` | 1 | The empty conversation |
| `unsupported` | 8 | An image, a video, an image in a system message, `add_vision_id`, `add_generation_prompt=False`, an unknown role (`developer`), `reasoning_content` on a user message (upstream ignores it; the package refuses it), and content as a list of text parts (unsupported by the typed API; the `IChatClient` adapter concatenates text parts, as the template does, and must equal the upstream rendering) |

## Result with tool calling (2026-10-14)

On `qwen-tool-calling`, all 121 cases pass:

- `Qwen35ChatTemplate`: 105 renderings equal (40 with tools, tool calls or tool results); 8 errors with equal messages; 8 unsupported cases rejected or not expressible. The owned tokenizer gives `prompt_ids` on all 105 renderings.
- `Qwen35ChatClient`: the 106 rendering cases (105 plus the text-part list) reach the model as `prompt_ids`, with tools given as `AIFunctionDeclaration`, calls as `FunctionCallContent` and results as `FunctionResultContent`. All 8 template errors raise `Qwen35TemplateException`. The other 7 unsupported cases raise `NotSupportedException`.

The tool cases replace the five earlier `uns-tools`, `uns-tools-system`, `uns-tool-message`, `uns-two-tool-messages` and `uns-tool-calls` cases, which were rejected while tools were unsupported. Tool cases of the same shapes replace them (`tools-one`, `tools-one-system`, `call-one-no-tools`, `result-without-call`, `call-with-text` and `result-then-user`), with their own messages and upstream renderings.

## Result at the time of the proposal

On `r8-qwen-gaps`, all 85 cases pass, with no change to the renderer:

- `Qwen35ChatTemplate`: 65 renderings equal; 7 errors with equal messages; 13 unsupported cases rejected or not expressible. The owned tokenizer gives `prompt_ids` on all 65 renderings.
- `Qwen35ChatClient`: the 66 rendering cases (65 plus the text-part list) reach the model as `prompt_ids`. All 7 template errors raise `Qwen35TemplateException`. The other 12 unsupported cases raise `NotSupportedException`.

The tests are `TemplateCorpusTests` (`Starlings.Models.Qwen35.Tests`) and `TemplateCorpusChatClientTests` (`Starlings.Models.Qwen35.Extensions.AI.Tests`).

## Decisions for the reviewer

1. Accept the corpus as development fixtures. Like `prompts-v1.jsonl`, the cases were written by an agent; they are not a protected release set (P-030).
2. The `owned` and `unsupported_feature` fields are the agent's statement of the supported subset, taken from `Qwen35ChatTemplate`'s documentation. They are not upstream data. The upstream data is `rendered`, `error`, `error_source` and `prompt_ids`.
3. P-029's acceptance reads "rendered prompts match the upstream reference on the template corpus; unsupported template features fail explicitly". Whether this corpus is enough to move `template.owned_renderer.text` and `template.enable_thinking` beyond experimental is the maintainer's decision. The proposal does not change their status.
