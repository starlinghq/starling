# Clover-35B-A3B-Preview model definition: development evidence

Development evidence for `Starlings.Models.Clover` (`model-pipeline/admissions/clover.md`), produced on 2026-10-28 on Linux x64 (Intel Xeon at 2.10 GHz, 4 vCPUs) in a cloud container with no GPU. It is not release qualification: no gate ran, no fixture is protected, and nothing here is binding. The model itself did not run. No backend in this repository implements its block-diffusion decoder, and the checkpoint's weights were not downloaded.

## Inputs

- Checkpoint files: `Diffuspace/Clover-35B-A3B-Preview` at `4b0402b65e523d0183a14c431955438714cb16ff`: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `config.json`, downloaded at that revision and hashed (`environment-template.json`).
- Corpus: `model-pipeline/corpora/five-models/chat.jsonl` (14 cases, the one the Qwen and LFM3 chat candidates use) and `model-pipeline/corpora/clover/chat.jsonl` (16 cases). The Clover cases cover the template's defaults, each of `enable_thinking`, `preserve_thinking`, `add_non_thinking_prefix` and `add_generation_prompt` switched off, and reasoning split from content (two `</think>`, an unclosed `<think>`, an empty `reasoning_content`). They also cover a whitespace-only system turn, a `<tool_response>` user turn that is not a query, and the literal `<|mask|>` in user text.
- Reference environment: `model-pipeline/reference/clover/requirements.txt` (hash-locked; Python 3.12, transformers 5.19.0, the version the model card names, tokenizers 0.23.3; no torch).
- Owned tokenizer: `Starlings.Tokenizers` 0.3.2, the version this repository pins, on the checkpoint's `tokenizer.json`. The named profile `clover-bpe-chat` and its conformance fixtures are in starlinghq/tokenizers#20.

## Results

| Check | Compared with | Result |
| --- | --- | --- |
| Template text (`CloverChatTemplateTests.RendersLikeTheReference`, in CI) | transformers 5.19.0 `apply_chat_template` on the checkpoint's template (`run_clover_template.py`) | 30/30 renderings equal |
| Prompt ids (`CloverChatTemplateTests.PromptIdsMatchTheReferenceTokenizer`, needs `tools/fetch-tokenizer-assets.sh`) | the transformers tokenizer (`Qwen2Tokenizer`), `add_special_tokens=False` | 30/30 equal, including the literal `<|endoftext|>`, `<|im_start|>`, `<|im_end|>` and `<|mask|>` (151669) in user text |
| Runtime pipeline | transformers 5.19.0 `backend_tokenizer.to_str()` against the Hub `tokenizer.json` | equal as JSON: the asset is the Hub file |

## Not done

- **Generation.** No backend; see the admission's "Execution route".
- **Tool calling.** The template's tool sections are not ported; the port rejects tools, tool calls and tool turns.

## Files

| File | Contents |
| --- | --- |
| `environment-template.json` | Runner environment: pin, tokenizer class, template, tokenizer and tokenizer-config digests, corpus digests, package versions |

The renderings and prompt ids are `ai-dotnet/tests/Starlings.Models.Clover.Tests/Data/template-reference.jsonl` (byte-identical to the runner's `template.jsonl`).

Commands (paths shortened):

```sh
cd model-pipeline
python -I reference/clover/run_clover_template.py --checkpoint <snapshot with the four files> \
    --corpus corpora/five-models/chat.jsonl --corpus corpora/clover/chat.jsonl --out <out>
cd .. && tools/fetch-tokenizer-assets.sh
dotnet test --project ai-dotnet/tests/Starlings.Models.Clover.Tests
```
