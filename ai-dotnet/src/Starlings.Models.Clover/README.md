# Starlings.Models.Clover

The model definition of [`Diffuspace/Clover-35B-A3B-Preview`](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview) at revision `4b0402b65e523d0183a14c431955438714cb16ff`, without a runtime. It holds what a backend needs to prompt the model and check a build against the checkpoint. No backend in this repository runs Clover: it is a block-diffusion language model (a 35B mixture-of-experts fine-tune of `Qwen/Qwen3.6-35B-A3B` with about 3B active parameters per token). It fills blocks of 32 masked positions in parallel, and neither ONNX Runtime GenAI nor any published ONNX export implements that decoder. The reasons and the open decisions are in `model-pipeline/admissions/clover.md`.

This library is **experimental**, source only and not admitted.

- `CloverChatTemplate`: the checkpoint's `chat_template.jinja` rendered in C# for text turns (system, user, assistant with reasoning). It carries the template's three switches with their upstream defaults: `enable_thinking` (true), `preserve_thinking` (true) and `add_non_thinking_prefix` (true). `CloverChatTemplate.Instance` is the same renderer behind `IChatTemplate`.
- `CloverCheckpoint`: repository and revision; the special token ids (`<|mask|>` 151669, `<|endoftext|>` 248044, `<|im_start|>` 248045, `<|im_end|>` 248046, `<think>` 248068, `</think>` 248069); the owned tokenizer profile `clover-bpe-chat` and its digest; `Profile`, a `ModelProfile` with vocabulary 248,320, hidden size 2,048, 40 layers, head size 256, context 131,072, end-of-sequence ids 248044 and 248046, Apache-2.0.
- `CloverCheckpoint.Lock`: the asset lock of the Hub checkpoint's model files (the 26 bf16 safetensors shards and their index, the two bidirectional GDN parameter files, configs, tokenizer and template; about 69.3 GB). Nothing downloads unless you pass it to `ModelAssetStore.DownloadAsync`.

```csharp
using Starlings.Generation;
using Starlings.Models.Clover;
using Starlings.Tokenizers;

string text = CloverChatTemplate.Render([ChatTurn.User("What is diffusion?")], enableThinking: false);
var tokenizer = Tokenizer.FromFile(tokenizerPath);                  // the checkpoint's tokenizer.json (clover-bpe-chat)
int[] promptIds = [.. tokenizer.Encode(text, addSpecialTokens: false).Ids];
// ... run promptIds on a block-diffusion backend; stop at CloverCheckpoint.Profile.EosTokenIds ...
```

The checkpoint's `tokenizer.json` loads with the `Starlings.Tokenizers` version this repository pins: it uses only components that version implements. The named profile `clover-bpe-chat`, with its conformance evidence, comes from starlinghq/tokenizers#20. The pin is raised once that is released.

## What is checked

| Check | Result | Where |
| --- | --- | --- |
| Template text against transformers 5.19.0 `apply_chat_template` | 30/30 equal | `CloverChatTemplateTests` (CI) |
| Prompt ids of the owned tokenizer against the transformers tokenizer (`Qwen2Tokenizer`) | 30/30 equal | `CloverChatTemplateTests` (needs the tokenizer asset, `tools/fetch-tokenizer-assets.sh`) |
| Generation | not possible here | no backend |

Evidence: [docs/evidence/clover-onboarding-2026-10-28](https://github.com/starlinghq/starling/blob/main/docs/evidence/clover-onboarding-2026-10-28/README.md).

## Capabilities

`CloverCapabilities.All` lists every capability with an explicit status. In short:

- **Experimental:** the text template with its three switches, and the checkpoint lock.
- **Unsupported (rejected):** generation of any kind, including the upstream LowConfidence block decoding and self-speculative decoding. Also unsupported: sampling and the card's repetition and length controls, logprobs, guided decoding, tools and tool calls (the template's tool sections), and image input (the checkpoint is text only).
- **Untested:** AOT.

## License

Clover-35B-A3B-Preview by DiffuSpace, derived from Qwen3.6-35B-A3B (Copyright 2026 Alibaba Cloud), Apache-2.0 (`LICENSE` in the Hub repository at the pinned commit). `CloverChatTemplate` is a C# port of the checkpoint's `chat_template.jinja`, Apache-2.0 ([THIRD-PARTY-NOTICES](https://github.com/starlinghq/starling/blob/main/ai-dotnet/THIRD-PARTY-NOTICES.md)). The package does not redistribute the weights. Its own code is MIT OR Apache-2.0.
