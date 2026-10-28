# Starlings.Models.Lfm3

Text generation with `LiquidAI/LFM3-1.7B` on ONNX Runtime, through Liquid's own ONNX export (`LiquidAI/LFM3-1.7B-ONNX` at `c680759afee1ce8310c90d6c621f4558200dee77`). It includes the LFM3 chat template for text turns and a managed greedy decode loop over the export's step graph, with its convolution and attention caches. Prompts are tokenized by the owned tokenizer (`Starlings.Tokenizers`, profile `lfm3-bpe-chat`). The package ships the asset lock, not the weights: acquire them explicitly. The application adds the ONNX Runtime native package.

This library is **experimental**, source only and not admitted: the admission decision is open (`model-pipeline/admissions/lfm3.md`, model request #217), and no registry profile or release gate exists. It needs `Starlings.Tokenizers` with the `lfm3-bpe-chat` profile (starlinghq/tokenizers#18); the released `0.3.2` rejects the LFM3 pre-tokenizer regex at load.

## What is checked

| Check | Result | Where |
| --- | --- | --- |
| Template text against transformers 5.21.0 `apply_chat_template` | 19/19 equal | `Lfm3ChatTemplateTests` (CI) |
| Prompt ids against the transformers tokenizer | 19/19 equal | `Lfm3ParityTests` (needs the snapshot) |
| Greedy ids and step-0 logits against the same fp32 graph on ONNX Runtime 1.30.0 in Python | 19/19 equal ids, logits max abs 0 | `Lfm3ParityTests`, [evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/lfm3-onboarding-2026-10-28/README.md) |
| Against the PyTorch checkpoint | not run | `model-pipeline/reference/lfm3/run_lfm3.py reference` |

## Quick start

```csharp
using Starlings.Generation;
using Starlings.Models.Lfm3;

// One-time setup: download the fp32 graph (7.2 GB) and the tokenizer into the asset store.
Lfm3ModelFiles files = await Lfm3ModelAssets.DownloadAsync(Lfm3Precision.Fp32);

// At startup: load offline.
using Lfm3ChatModel model = Lfm3ChatModel.Create(Lfm3ModelAssets.OpenLocal(Lfm3Precision.Fp32));
GenerationResult reply = model.Chat(
    [ChatTurn.System("Answer in one short sentence."), ChatTurn.User("What is a heat pump?")],
    new GenerationOptions { MaxNewTokens = 128 },
    enableThinking: false);
Console.WriteLine(reply.Text);
```

With `enableThinking: true` (the template's default) the model writes its reasoning first and closes it with `</think>`. The completion text keeps that reasoning: splitting it off is the caller's job.

## Capabilities

`Lfm3Capabilities.All` lists every capability with an explicit status. In short:

- **Experimental:** greedy generation on CPU (fp32), streaming (`GenerateStreaming`), stop sequences and extra stop ids, chunked prefill, cancellation between calls, the text template with `enableThinking` and `preserveThinking`, asset acquisition.
- **Untested:** the fp16 graph (`Lfm3Precision.Fp16`) and CUDA, the intended RTX 3090 profile. The export's card notes that the CUDA provider has no fp32 GroupQueryAttention kernel, so CUDA needs the fp16 graph. Also untested: concurrent requests, contexts beyond the evidence prompts, and AOT.
- **Unsupported (rejected):** sampling (the card's `temperature 0.1, top_k 50, repetition_penalty 1.1` included), logprobs, guided decoding, tools and tool calls, image, video and audio input, prefix reuse, batches, the quantized graphs.

`Lfm3Options.ContextLength` defaults to 32,768 tokens. The 11 attention layers' caches grow by 44 KB per token in fp32. The checkpoint's 262,144 positions are accepted but untested.

## License

The weights, tokenizer and chat template are Liquid AI's, under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd/LICENSE). That license grants commercial use only to a legal entity below 10 million USD annual revenue (its section 5). This package does not redistribute the weights; an application that downloads them accepts those terms. `Lfm3ChatTemplate.cs` is a C# port of the checkpoint's `chat_template.jinja` ([THIRD-PARTY-NOTICES](https://github.com/starlinghq/starling/blob/main/ai-dotnet/THIRD-PARTY-NOTICES.md)). The package's own code is MIT OR Apache-2.0.
