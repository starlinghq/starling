# Experimental Qwen text chat

The admitted profiles are `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` and `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`. Both run CPU fp32 with ONNX Runtime GenAI 0.17.1 and the owned tokenizer. Add `Microsoft.ML.OnnxRuntimeGenAI` to the application for the native runtime.

```csharp
using Starlings.Models.QwenChat;
using var model = QwenChatModel.Load(QwenChatProfile.Qwen3, bundleDirectory);
var result = model.Generate([new(QwenChatRole.User, "Explain gravity briefly.")]);
Console.WriteLine(result.Text);
```

Build the hash-verified original snapshot with `model-pipeline/reference/qwen-chat/build_bundle.py`. The loader validates the admitted original tokenizer and template, architecture, vocabulary, EOS ids and proposed graph lock. It never downloads weights. Graph hashes are observed candidate evidence, not a published independently adopted asset lock.

These are experimental text-only profiles. Qwen3 explicitly uses `enable_thinking=false`, which differs from its upstream default. Qwen2.5 inserts the upstream default system instruction when none is supplied. Tools, tool messages, multimodal input, sampling and Qwen3 thinking mode are unsupported. Each request has its own disposable generator/cache; continuing chat requires resending the whole conversation. CPU native threads are bounded to two. Cancellation is checked before prefill and between decode steps; a running native operation is not interruptible.

Both native paths overlay `model.pad_token_id=151935`, the last embedding row, after verifying neither original vocabulary nor added tokens maps to it (maximum tokenizer ids: Qwen3 151668; Qwen2.5 151664). GenAI otherwise treats literal `<|endoftext|>` id 151643 in prompt text as padding. Original configuration and graph files are never modified by this runtime overlay.

Fresh reference and parity reports remain proposals. They do not award protected G1/G2 qualification, approve releases, or change support status.
