# Starlings.Models.Phi4

The model definition of [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) at revision `cfbefacb99257ffa30c83adab238a50856ac3083`, without a runtime. Generation goes through the generic `GenAIChatModel` in `Starlings.Generation.OnnxGenAI`, which reads `Phi4Checkpoint.ChatDefinition`.

This library is **experimental** and source only. It was admitted at maturity Experimental on 2026-11-04 with no package, no registry manifest and no hosted bundle (`model-pipeline/admissions/phi4-mini.md`).

- `Phi4ChatTemplate`: the checkpoint's `chat_template` rendered in C# for text turns (system, user, assistant).
- `Phi4Checkpoint`: repository and revision; the special token ids; the decoder shape; `Profile` (a `ModelProfile`); `Lock` (the asset lock of the Hub files a bundle is built from, about 7.7 GB of bf16 weights); `BundleFiles` (the lock of the six built bundle files); `ChatDefinition` (the entry of the route contract table, embedded here, with the template, the bundle lock and the tokenizer check); `OpenTokenizer`, which verifies the pinned `tokenizer.json` and loads it with the owned tokenizer (`phi4-bpe-chat`, `Starlings.Tokenizers` 0.3.6).
- `Phi4Capabilities.All`: every capability with an explicit status.

```csharp
using Starlings.Generation;
using Starlings.Models.Phi4;

string text = Phi4ChatTemplate.Render([ChatTurn.System("Be brief."), ChatTurn.User("What is a river?")]);
var tokenizer = Phi4Checkpoint.OpenTokenizer(tokenizerPath);
int[] promptIds = [.. tokenizer.Encode(text, addSpecialTokens: false).Ids];
```

## Generation

No hosted bundle exists, and nothing downloads one. Build it from the pinned snapshot with `model-pipeline/reference/genai-chat/build_bundle.py` (the `bundle` stage of `model-pipeline/recipes/phi4-mini-cpu.yaml`). `GenAIChatModel.Load` verifies all six files against `Phi4Checkpoint.BundleFiles`, then loads with the pad overlay.

```csharp
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Phi4;

using GenAIChatModel model = GenAIChatModel.Load(Phi4Checkpoint.ChatDefinition, bundleDirectory);
GenerationResult result = model.Generate([ChatTurn.User("What is a river?")], new GenerationOptions { MaxNewTokens = 64 });
Console.WriteLine(result.Text);
```

The application references `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 for the native runtime.

Sizes: `model.onnx.data` is 15.3 GB. Anonymous memory peaked at 17.4 GiB in the evidence run's .NET stage, in 15-second samples, inside a 20 GiB limit. One or two native threads.

Prompt plus completion should stay below 4,096 tokens: past that the long rope factors start, and no evidence covers them. The loader does not refuse a longer request; the status is untested, not unsupported.

## Template notes

Text is inserted as given: a special token's text in a turn becomes that special token. The role tags have `rstrip` in `tokenizer.json`, so the whitespace after a tag does not reach the model. An assistant turn's reasoning is ignored, as the upstream template ignores it.

## Capabilities

- **Experimental:** the text template, the checkpoint lock, greedy CPU fp32 generation through `GenAIChatModel`, streaming, cancellation, and stop tokens and sequences.
- **Untested:** prompt plus completion at or past 4,096 tokens, where the long rope factors start; sampling, logprobs and guided decoding on this checkpoint; AOT.
- **Unsupported (rejected or absent):** tools (the template's tool section, tool calls and tool turns), thinking, vision, CUDA, fp16, `Microsoft.Extensions.AI`.

Evidence: `docs/evidence/phi4-mini-onboarding-2026-11-04/README.md`.

## License

Phi-4-mini-instruct, Copyright (c) Microsoft Corporation, MIT (`LICENSE` in the Hub repository at the pinned commit). `Phi4ChatTemplate` is a C# port of its `chat_template` (`ai-dotnet/THIRD-PARTY-NOTICES.md`).
