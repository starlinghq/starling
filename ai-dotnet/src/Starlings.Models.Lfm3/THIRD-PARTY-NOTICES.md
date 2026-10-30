# Third-party notices: Starlings.Models.Lfm3

## LFM3 (Liquid AI): LFM Open License v1.0

**Commercial use is licensed only below 10 million USD annual revenue.** Read [LICENSE.txt](LICENSE.txt) in this package before you use it.

- `Lfm3ChatTemplate.cs` is a C# port of the text sections of the `chat_template.jinja` of `LiquidAI/LFM3-1.7B` at commit `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd` (SHA-256 `f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818`), Copyright Liquid AI, Inc., licensed under the LFM Open License v1.0. Modified: rewritten in C# and restricted to text system, user and assistant turns. The file carries this notice in its header.
- The weights and tokenizer the package downloads on request (`Lfm3ModelAssets`), from `LiquidAI/LFM3-1.7B-ONNX` at `c680759afee1ce8310c90d6c621f4558200dee77`, are Liquid AI's under the same license. They are not in the package.
- The full license text is in `LICENSE.txt`, as published with the checkpoint (SHA-256 `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c`).

## Owned tokenizer

Tokenization uses `Starlings.Tokenizers` (MIT OR Apache-2.0), profile `lfm3-bpe-chat`, a package dependency with its own notices.
