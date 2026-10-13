# Qwen template attribution

The C# text renderers adapt the `chat_template` strings from `tokenizer_config.json` in Qwen/Qwen3-0.6B at c1899de289a04d12100db370d81485cdf75e47ca and Qwen/Qwen2.5-0.5B-Instruct at 7ae557604adf67be50417f59c2c2f167def9a775.

Copyright 2024 Alibaba Cloud. Licensed under Apache License 2.0. The complete upstream license is included in LICENSE.Qwen.txt.

Source template SHA-256: Qwen3 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`; Qwen2.5 `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` (UTF-8 bytes of the JSON string without an added newline).

Modifications: C# rendering restricted to text system/user/assistant messages, with Qwen3 thinking disabled. Tools and multimodal content are rejected. Model weights are not distributed in this package.
