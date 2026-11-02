# Qwen3 Embedding series attribution

`Qwen3EmbeddingProfile` reproduces, verbatim, the query prompt of `config_sentence_transformers.json` in Qwen/Qwen3-Embedding-0.6B at 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3, and the reranker template text (system prompt, user-turn layout, assistant opening with an empty thinking block, and the default instruction) of `chat_template.jinja` and the model card of Qwen/Qwen3-Reranker-0.6B at e61197ed45024b0ed8a2d74b80b4d909f1255473.

Copyright 2025 Alibaba Cloud. Licensed under Apache License 2.0. The complete upstream license is included in LICENSE.Qwen.txt.

Modifications: the template is written as C# string constants and composed in code (prefix, content, suffix) instead of being rendered from Jinja. Model weights, tokenizer files and ONNX graphs are not distributed in this package.
