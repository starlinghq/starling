# Third-party notices: model-pipeline

The release evaluation sets under `protected/*/release/` and `protected/release-shared/` contain excerpts of the public datasets below, selected by the pinned scripts in `reference/stable/` (each set's `SOURCE.md` names the exact revision, digest and selection). They are evaluation data, not part of any shipped package. Each keeps its own license; where that differs from this repository's MIT OR Apache-2.0, the directory says so in its `LICENSE` file.

| Dataset | Rights holder | License | Used in |
| --- | --- | --- | --- |
| MASSIVE 1.1 | Amazon.com, Inc. | CC BY 4.0 | tokenizer holdouts and text release sets (e5, EmbeddingGemma 2, GLiNER2, Qwen) |
| HumanEval | OpenAI | MIT | tokenizer holdouts |
| OpenAssistant Conversations (oasst1) | LAION and contributors | Apache-2.0 | tokenizer holdouts; Qwen release prompts |
| MIRACL (via `mteb/MIRACLRetrievalHardNegatives`) | Zhang et al.; Wikipedia contributors | topics and judgments Apache-2.0; passages CC BY-SA 4.0, so the subset is **CC BY-SA 4.0** | `protected/release-shared/retrieval/miracl-dev-hn-8lang-v1/` |
| WNUT-17 Emerging and Rare Entities | Derczynski et al. | CC BY 4.0 | `protected/gliner2-base-v1/release/ner/wnut17-test-v1/`, GLiNER2 extraction release set |
| Few-NERD | Ding et al. | **CC BY-SA 4.0** | `protected/gliner2-base-v1/release/ner/fewnerd-test-v1/` (alternative set, not budgeted) |
| GSM8K | OpenAI | MIT | `protected/qwen3.5-0.8b/release/task-quality/gsm8k-test-v1/` |
| Wikimedia Commons media | per file (public domain, CC0, CC BY 2.0, CC BY 4.0) | per file, verified through the Commons API | EmbeddingGemma 2 image and video release sets: referenced by URL and SHA-256 only, not redistributed; attribution per file in `licenses.json` and `SOURCE.md` |
