# Third-party notices: model-pipeline

The release evaluation sets under `protected/*/release/` and `protected/release-shared/` contain excerpts of the public datasets below, selected by pinned procedures (each set's `SOURCE.md` names the exact revision, digest and selection script; the `reference/stable/` scripts it names are not in this tree). They are evaluation data, not part of any shipped package. Each keeps its own license; where that differs from this repository's MIT OR Apache-2.0, the directory says so in its `LICENSE` file.

| Dataset | Rights holder | License | Used in |
| --- | --- | --- | --- |
| MASSIVE 1.1 | Amazon.com, Inc. | CC BY 4.0 | tokenizer holdouts and text release sets (e5, EmbeddingGemma 2, GLiNER2, Qwen) |
| HumanEval | OpenAI | MIT | tokenizer holdouts |
| OpenAssistant Conversations (oasst1) | LAION and contributors | Apache-2.0 | tokenizer holdouts; Qwen release prompts |
| MIRACL (via `mteb/MIRACLRetrievalHardNegatives`) | Zhang et al.; Wikipedia contributors | topics and judgments Apache-2.0; passages CC BY-SA 4.0, so the subset is **CC BY-SA 4.0** | `protected/release-shared/retrieval/miracl-dev-hn-8lang-v1/` |
| WNUT-17 Emerging and Rare Entities | Derczynski et al. | CC BY 4.0 | `protected/gliner2-base-v1/release/ner/wnut17-test-v1/`, GLiNER2 extraction release set |
| Few-NERD | Ding et al. | **CC BY-SA 4.0** | `protected/gliner2-base-v1/release/ner/fewnerd-test-v1/` (alternative set, not budgeted) |
| LibriSpeech ASR corpus (dev-clean, via `hf-internal-testing/librispeech_asr_dummy`) | Panayotov, Chen, Povey and Khudanpur; LibriVox volunteers (public-domain audiobooks) | CC BY 4.0 | `reference/corpus/asr-v1/` (four utterances as WAV) and their transcripts in `reference/corpus/asr-corpus-v1.jsonl`: Whisper route input, not a release set |
| GSM8K | OpenAI | MIT | `protected/qwen3.5-0.8b/release/task-quality/gsm8k-test-v1/` |
| Wikimedia Commons media | per file (public domain, CC0, CC BY 2.0, CC BY 4.0) | per file, verified through the Commons API | EmbeddingGemma 2 image and video release sets: referenced by URL and SHA-256 only, not redistributed; attribution per file in `licenses.json` and `SOURCE.md` |
| Wikimedia Commons media, G3 retrieval sets | per file (images: public domain, CC0, CC BY 2.0, 2.5, 3.0, 4.0; videos: public domain, CC BY 2.0, 3.0) | per file, verified through the Commons API | `protected/embeddinggemma-2/release/image-retrieval-v1/`, `video-retrieval-v1/`: referenced by URL and SHA-256 only, not redistributed; attribution per file in `dataset.json` and `licenses.json` |
| Wikimedia Commons file descriptions | contributors of each file's description page | **CC BY-SA 4.0** (public domain where copied from public-domain captions) | the queries of `image-retrieval-v1/` and `video-retrieval-v1/` (selected by `reference/embeddinggemma2/select_eg2_retrieval.py`); each directory's `LICENSE` says so |

## Converted model graphs

The reference scripts below convert upstream model weights into graphs that Starling hosts on Hugging Face. The graphs are not in this tree; each hosted repository carries the upstream license and the notice named here, which lists the modifications (Apache-2.0, section 4(b)).

| Upstream model (revision) | Rights holder | License | Script | Notice | Hosted at |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3-Embedding-0.6B` (`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`) | Qwen team, Alibaba Cloud | Apache-2.0 | `reference/qwen3-embedding/export_qwen3_onnx.py`, `release_qwen3_onnx.py` | `reference/qwen3-embedding/NOTICE-qwen3-embedding-onnx.txt` | `Starlinghq/qwen3-embedding-0.6b-onnx` (tag `r1`) |
| `Qwen/Qwen3-Reranker-0.6B` (`e61197ed45024b0ed8a2d74b80b4d909f1255473`) | Qwen team, Alibaba Cloud | Apache-2.0 | same | `reference/qwen3-embedding/NOTICE-qwen3-reranker-onnx.txt` | `Starlinghq/qwen3-reranker-0.6b-onnx` (tag `r1`) |
