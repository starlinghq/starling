# Proposed: EmbeddingGemma 2 task-prompt rows (capability `prompts_task`)

Status: **proposed** (2026-10-14). Not accepted and not under `model-pipeline/protected/`. If the maintainer accepts it, the intended destination is `model-pipeline/protected/embeddinggemma-2/task-prompts/`, byte-identical. `CpuParityTests.TaskPromptParityWithTheOwnedTokenizer` and `CudaParityTests.TaskPromptParityWithTheOwnedTokenizer` read the protected copy when it exists and this proposal otherwise.

## Rows (`task-prompt-embeddings.jsonl`)

80 rows: four texts x the 20 prompt names of `config_sentence_transformers.json` at `914f7f89142e33e77833254d9c9b90c3cef7303b` (eight distinct prompt texts). The layout is that of the protected `embeddings.jsonl` (`id`, `prompt`, `prompt_text`, `text`, `input_ids`, `embedding`, `norm`), so the existing text-path check reads it unchanged.

| Text id | Source | Tokens with the `query` prompt |
| --- | --- | ---: |
| `alice-opening-clause` | Alice's Adventures in Wonderland, Project Gutenberg eBook #11 (public domain), first paragraph up to the colon | 31 |
| `alice-rabbit-paragraph` | Same book, the paragraph starting "There was nothing so _very_ remarkable" | 180 |
| `constitution-preamble` | The United States Constitution, Project Gutenberg eBook #5 (public domain), the preamble | 73 |
| `fixture-chinese` | The `chinese` case of the accepted text fixture | 17 |

The Gutenberg files are the ones pinned by SHA-256 in `../long-input/texts.json`.

## Reference source

`model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_task_prompts.py`: sentence-transformers 6.1.0 `encode(text, prompt_name=<name>)`, PyTorch 2.14.1 fp32 CPU, transformers 5.19.0 (`requirements-video.txt`, the environment of the accepted fixtures), 6 threads. `environment.json` records packages, asset digests and checks:

- `input_ids` are sentence-transformers' own preprocessing of the prompted text and equal `tokenizer.json` on `prompt + text` for all 80 rows.
- Names that share a prompt text give identical embeddings (max abs 0).
- A padded batch of the 20 prompted inputs vs the single rows: max abs 2.0e-7.

## Result at the time of the proposal

C# (`EmbeddingGemma2TextEmbedder`, owned tokenizer, ORT 1.30.0 CPU), singly and in padded batches per prompt: ids exact on all 80 rows; embedding max abs 3.6e-7, min cosine 0.999999999997, against the protected `onnx-fp32-cpu` bound 5e-6 (`docs/evidence/capability-closure-2026-10-14/`).

## Decision for the reviewer

Accept the rows as development data for `prompts_task`, with the protected text bounds of `onnx-fp32-cpu` and `onnx-fp32-cuda`.
