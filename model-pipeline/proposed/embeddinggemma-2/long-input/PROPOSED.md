# Proposed: EmbeddingGemma 2 long-input fixtures (2,049 to 8,192 tokens)

Status: **proposed** (R7 W3, 2026-10-13). Not accepted fixtures; nothing reads them as a bound. The admission record and the feature contract list inputs between 2,049 and 8,192 tokens as untested; the manifests' `input_2050_to_8192_tokens` row (`untested`) moves only after the maintainer accepts these rows and a parity run with accepted bounds passes. This change does not move it.

## Rows (`long-embeddings.jsonl`)

17 rows, one input each (`csharp-compare.md` lists them with token counts):

| Kind | Rows | Tokens |
| --- | --- | --- |
| Text, no prompt / `query` / `document` | Alice passages; the Constitution (whole, `document`); the M0 tokenizer corpus cycled | 2,600; 4,096; 5,950; 6,144; 8,191; 8,192 (x3) |
| Text with six or eight images | Alice with the photos and the two synthetic images between paragraphs | 6,000; 8,192 (`document`) |
| Text with video at the 32-frame limit | 40 s clip thinned to 32 frames; 33 s clip (32 frames, `query`); two videos and an image (`document`) | 7,000; 8,192; 8,192 |
| Just over the limit (`expect: reject`) | text; the 8,192-token text with the `document` prompt (prompt tokens count); six images; 32-frame video | 8,193; 8,198; 8,193; 8,193 |

`expect: reject` rows are just over 8,192 tokens (the card's context window; the C# package rejects them with `EmbeddingGemma2InputTooLongException`, upstream embeds them without truncating). Their upstream embedding is kept for information only.

## Sources and reference environment

- Text: two public-domain Project Gutenberg plain-text files, pinned by SHA-256 (`texts.json`: Alice's Adventures in Wonderland, eBook #11, `01b38ea4...`; the United States Constitution, eBook #5, `d177168e...`; header, footer and license text removed) and the repository's M0 tokenizer corpus (`tokenizer-corpus-v1.jsonl`, cycled). Each row carries the exact text used. Each row's text is the longest prefix of its passage whose rendered input (prompt, chat template, media placeholders expanded, as sentence-transformers' own preprocessing counts) has exactly the target token count.
- Media: the accepted image set (`make_eg2_images.py`) and video set (`make_eg2_videos.py`), each file checked against its protected SHA-256. Videos at the 32-frame limit: `h264-long-320x240-30fps-40s` (40 s, thinned to 32 frames), `h264-33s-160x120-30fps`; also `h264-ntsc-640x480-29.97fps-12s` and the photos.
- Reference: `model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_long.py` at this commit, PyTorch fp32 CPU, sentence-transformers 6.1.0, transformers 5.19.0, torch 2.14.1, torchcodec 0.17.0 on the pinned LGPL FFmpeg 8.1 (`requirements-video.txt`), checkpoint `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b` (`long-environment.json`). Run on 2026-10-13 from this branch's sources (`long-environment.json` records the commit, marked dirty because the runner was not yet committed; it is unchanged in the commit). Stability (`long-reference.json`): padded batch of two vs single rows 6.0e-8 (text) and 4.5e-8 (images + video); eager vs sdpa attention 4.5e-8 (text, 8,192 tokens) and 8.2e-8 (video row); a repeated run identical; `tokenizer.json` ids equal the rendered ids of every text row; no NaN/Inf. Upstream neither truncates nor rejects above 8,192 tokens.
- Tensors (token embeddings, soft tokens) are outside Git (`--tensors`); `tensors.json` pins them.

## C# comparison (`csharp-compare.md`, `csharp-compare.json`)

`Starlings.Fidelity.Tool embeddinggemma2-long` (CPU, text-only rows on the released text graph, media rows on the vision and multimodal graphs, FFmpeg 8 adapter, SkiaSharp images) compared by `compare_embeddinggemma2_long_csharp.py`:

| Check | Result |
| --- | --- |
| Rendered ids (owned tokenizer, media placeholders expanded), 13 embedded rows | all equal |
| Rows over 8,192 tokens (4) | all rejected with `EmbeddingGemma2InputTooLongException`, token counts equal to the reference's |
| Text rows (8): embedding max abs, min cosine | 1.0e-7 (`txt-corpus-cycle-8192-none`), 0.9999999999996 |
| Media rows (5): embedding max abs, min cosine | 6.8e-7 (`mm-alice-video32-8192-query`), 0.999999999987 |
| Padded batches of two vs C# single rows (text; images + video) | 0 and 0 |
| NaN/Inf | none |

For orientation only, the protected bounds of the short-input profiles are embedding max abs 1e-5 (and min cosine 0.9999995 for media). No bound applies to these rows until the maintainer sets one.

## What the maintainer would do

Accept (or change) the rows and move them under `model-pipeline/protected/embeddinggemma-2/` with tolerances of the maintainer's choosing; then a parity run against them can move `input_2050_to_8192_tokens` from `untested`.
