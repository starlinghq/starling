# Proposed: EmbeddingGemma 2 image budgets and GIF, BMP and WebP decoding

Status: **proposed** (2026-10-14). Not accepted and not under `model-pipeline/protected/`. If the maintainer accepts it, the intended destination is `model-pipeline/protected/embeddinggemma-2/image-extra/`, byte-identical. The tests named below read the protected copy when it exists and this proposal otherwise.

Both parts come from `model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_image_extra.py`, run on the accepted image set (`make_eg2_images.py`, every file checked against `images.json`). The environment is `requirements-video.txt`: sentence-transformers 6.1.0, transformers 5.19.0, torch 2.14.1 CPU and Pillow 12.3.0. Package versions and asset digests are in `image-extra-environment.json`.

## `image-budget-embeddings.jsonl`: capability `image_budget_other`

24 rows: six fixture images (`photo-astronaut`, `photo-coffee`, `photo-camera`, `aspect-wide-2000x40`, `size-tiny-8x6`, `size-huge-4000x3000`), each embedded alone with no prompt, at `max_soft_tokens` 70, 140, 560 and 1,120. The budget is set on the processor's `Gemma4ImageProcessor`, which accepts exactly these values; the accepted fixtures cover 280. The rows use the layout of the protected `image-embeddings.jsonl`, plus `max_soft_tokens`. The runner restates the processor at each budget and checks that it equals the processor bit for bit.

The test is `ImageParityTests.OtherBudgetsMatchTheReference`, with the same test in `Starlings.Models.EmbeddingGemma2.Tests.Cuda`. It checks the following against the protected `image-fp32-cpu` bounds (embedding max abs 1e-5, min cosine 0.9999995):

- Decoded RGB, resized pixels, pixel values, positions, soft-token counts and rendered ids are exact on all 24 rows.
- On CPU, the embedding max abs is 9.2e-6 (`photo-camera@560`) and the min cosine is 0.9999999976.

The worst row sits close to the bound. The other budgets stay at or below 1.9e-6.

## `image-formats.json`: capability `image_decode_gif_bmp_webp`

There are 13 files. Pillow writes them from fixture images into a directory outside Git, and each keeps the license of its source image:

- GIF: palette, transparent, grayscale, and a two-frame animation, of which frame 0 is used.
- BMP: 24-bit, 32-bit with alpha, 8-bit palette, 8-bit grayscale, and 1-bit.
- WebP: lossless, lossy, lossy with alpha, and lossless with alpha.

The reference digest is that of `np.asarray(Image.open(f).convert("RGB"))`, which is what upstream's processor receives.

The test is `DecoderTests.GifBmpAndWebpDecodeLikePil` in `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp.Tests`, which needs `STARLING_EMBEDDINGGEMMA2_IMAGE_FORMATS`. SkiaSharp's decoded RGB equals PIL's on 12 of the 13 files.

The exception is `gif-transparent`. PIL keeps the palette colour of transparent pixels (77,042 pixels here), while SkiaSharp zeroes them. The decoder now rejects a GIF with transparent pixels with `EmbeddingGemma2ImageRejection.GifTransparency`, and the CI test `GifWithTransparencyIsRejectedWithATypedException` checks this.

## Decision for the reviewer

Accept the rows as development data for `image_budget_other` and `image_decode_gif_bmp_webp`, judged with the protected image bounds and `decoder_skiasharp.decoded_rgb_vs_pil: exact`.
