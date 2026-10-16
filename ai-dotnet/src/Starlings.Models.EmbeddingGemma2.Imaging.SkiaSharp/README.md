# Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp

Decodes image files and streams with [SkiaSharp](https://github.com/mono/SkiaSharp) into `EmbeddingGemma2Image` for `Starlings.Models.EmbeddingGemma2`, reproducing what the upstream processor receives from PIL (`Image.open(...).convert("RGB")`). The model package itself takes decoded RGB and has no native dependency; this adapter is optional.

```csharp
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;

EmbeddingGemma2Image image = EmbeddingGemma2ImageDecoder.DecodeFile("cat.jpg");
using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(EmbeddingGemma2ModelAssets.OpenLocalImageAssets());
float[] embedding = embedder.Embed(image);
```

## Maturity

<!-- catalog:maturity profiles=embeddinggemma-2/image-* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/image-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/image-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Rules

| Input | Behaviour | Status |
| --- | --- | --- |
| PNG (RGB, grayscale, RGBA, gray+alpha, palette, palette with transparency, 1-bit), JPEG (baseline, progressive, 4:2:0, 4:4:4, CMYK) | Decoded RGB equal to PIL 12.3.0's on the 28 supported fixture images | experimental |
| GIF (palette, grayscale; first frame of an animation), BMP (24-bit, 32-bit with alpha, 8-bit palette and grayscale, 1-bit), WebP (lossless, lossy, with and without alpha) | Decoded RGB equal to PIL 12.3.0's on 12 files written by Pillow from fixture images (proposed `model-pipeline/proposed/embeddinggemma-2/image-extra/`) | experimental |
| GIF with transparent pixels | Rejected: `EmbeddingGemma2UnsupportedImageException` with `Reason = GifTransparency`. PIL keeps the palette colour of transparent pixels; SkiaSharp zeroes them, so the embedding would differ | unsupported |
| Alpha | Dropped, never composited (a transparent pixel keeps its stored colour), as upstream | as upstream |
| Colour profiles | Ignored (no colour management), as PIL | as upstream |
| EXIF orientation | Ignored by default, as upstream. `new EmbeddingGemma2ImageDecoderOptions { ApplyExifOrientation = true }` rotates or mirrors the image upright like Pillow's `exif_transpose`: **a deliberate deviation from upstream** (the embedding then differs from sentence-transformers') | experimental (opt-in) |
| 16 bits per channel (16-bit PNG and other high-bit-depth sources) | Rejected: `EmbeddingGemma2UnsupportedImageException` with `Reason = HighBitDepth`. Upstream saturates such images to white; any decoder that scales them would give another embedding | unsupported |
| More than 50 megapixels (`MaxPixels`) | Rejected from the header before decoding: `EmbeddingGemma2ImageTooLargeException` | unsupported above the limit |
| Truncated or corrupt data | Rejected (`Reason = Undecodable`), as PIL rejects it | unsupported |
| Other formats (ICO, WBMP, DNG, HEIF, AVIF, ...) | Rejected (`Reason = UnsupportedFormat`) | unsupported |

Linux x64 is tested. The adapter brings `SkiaSharp.NativeAssets.Linux.NoDependencies` (no fontconfig needed); on Windows and macOS, `SkiaSharp` brings its own natives (untested here).

## Licenses

The package is MIT OR Apache-2.0. It depends on SkiaSharp 4.153.1 (MIT, Microsoft) and its native library, which bundles third-party code under the notices in the natives package's `THIRD-PARTY-NOTICES.txt`, among them a GIF decoder tri-licensed MPL 1.1 / GPL 2.0 / LGPL 2.1 (used under the MPL 1.1, unmodified) and the Adobe DNG SDK license ("This product includes DNG technology under license by Adobe Systems Incorporated."). Applications that ship this adapter ship those natives and should carry that file. Summary: `THIRD-PARTY-NOTICES.md` in this package. ImageSharp is not used (Split License, not OSI-approved).

Deployments of the model must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) (model card).
