# Starlings.Models.PaddleOcr

OCR with PP-OCRv5 mobile on ONNX Runtime: the DB text detector finds text lines, each line is cropped upright and read
by the CTC line recognizer. The graphs are PaddlePaddle's official ONNX exports
(`PaddlePaddle/PP-OCRv5_mobile_det_onnx`, `PaddlePaddle/PP-OCRv5_mobile_rec_onnx`, Apache-2.0); the recognizer's
dictionary (18,383 entries: Simplified and Traditional Chinese, English, Japanese, symbols) comes from its pinned
`inference.yml`. All post-processing is owned C# (no OpenCV, no native dependency besides ONNX Runtime). This package
ships the asset locks, not the weights; acquire them explicitly. The application adds the ONNX Runtime native package.

This library is **experimental** and not yet qualified. Its evidence is proposed: the model-pipeline route
`ppocr-onnx-cpu-v1` (`model-pipeline/candidates/ppocrv5-mobile/`) compares it against PaddleOCR 3.7.0 on the Paddle
inference engine on ten rendered images. No tolerances or reference fixtures are adopted, no registry manifest exists
yet and no release gate has run.

## Install

Not published to nuget.org: the package is not in the 0.4.0 publishable set. Build it from this repository (a
`ProjectReference` to `ai-dotnet/src/Starlings.Models.PaddleOcr/Starlings.Models.PaddleOcr.csproj`) and add
`Microsoft.ML.OnnxRuntime` (CPU natives) to the application.

## Quick start

```csharp
using Starlings.Models.PaddleOcr;

// One-time setup: download the two graphs and their inference.yml files (about 21 MB) into the asset store.
var files = await PaddleOcrModelAssets.DownloadAsync();

// Load offline at startup; both configurations are checked against the contract first.
using var ocr = PaddleOcrEngine.Create(files);

// Decoded pixels: RGB, BGR or RGBA.
var image = PaddleOcrImage.FromRgb24(width, height, rgbPixels);
foreach (TextLine line in ocr.Recognize(image))
{
    Console.WriteLine($"{line.Text} ({line.Confidence:F3}) at {line.Box.TopLeft}");
}
```

The package decodes no image file. To read PNG, JPEG, GIF, BMP or WebP files, decode them first, for example with
`Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp` (decoded RGB equal to PIL's, which for 8-bit PNG equals OpenCV's
`imread`, the reference's reader):

```csharp
var decoded = EmbeddingGemma2ImageDecoder.DecodeFile("scan.png");
var lines = ocr.Recognize(PaddleOcrImage.FromRgb24(decoded.Width, decoded.Height, decoded.Pixels.Span));
```

## Pipeline

The steps and constants of paddlex 3.7.2 for these two models, all fixed by the contract:

| Step | Behaviour |
| --- | --- |
| Detector input | BGR; tiny images zero-padded to 32 x 32; longer side limited to 960 (4,000 cap), sides rounded half to even to multiples of 32; OpenCV `INTER_LINEAR` resize (fixed-point uint8, restated bit for bit); `(x / 255 - mean) / std` with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) |
| DB post-processing | probability > 0.3; contours (Suzuki-Abe, OpenCV's `findContours` order); min-area rectangles; box score = mean probability inside the box (fast mode) ≥ 0.6; unclip by area x 1.5 / perimeter (Clipper round offset); min-area rectangle again; boxes shorter than 3 (5 after unclip) dropped; at most 1,000 candidates |
| Order | top to bottom by the first corner, then left to right within 10 pixels of height |
| Crop | min-area rectangle of the box, perspective warp (OpenCV `INTER_CUBIC`, replicated border, restated bit for bit); crops at least 1.5 times taller than wide are turned 90 degrees counter-clockwise |
| Recognizer input | height 48, width from the aspect ratio (at least 320, at most 3,200), `(x / 255 - 0.5) / 0.5`, zero-padded; one crop per run |
| Decoding | greedy CTC: per-step argmax, repeats collapsed, blank dropped; label map blank + dictionary + space; confidence = mean of the kept maxima |

## Evidence (proposed)

On the onboarding corpus (`model-pipeline/reference/corpus/ocr-corpus-v1.jsonl`: documents, an invoice, Simplified
Chinese, small print, a synthetic application window, rotated, perspective-warped, sideways, noisy and text-free
images), with the pinned models on Linux x64 CPU:

- the detector input and every recognizer crop are byte-for-byte the reference's tensors;
- the probability maps and recognizer probabilities equal Python ONNX Runtime's exactly on the same inputs;
- the owned DB post-processing on the reference probability maps gives the reference boxes (IoU 1) and box scores;
- end to end from the image, every box matches the reference (IoU 1) and every line's text equals it.

The geometry ports are also checked against OpenCV 4.10 and pyclipper 1.4.0 on 378 seeded cases in this package's
tests (`TestData/opencv-geometry-v1.json`).

## Supported (experimental)

- Text detection and line recognition on Linux x64 CPU with the pinned mobile detector and recognizer.
- English and Simplified Chinese are covered by the evidence; Traditional Chinese and Japanese are in the dictionary
  but untested.
- Rotated and perspective-skewed lines (the crop warps them upright); sideways lines through the tall-crop rule.
- Images up to 32,767 pixels per side and 64 Mpx in total; larger ones are refused with
  `PaddleOcrImageTooLargeException` (upstream stores box corners as int16, so they would wrap there).
- Cancellation between lines; a cancelled token runs nothing.
- The asset locks and explicit acquisition (`PaddleOcrModelAssets`).

## Unsupported

Text-line orientation classification (upside-down lines are read as they stand), document orientation and unwarping,
polygon (curved) boxes, word boxes, batched recognition, the PP-OCRv5 server models, image file decoding,
`Microsoft.Extensions.AI` integration, CUDA qualification, training and quantized graphs. A single engine is not proven
safe for concurrent calls.

## Capabilities

See `PaddleOcrCapabilities` (statuses are the shared `Starlings.Models.CapabilityStatus`): every capability has an
explicit status, and none is `Supported`. The proposed profile
is `PaddleOcrCapabilities.CpuProfile` (`ppocrv5-mobile/onnx-fp32-cpu`).

## Third-party code

The OpenCV, Clipper, PaddleX and NumPy behaviour this package restates is listed in `ai-dotnet/PROVENANCE.md` and
`ai-dotnet/THIRD-PARTY-NOTICES.md` (OpenCV and PaddleX Apache-2.0, Clipper Boost Software License 1.0, NumPy
BSD-3-Clause); the notices file ships in the package.
