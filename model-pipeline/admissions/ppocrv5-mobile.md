# Admission: PP-OCRv5 mobile (PaddlePaddle)

Model request #223, design #224 (route PR #227, library and evidence PR #228). Plan section 7, pipeline stage 1. The
licenses were read from the model card front matter at the pinned revisions; the pinned files are the four of
`PaddleOcrModelAssets` (size and SHA-256), the same as the candidate contract `candidates/ppocrv5-mobile/profile.json`.

| Field | Value |
| --- | --- |
| Candidate | `PaddlePaddle/PP-OCRv5_mobile_det_onnx` (DB text detector) and `PaddlePaddle/PP-OCRv5_mobile_rec_onnx` (CTC line recognizer), PaddlePaddle's official ONNX exports of `PaddlePaddle/PP-OCRv5_mobile_det` and `PaddlePaddle/PP-OCRv5_mobile_rec` |
| Code revision | No upstream code runs in the package. Its pre- and post-processing restates PaddleX at `ffb64904d23708863ff5b8da312a5cbd52a7f462` (tag v3.7.2), with OpenCV 4.10.0, Clipper 6.4.2 (pyclipper 1.4.0) and NumPy 2.3.5 behaviour (`ai-dotnet/PROVENANCE.md`) |
| Weight revision | Detector `e6f4fa85f00e168c862bc462aebca69eef9b3d3d`; recognizer `ed152b8b495f84de93cda5709d768548a9127622`. The route's reference also pins the Paddle inference models `PaddlePaddle/PP-OCRv5_mobile_det` at `0d63e78e2b680928f6b1747d76a08db6e645efb7` and `PaddlePaddle/PP-OCRv5_mobile_rec` at `682f20538d8c086cb2128e5cfac775e6c4904e85` |
| Requested by | #223: read the text in screenshots, scanned documents and video frames for search and RAG, next to the EmbeddingGemma 2 image and video embeddings |
| Proposed maturity at first release | Experimental |
| Decision | Admitted at maturity Experimental: `ppocrv5-mobile/onnx-fp32-cpu` (registry manifest `model-pipeline/registry/ppocrv5-mobile/onnx-fp32-cpu/manifest.yaml`), to be published as `Starlings.Models.PaddleOcr` from the next ai-dotnet release |
| Decided by and date | Mark Walls, 2026-11-23 |

## Rights (hard blockers)

| Item | License or terms | Redistribution |
| --- | --- | --- |
| Code | Starling's own (MIT OR Apache-2.0). Restated behaviour of PaddleX and OpenCV (Apache-2.0), Clipper (BSL-1.0) and NumPy (BSD-3-Clause), recorded in `ai-dotnet/THIRD-PARTY-NOTICES.md`, which the package packs | Yes |
| Weights | Apache-2.0 (`license: apache-2.0` in the model card front matter of both export repositories at the pinned revisions) | Yes; Starling does not redistribute them: the package downloads them from the PaddlePaddle repositories (`PaddleOcrModelAssets.DownloadAsync`) |
| Tokenizer assets | None. The recognizer's character dictionary (18,383 entries) is in its `inference.yml`, under the weights' Apache-2.0 | As the weights |
| Evaluation data | `reference/corpus/ocr-corpus-v1.jsonl`: ten images Starling renders (`reference/ppocr/make_ocr_corpus.py`), MIT OR Apache-2.0, with Noto fonts under SIL OFL 1.1 | n/a |

## Score

Recorded with the decision; no score was taken when the request was opened. Rows marked "not scored" have no recorded
basis and are left open rather than guessed.

| Criterion | Weight | Score (0 to 5) | Notes |
| --- | --- | --- | --- |
| User demand | 25% | not scored | One model request (#223); no measured demand |
| Backend feasibility | 20% | 5 | Two small fp32 graphs (about 21 MB) on ONNX Runtime CPU; outputs equal Python ONNX Runtime's exactly (#228) |
| Reusable architecture coverage | 20% | 4 | First OCR route: image to text lines; DB detection and CTC decoding are shared by the PP-OCRv5 server models and other PP-OCR recognizers |
| Access and licensing clarity | 15% | 5 | Apache-2.0 weights from the vendor, no gating |
| Testability | 10% | 4 | PaddleOCR on the Paddle inference engine as the reference, step by step; rendered corpus with ground truth |
| Maintenance burden (5 = low) | 10% | not scored | Owned ports of OpenCV and Clipper geometry (bit for bit on 378 seeded cases) are code Starling maintains; no measured burden yet |
| **Weighted total** | | not computed | Two rows not scored |

## Execution route

ONNX Runtime 1.30.0, CPU provider, fp32, on the official exports as they are (no Starling export or hosting). The
detector's probability map and the recognizer's per-step class probabilities are the graph outputs. Everything
around the graphs is owned C#: detector resize and normalization, DB post-processing (contours, min-area rectangles,
box score, Clipper round unclip), perspective crops, recognizer resize and greedy CTC decoding. No OpenCV and no native
dependency besides ONNX Runtime. Onboarding route `ppocr-onnx-cpu-v1` (`model-pipeline/ONBOARDING.md`, "PP-OCRv5
route"). The pipeline cannot run text-line orientation classification, document orientation or unwarping (separate
PP-OCR modules, not implemented).

## Tokenizer profile

None. The recognizer decodes CTC labels through a dictionary, not a tokenizer: blank, the 18,383 entries of the pinned
`inference.yml`, then the space character. `PaddleOcrConfiguration` checks the entry count, uniqueness and SHA-256
(`17665d27…`); the manifest records them as `processing.dictionary`. `Starlings.Tokenizers` is not a dependency.

## Feature contract

| Upstream feature | In scope | Evidence source |
| --- | --- | --- |
| Text detection (DB, quadrilateral boxes) | Yes, experimental | #228: detector input byte for byte, probability maps equal Python ONNX Runtime, boxes IoU 1 against the reference |
| Line recognition (CTC, greedy) | Yes, experimental | #228: crops byte for byte, probabilities equal Python ONNX Runtime, line text equal to the reference |
| English, Simplified Chinese | Yes, experimental | #228 (corpus images) |
| Traditional Chinese, Japanese | In the dictionary; untested | none |
| Rotated, perspective-skewed and sideways lines | Yes, experimental | #228 (rotated, perspective and vertical images) |
| Text-line orientation classification, document orientation, unwarping | No | n/a |
| Polygon boxes, word boxes, batched recognition | No | n/a |
| PP-OCRv5 server models | No (a later profile on the same route) | n/a |
| CUDA | Untested | none |

## Hardware and compute estimate

About 21 MB of fp32 graphs. The onboarding run (10 images, 36 reference lines, all seven stages) runs on two threads
of a Linux x64 CPU; the Paddle reference environment is Python 3.12 with paddlepaddle 3.3.1 CPU (oneDNN off).

## Open risks

- References are proposed, nothing is adopted into `protected/`, and no gate has run. The evidence is the development
  run in #228 (487 checks passed); no evidence directory is committed.
- The corpus is ten rendered images; no real scan or photograph is covered.
- Upside-down lines are read as they stand (no orientation classifier).
- The geometry ports (OpenCV resize, warp, contours, hull, rotating calipers, fill; Clipper offset) are verified bit
  for bit on seeded cases and the corpus, not on every input; a divergence would show as a box or crop difference.
- The registry manifest records the recognizer export through `origin.base` (files under `base/`), because the
  manifest schema has one weights source; it is not a base model.
