Native AOT smoke check (ubuntu.24.04-x64, Starlings 0.3.0, commit a691d5c); tools/aot-smoke-ai.sh.

| Package | Starlings.* warnings | Other warnings | AOT publish | AOT run | Verdict |
| --- | --- | --- | --- | --- | --- |
| Starlings.Acquisition | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Runtime.Onnx | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.E5 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.E5.Extensions.AI | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.E5.MLNet | 0 | 8 | ok | ok | clean in Starlings.* (recorded AOT unsupported) |
| Starlings.Models.Gliner2 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2.Extensions.AI | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2.MLNet | 0 | 8 | ok | ok | clean in Starlings.* (recorded AOT unsupported) |
| Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2.Video.FFmpeg | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2.Video.FFmpeg9 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Generation.OnnxGenAI | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Qwen35 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Qwen35.Extensions.AI | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Kev | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.QwenChat | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Bert | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Whisper | 0 | 0 | ok | ok | AOT-clean |
