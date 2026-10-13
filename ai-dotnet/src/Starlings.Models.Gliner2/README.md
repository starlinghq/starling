# Starlings.Models.Gliner2

**Preview.** English entity extraction with [`fastino/gliner2-base-v1`](https://huggingface.co/fastino/gliner2-base-v1) at revision `f9634218e53580c56edf0de97ca1a7d3f1c2354e` (GLiNER2 2.0.0, source `55656fbf`), on ONNX Runtime 1.30.0, in-process and offline after an explicit acquisition. You name the entity labels; the model returns labelled spans with offsets and confidences. The API and the supported platforms may still change.

- GLiNER2's text handling is ported with Python semantics (normalization, the regular-expression word splitter, `str.lower()`), and the schema prefix is encoded with the owned tokenizer `Starlings.Tokenizers` (profile `gliner2-base-v1-unigram`). No Microsoft.ML.Tokenizers, no native tokenizer, no Python.
- One ONNX graph (token ids and routing positions to count and span logits); GLiNER2's count gate, threshold, overlap suppression and result order in C#.
- The package ships the asset lock (sizes and SHA-256), not the 838 MB of model files. Acquisition is an explicit call; inference never downloads.

## Install

```sh
dotnet add package Starlings.Models.Gliner2 --prerelease
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # CPU natives
# or, for CUDA on Linux x64: Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0 (reference exactly one)
```

## Use

```csharp
using Starlings.Models.Gliner2;

Gliner2ModelFiles files = Gliner2ModelAssets.OpenLocal();   // offline, verified, never downloads
using var extractor = Gliner2EntityExtractor.Create(files);

var result = extractor.Extract(
    "Apple CEO Tim Cook announced the iPhone 15 in Cupertino.",
    ["company", "person", "product", "location"],
    threshold: 0.5f);
foreach (var e in result.Entities)
{
    Console.WriteLine($"{e.Label}: {e.Text} [{e.Start}, {e.End}) {e.Confidence:F3}");
}
```

The ONNX graph is not on the Hub: Starling converts the pinned checkpoint (reproducible export, digest `0ba2db95...`) and will host it as a release asset. Until that release exists, `Gliner2ModelAssets.DownloadAsync` fetches `tokenizer.json` and reports the graph as not found; place the exported graph at `onnx/gliner2-entities-fp32.onnx` in the snapshot directory or call `Gliner2ModelAssets.OpenDirectory(dir)`.

## Contract

- Entity extraction with labels and a threshold, English. Offsets (`Start`, `End` in UTF-16 code units; `ScalarStart`, `ScalarEnd` in Unicode scalar values) index the original text and never exceed its length; `Text == input[Start..End]`.
- Texts above 512 tokens (schema prefix included) are rejected with `Gliner2InputTooLongException`; nothing is truncated.
- One label set per call. Result order: labels in request order, then descending confidence.
- Not offered: classification, relation extraction, structured extraction, GLiNER2's plain de-duplicated output, per-label thresholds, `extract_long`. Label descriptions: untested. Full list: `Gliner2Capabilities.All`.

## Status

Profile `gliner2-base-v1/onnx-fp32-cpu`: Linux x64 CPU and CUDA (`use_tf32=0`) experimental against the protected development fixtures and binding tolerances (one workstation). Evidence and reproduction: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#gliner2-entity-extraction-p-015). License of the weights and the converted graph: Apache-2.0 (model card); GLiNER2 by Fastino AI, backbone initialised from `microsoft/deberta-v3-base` (MIT).
