# Gliner2Ner sample

English entity extraction with `fastino/gliner2-base-v1`, using only the published packages (`Starlings.Models.Gliner2`, which brings `Runtime.Onnx`, `Acquisition` and `Tokenizers`) and the ONNX Runtime CPU natives (`Microsoft.ML.OnnxRuntime` 1.30.0). Requires the .NET 10 SDK. The model's profiles `gliner2-base-v1/onnx-fp32-cpu` and `onnx-fp32-cuda` are Stable; maturity is per profile (the `Starlings.Models.Gliner2` README).

1. `download`: explicit acquisition of the pinned files: `tokenizer.json` from the Hub repository at the pinned commit, and the ONNX graph Starling converts from the same commit, from its explicit source. The only step that uses the network.
2. `extract`: offline. Verifies the files, loads the graph, and extracts entities of the given labels (default: person, organization, location, date, product) from the given texts or two built-in sentences, then shows that a text above 512 tokens is rejected.

```sh
dotnet run -- download [--cache <dir>]     # about 840 MB, once
dotnet run -- extract [--cache <dir> | --snapshot <dir>] [--threshold 0.5] [--labels person,location] ["your text" ...]
```

The graph downloads from the public Hugging Face repository `Starlinghq/gliner2-base-v1-onnx` at its pinned commit; no token is needed. To work offline, place it in the snapshot directory (`<cache>/fastino/gliner2-base-v1/f9634218e53580c56edf0de97ca1a7d3f1c2354e/onnx/gliner2-entities-fp32.onnx`), or use `extract --snapshot <dir>` with a directory holding `tokenizer.json` and `onnx/gliner2-entities-fp32.onnx`. The graph is produced by `model-pipeline/reference/gliner2/export_gliner2_onnx.py` and `release_gliner2_onnx.py`; the lock checks its SHA-256.

Without `--cache` the files go to `$STARLING_MODELS` or the user cache directory (`~/.cache/starlings/models` on Linux). `--labels` takes a comma-separated list; arguments that are not options are the texts.

The sample uses `PackageReference`s only, no project references: `Starlings.Models.Gliner2` 0.3.0 (the `StarlingsVersion` property) restores from nuget.org. `tools/release-drill-ai.sh` also packs the packages into a local feed and builds this sample from a copy outside the repository.

For CUDA on Linux x64, replace `Microsoft.ML.OnnxRuntime` with `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 and set `Provider = OnnxExecutionProvider.Cuda` on the `OnnxSessionOptions` in `Program.cs`; the sample has no CUDA flag, and the CUDA 13 runtime must be on the loader path. A Linux process that runs GLiNER2 in deployment needs `MALLOC_MMAP_THRESHOLD_=65536` in its environment, on CPU and CUDA (the `Starlings.Models.Gliner2` README, "Memory on Linux").
