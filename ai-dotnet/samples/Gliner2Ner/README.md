# Gliner2Ner sample

English entity extraction with `fastino/gliner2-base-v1`, using only the published packages (`Starlings.Models.Gliner2`, which brings `Runtime.Onnx`, `Acquisition` and `Tokenizers`) and the ONNX Runtime CPU natives. **Preview** packages.

1. `download`: explicit acquisition of the pinned files: `tokenizer.json` from the Hub repository at the pinned commit, and the ONNX graph Starling converts from the same commit, from its explicit source. The only step that uses the network.
2. `extract`: offline. Verifies the files, loads the graph, and extracts entities of the given labels (default: person, organization, location, date, product) from the given texts or two built-in sentences, then shows that a text above 512 tokens is rejected.

```sh
dotnet run -- download [--cache <dir>]
dotnet run -- extract [--cache <dir> | --snapshot <dir>] [--threshold 0.5] [--labels person,location] ["your text" ...]
```

The graph's hosting location is not published yet (`Gliner2ModelAssets.ModelSourceIsPlaceholder`): until it is, `download` fails on the graph unless it is already in the snapshot directory (`<cache>/fastino/gliner2-base-v1/f9634218e53580c56edf0de97ca1a7d3f1c2354e/onnx/gliner2-entities-fp32.onnx`), or use `extract --snapshot <dir>` with a directory holding `tokenizer.json` and `onnx/gliner2-entities-fp32.onnx`. The graph is produced by `model-pipeline/reference/gliner2/export_gliner2_onnx.py` and `release_gliner2_onnx.py`; the lock checks its SHA-256.

The packages are not on nuget.org yet; `tools/release-drill-ai.sh` packs them into a local feed and builds this sample from a copy outside the repository.
