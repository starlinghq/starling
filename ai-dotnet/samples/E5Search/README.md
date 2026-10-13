# E5Search sample

Semantic search over five multilingual passages with `intfloat/multilingual-e5-small`, using only the published packages (`Starlings.Models.E5.Extensions.AI`, which brings `Models.E5`, `Runtime.Onnx`, `Acquisition` and `Tokenizers`) and the ONNX Runtime CPU natives. **Preview** packages.

1. `download`: explicit acquisition of the pinned model files (repository, commit, size and SHA-256 from the package's asset lock). The only step that uses the network.
2. `search`: offline. Verifies the cached files, loads the model, embeds passages with `"passage: "` and queries with `"query: "`, ranks by cosine similarity (dot product of unit vectors), then embeds the same texts through the Microsoft.Extensions.AI `IEmbeddingGenerator` and checks the vectors are identical.

```sh
dotnet run -- download [--cache <dir>]     # 487 MB from huggingface.co, once
dotnet run -- search [--cache <dir>] ["your query" ...]
```

Without `--cache` the files go to `$STARLING_MODELS` or the user cache directory (`~/.cache/starlings/models` on Linux).

The packages are not on nuget.org yet (publishing waits for the `release` environment and the NuGet prefix reservation, `docs/kickoff.md`). Until then, `tools/release-drill-ai.sh` packs them into a local feed and runs this sample from a copy outside the repository; `DRILL_KEEP=1` keeps that directory (its `feed/` and `apps/NuGet.config`) for your own use. The `Directory.Build.*` files in `samples/` only stop the repository's build settings from applying, so the sample builds the same anywhere.

For CUDA on Linux x64, replace `Microsoft.ML.OnnxRuntime` with `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 and set `OnnxSessionOptions.Provider = OnnxExecutionProvider.Cuda` (the CUDA 13 runtime must be on the loader path). That combination has not been run from packages.
