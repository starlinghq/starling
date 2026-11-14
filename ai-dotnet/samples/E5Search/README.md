# E5Search sample

Semantic search over five multilingual passages with `intfloat/multilingual-e5-small`, using only the published packages (`Starlings.Models.E5.Extensions.AI`, which brings `Models.E5`, `Runtime.Onnx`, `Acquisition` and `Tokenizers`) and the ONNX Runtime CPU natives (`Microsoft.ML.OnnxRuntime` 1.30.0). Requires the .NET 10 SDK. The model's profiles `multilingual-e5-small/onnx-fp32-cpu` and `onnx-fp32-cuda` are Stable; maturity is per profile (the `Starlings.Models.E5` README).

1. `download`: explicit acquisition of the pinned model files (repository, commit, size and SHA-256 from the package's asset lock). The only step that uses the network.
2. `search`: offline. Verifies the cached files, loads the model, embeds passages with `"passage: "` and queries with `"query: "`, ranks by cosine similarity (dot product of unit vectors), then embeds the same texts through the Microsoft.Extensions.AI `IEmbeddingGenerator` and prints the largest absolute difference from the typed API's vectors.

```sh
dotnet run -- download [--cache <dir>]     # 487 MB from huggingface.co, once
dotnet run -- search [--cache <dir>] ["your query" ...]
```

Without `--cache` the files go to `$STARLING_MODELS` or the user cache directory (`~/.cache/starlings/models` on Linux).

The sample uses `PackageReference`s only, no project references: `Starlings.*` 0.5.0 (the `StarlingsVersion` property) restores from nuget.org. The `Directory.Build.*` files in `samples/` only stop the repository's build settings (central package management, locked restore, analyzers) from applying, so the sample builds the same anywhere. `tools/release-drill-ai.sh` packs the packages into a local feed and runs this sample from a copy outside the repository; `DRILL_KEEP=1` keeps that directory (its `feed/` and `apps/NuGet.config`). The drill fails if `StarlingsVersion` differs from the version it packs.

## Container

A Dockerfile builds the same sample into a Linux image with no Python. Multi-stage: the SDK image publishes, the final layer is `mcr.microsoft.com/dotnet/runtime:10.0-noble-chiseled` (Ubuntu chiseled: no shell, no package manager, no Python; it ships the .NET runtime and the native libraries ONNX Runtime needs — libstdc++6, libgcc-s1, libgomp1). The final image is about 548 MB; the model is not baked in, it lives in a volume so a container restart does not re-download.

```sh
docker build -t starling-e5search .
docker volume create starling-models
docker run --rm -v starling-models:/models starling-e5search download    # once; 487 MB from huggingface.co
docker run --rm --network none -v starling-models:/models starling-e5search search "Warum rostet Eisen?"
```

The image sets `STARLING_MODELS=/models`; mount a volume (or bind mount) there — no `--cache` flag is needed. `--network none` works once the model is in the volume: every command except `download` is offline. Checking there is no Python:

```sh
$ docker export $(docker create starling-e5search) | tar -t | grep -i python
$ # no output: no Python anywhere in the image
```

Running the image gives the same scores as the quickstart above (`top: rust (0.9142)`, `top: summit (0.8816)`, CPUExecutionProvider, max abs difference from the typed API 0).

For CUDA on Linux x64, replace `Microsoft.ML.OnnxRuntime` with `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 and set `OnnxSessionOptions.Provider = OnnxExecutionProvider.Cuda` in `Program.cs`; the sample has no CUDA flag (the CUDA 13 runtime must be on the loader path). The release drill's CUDA mode runs e5 on CUDA from the packages through `tools/release-drill/cuda-consumer`, not through this sample.
