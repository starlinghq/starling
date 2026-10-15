# Starlings.Runtime.Onnx

ONNX Runtime session ownership for Starlings models: provider selection with no silent fallback (CPU, or CUDA with TF32 always off), native/managed version check, diagnostics (versions, providers, options, model SHA-256, graph signature), node-boundary cancellation and reference-counted disposal. The API may still change.

## Maturity

<!-- catalog:maturity profiles=multilingual-e5-small/*,gliner2-base-v1/*,embeddinggemma-2/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/image-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `embeddinggemma-2/image-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `embeddinggemma-2/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: experimental |
| `embeddinggemma-2/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `embeddinggemma-2/video-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `embeddinggemma-2/video-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `gliner2-base-v1/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; linux-x64 cuda: experimental; windows-x64 cpu: experimental |
| `gliner2-base-v1/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `multilingual-e5-small/onnx-fp32-cpu` | preview | linux-x64 cpu: experimental; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Native runtime

The package references only `Microsoft.ML.OnnxRuntime.Managed` (exactly 1.30.0). The application references exactly one native package of the same version:

| Deployment | Package |
| --- | --- |
| CPU | `Microsoft.ML.OnnxRuntime` 1.30.0 |
| CUDA, Linux x64 | `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 (needs the CUDA 13 runtime on the loader path; not shipped) |

Both ship `libonnxruntime` under the same file name. Without a native package, or with a different version, session creation fails with `OnnxRuntimeUnavailableException` naming the package to add.

Most applications use this package through a model package such as `Starlings.Models.E5`. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#runtime-component-p-005).
