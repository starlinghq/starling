# Starlings.Runtime.Onnx

**Preview.** ONNX Runtime session ownership for Starlings models: provider selection with no silent fallback (CPU, or CUDA with TF32 always off), native/managed version check, diagnostics (versions, providers, options, model SHA-256, graph signature), node-boundary cancellation and reference-counted disposal. The API may still change.

## Native runtime

The package references only `Microsoft.ML.OnnxRuntime.Managed` (exactly 1.30.0). The application references exactly one native package of the same version:

| Deployment | Package |
| --- | --- |
| CPU | `Microsoft.ML.OnnxRuntime` 1.30.0 |
| CUDA, Linux x64 | `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 (needs the CUDA 13 runtime on the loader path; not shipped) |

Both ship `libonnxruntime` under the same file name. Without a native package, or with a different version, session creation fails with `OnnxRuntimeUnavailableException` naming the package to add.

Most applications use this package through a model package such as `Starlings.Models.E5`. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#runtime-component-p-005).
