# Starlings.Runtime.Onnx

ONNX Runtime session ownership for Starlings models: provider selection with no silent fallback (CPU, or CUDA with TF32 always off), native/managed version check, diagnostics (versions, providers, options, model SHA-256, graph signature), cancellation (at node boundaries on CPU, after the current run on CUDA) and reference-counted disposal. The API may still change.

## Maturity

<!-- catalog:maturity profiles=multilingual-e5-small/*,gliner2-base-v1/*,embeddinggemma-2/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/image-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/image-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `embeddinggemma-2/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/video-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `embeddinggemma-2/video-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `gliner2-base-v1/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; linux-x64 cuda: supported; windows-x64 cpu: experimental |
| `gliner2-base-v1/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `multilingual-e5-small/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Native runtime

The package references only `Microsoft.ML.OnnxRuntime.Managed` (exactly 1.30.0). The application references exactly one native package of the same version:

| Deployment | Package |
| --- | --- |
| CPU | `Microsoft.ML.OnnxRuntime` 1.30.0 |
| CUDA, Linux x64 | `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 (needs the CUDA 13 runtime on the loader path; not shipped) |

Both ship `libonnxruntime` under the same file name. Without a native package, or with a different version, session creation fails with `OnnxRuntimeUnavailableException` naming the package to add.

## Threads in containers

When `IntraOpNumThreads` is not set and a CPU quota or affinity mask limits the process (`Environment.ProcessorCount` below the host's online CPUs on Linux), the session uses `Environment.ProcessorCount` intra-op threads. ONNX Runtime's own default ignores a container's CPU quota and starts one thread per physical core of the host (147 threads, each with its own glibc malloc arena, for one CUDA e5 session in a 31-CPU container on a 256-CPU host). On an unconstrained host the ONNX Runtime default is kept. `OnnxSessionDiagnostics.IntraOpNumThreads` reports the count used. Evidence: [CUDA soak growth](https://github.com/starlinghq/starling/blob/main/docs/evidence/cuda-soak-growth/README.md).

Most applications use this package through a model package such as `Starlings.Models.E5`. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#runtime-component-p-005).

## Memory on Linux

When a session's native memory is released (after `Dispose` and the last in-flight run), the package calls glibc's `malloc_trim(0)` on Linux. glibc otherwise keeps the freed memory of the session (graph, initializers, ONNX Runtime's CPU arena, run buffers) resident in its heaps, so a process that loads and unloads models keeps up to about a model's worth of freed memory per malloc arena. On the hosted qualification runner the resident memory after unloading fell from about 1,480 to 400 to 500 MiB for EmbeddingGemma 2 text and from about 1,110 to 260 MiB for GLiNER2 without `MALLOC_MMAP_THRESHOLD_` ([evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/scheduled-regression-fix/README.md)).

- `malloc_trim` only returns memory that is already free. It changes no allocator setting (unlike `mallopt` or the `MALLOC_*` environment variables) and touches no live allocation. It runs once per released session, never on the request path; it costs a few milliseconds per release, and the next load faults the returned pages in again.
- It is a no-op on Windows and macOS, and on Linux without glibc (musl: the first release finds no `libc.so.6` and turns it off; `OnnxSessionDiagnostics.TrimNativeHeapOnRelease` can report `true` for sessions created before that).
- Opt out per session with `OnnxSessionOptions.TrimNativeHeapOnRelease = false`, or for the whole process (also for sessions a model package creates) with the `AppContext` switch `Starlings.Runtime.Onnx.DisableNativeHeapTrim`, for example in the application's project file:

  ```xml
  <ItemGroup>
    <RuntimeHostConfigurationOption Include="Starlings.Runtime.Onnx.DisableNativeHeapTrim" Value="true" />
  </ItemGroup>
  ```

- CPU memory arena: `OnnxSessionOptions.CpuMemoryArena` turns ONNX Runtime's CPU arena (`enable_cpu_mem_arena`) on or off; `null`, the default, keeps ONNX Runtime's default (on) for sessions created directly and lets a model package choose for the sessions it creates (EmbeddingGemma 2 turns it off on CPU). With the arena, ONNX Runtime keeps the CPU buffers of every run in regions it does not return while the session lives; their resident size can climb toward the region size over many runs, and regions of up to 32 MiB freed at release stay resident in glibc's per-thread heaps, which `malloc_trim` cannot return. Outputs are the same either way. `OnnxSessionDiagnostics.CpuMemoryArena` reports the effective setting. Evidence: [EmbeddingGemma 2 hosted soak](https://github.com/starlinghq/starling/blob/main/docs/evidence/eg2-hosted-soak/README.md).
- Transparent huge pages: with `/sys/kernel/mm/transparent_hugepage/enabled` set to `always`, the kernel's khugepaged collapses partly used 2 MiB ranges of the heaps into huge pages and fills in their absent pages, so resident memory can rise over time with no allocation behind it. A library cannot change that policy for its host; `madvise` (the Ubuntu default) or `never` avoids it.
