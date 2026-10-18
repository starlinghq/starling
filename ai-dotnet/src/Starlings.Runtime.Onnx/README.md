# Starlings.Runtime.Onnx

ONNX Runtime session ownership for Starlings models: provider selection with no silent fallback (CPU, or CUDA with TF32 always off), native/managed version check, diagnostics (versions, providers, options, model SHA-256, graph signature), cancellation (at node boundaries on CPU, after the current run on CUDA) and reference-counted disposal. The API may still change.

## Maturity

<!-- catalog:maturity profiles=multilingual-e5-small/*,gliner2-base-v1/*,embeddinggemma-2/*,kev-0.8b/*,kev-4b/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/image-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/image-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `embeddinggemma-2/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/video-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/video-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `gliner2-base-v1/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; linux-x64 cuda: supported; windows-x64 cpu: experimental |
| `gliner2-base-v1/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `kev-0.8b/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: experimental; windows-x64 cpu: untested |
| `kev-0.8b/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `kev-4b/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: experimental; windows-x64 cpu: untested |
| `kev-4b/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `multilingual-e5-small/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## API

```csharp
using Microsoft.ML.OnnxRuntime;
using Starlings.Runtime.Onnx;

using var session = OnnxModelSession.Create("model.onnx", new OnnxSessionOptions
{
    Provider = OnnxExecutionProvider.Cpu,          // or Cuda; no fallback either way
    ExpectedModelSha256 = "<lowercase hex sha256>", // optional; mismatch throws OnnxModelIntegrityException
});
Console.WriteLine(session.Diagnostics.OrtVersion);
using var outputs = session.Run(inputNames, inputValues, outputNames, cancellationToken);
```

- `OnnxModelSession.Create(modelPath | byte[] model, options)` loads once; `Run` is safe to call concurrently. One `Run` overload returns outputs the caller owns and disposes (before the session); the other writes into caller-allocated `OrtValue`s. Input values stay caller-owned. `Inputs`, `Outputs`, `Diagnostics`, `Placement` and `IsDisposed` describe the session. `Dispose` is idempotent; runs already executing complete first.
- `OnnxSessionOptions` (immutable record): `Provider` (default `Cpu`), `CudaDeviceId` (default 0), `IntraOpNumThreads` and `InterOpNumThreads` (null: defaults, below), `ExpectedModelSha256`, `ComputeModelSha256` (default true), `CapturePlacement` (default false; when true, profiles the first run and publishes `Placement`, an `OnnxPlacementReport` of nodes per provider), and the memory settings under "Memory on Linux".
- `OnnxSessionDiagnostics`: native and managed versions, available and session providers, provider options, thread counts, memory settings, graph optimization level, model source, size, SHA-256, metadata, and input and output signatures (`OnnxTensorInfo`).
- `OnnxRuntimeInfo`: `NativeVersion`, `ManagedVersion`, `AvailableProviders`, `CpuProviderName`, `CudaProviderName`.
- Exceptions: `OnnxRuntimeUnavailableException` (native library missing or of another version), `OnnxProviderUnavailableException` (the requested provider cannot be used; never followed by a fallback), `OnnxModelIntegrityException` (SHA-256 mismatch). A cancelled run throws `OperationCanceledException`; the session stays usable.

## Native runtime

The package references only `Microsoft.ML.OnnxRuntime.Managed` (exactly 1.30.0). The application references exactly one native package of the same version:

| Deployment | Package |
| --- | --- |
| CPU | `Microsoft.ML.OnnxRuntime` 1.30.0 |
| CUDA, Linux x64 | `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 (needs the CUDA 13 runtime, cuBLAS and cuDNN 9 on the loader path; not shipped) |

Both ship `libonnxruntime` under the same file name. Without a native package, or with a different version, session creation fails with `OnnxRuntimeUnavailableException` naming the package to add.

## Threads in containers

When `IntraOpNumThreads` is not set and a CPU quota or affinity mask limits the process (`Environment.ProcessorCount` below the host's online CPUs on Linux), the session uses `Environment.ProcessorCount` intra-op threads. ONNX Runtime's own default ignores a container's CPU quota and starts one thread per physical core of the host (147 threads, each with its own glibc malloc arena, for one CUDA e5 session in a 31-CPU container on a 256-CPU host). On an unconstrained host the ONNX Runtime default is kept. `OnnxSessionDiagnostics.IntraOpNumThreads` reports the count used. Evidence: [CUDA soak growth](https://github.com/starlinghq/starling/blob/main/docs/evidence/cuda-soak-growth/README.md).

Most applications use this package through a model package such as `Starlings.Models.E5`. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#runtime-component-p-005).

## Memory on Linux

When a session's native memory is released (after `Dispose` and the last in-flight run), the package calls glibc's `malloc_trim(0)` on Linux. glibc otherwise keeps the freed memory of the session (graph, initializers, ONNX Runtime's CPU arena, run buffers) resident in its heaps, so a process that loads and unloads models keeps up to about a model's worth of freed memory per malloc arena. On the hosted qualification runner the resident memory after unloading fell from about 1,480 to 400 to 500 MiB for EmbeddingGemma 2 text and from about 1,110 to 260 MiB for GLiNER2 without `MALLOC_MMAP_THRESHOLD_` ([evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/scheduled-regression-fix/README.md)).

- `malloc_trim` only returns memory that is already free. It changes no allocator setting (unlike `mallopt` or the `MALLOC_*` environment variables) and touches no live allocation. It runs once per released session, not on the request path; it costs a few milliseconds per release, and the next load faults the returned pages in again. The opt-in trims after load and after each run (below) call the same function.
- It is a no-op on Windows and macOS, and on Linux without glibc (musl: the first release finds no `libc.so.6` and turns it off; `OnnxSessionDiagnostics.TrimNativeHeapOnRelease` can report `true` for sessions created before that).
- Opt out per session with `OnnxSessionOptions.TrimNativeHeapOnRelease = false`, or for the whole process (also for sessions a model package creates) with the `AppContext` switch `Starlings.Runtime.Onnx.DisableNativeHeapTrim`, for example in the application's project file:

  ```xml
  <ItemGroup>
    <RuntimeHostConfigurationOption Include="Starlings.Runtime.Onnx.DisableNativeHeapTrim" Value="true" />
  </ItemGroup>
  ```

- CPU memory arena: `OnnxSessionOptions.CpuMemoryArena` turns ONNX Runtime's CPU arena (`enable_cpu_mem_arena`) on or off; `null`, the default, keeps ONNX Runtime's default (on) for sessions created directly and lets a model package choose for the sessions it creates (EmbeddingGemma 2 turns it off on CPU). With the arena, ONNX Runtime keeps the CPU buffers of every run in regions it does not return while the session lives; their resident size can climb toward the region size over many runs, and regions of up to 32 MiB freed at release stay resident in glibc's per-thread heaps, which `malloc_trim` cannot return. Outputs are the same either way. `OnnxSessionDiagnostics.CpuMemoryArena` reports the effective setting. Evidence: [EmbeddingGemma 2 hosted soak](https://github.com/starlinghq/starling/blob/main/docs/evidence/eg2-hosted-soak/README.md).
- Dedicated run threads: with `OnnxSessionOptions.DedicatedRunThreads = true`, runs execute on a small process-wide set of long-lived threads owned by this package, and the caller blocks until its run completes. A thread is started only when every existing one is busy, up to `Environment.ProcessorCount`; further concurrent runs wait for a free one. `null`, the default, runs on the calling thread for sessions created directly and lets a model package choose (EmbeddingGemma 2 and e5 use them on CPU). Why: without ONNX Runtime's CPU arena, a run allocates and frees its buffers with `malloc` on the thread that calls it, and glibc serves each thread from its own malloc arena. The free memory at the top of a non-main arena stays resident: `malloc_trim` does not return it, and `free` returns it only above the dynamic trim threshold (up to 64 MiB). When runs come from a rotating set of thread-pool threads, more and more arenas each keep 20 to 60 MiB, and resident memory steps up over an hour. On the dedicated threads the same few arenas serve every run, so that memory is reused. Results are the same either way. Cancellation works the same (the token sets ONNX Runtime's terminate flag); a run started from a dedicated thread executes inline. `OnnxSessionDiagnostics.DedicatedRunThreads` reports the setting. Evidence: [EmbeddingGemma 2 soak reliability](https://github.com/starlinghq/starling/blob/main/docs/evidence/eg2-soak-reliability/README.md).
- Trims after load and after each run: `OnnxSessionOptions.TrimNativeHeapAfterLoad` calls `malloc_trim(0)` once when the session has been created, and `OnnxSessionOptions.TrimNativeHeapAfterRun` calls it on the thread that ran, after each run. `null`, the default, means off for sessions created directly and lets a model package choose (the EmbeddingGemma 2 image and video path turns both on for its CPU sessions). Why: creating a large session allocates and frees copies of the graph and its initializers, and glibc keeps that memory resident while the session lives (about 400 MiB for the EmbeddingGemma 2 vision and multimodal sessions, varying from load to load). A run without ONNX Runtime's arena leaves its freed buffers as free chunks in the running thread's heap (60 to 190 MiB per thread for EmbeddingGemma 2 images, rising in steps). The trim returns both; it cannot return the top chunk of a non-main heap. The run trim is part of the run (milliseconds) and the next run faults the pages in again, so it suits runs that take seconds, not short ones. The `AppContext` switch above turns both off. `OnnxSessionDiagnostics.TrimNativeHeapAfterLoad` and `TrimNativeHeapAfterRun` report them. Evidence: [EmbeddingGemma 2 image loaded soak](https://github.com/starlinghq/starling/blob/main/docs/evidence/eg2-image-loaded-soak/README.md).
- Transparent huge pages: with `/sys/kernel/mm/transparent_hugepage/enabled` set to `always`, the kernel's khugepaged collapses partly used 2 MiB ranges of the heaps into huge pages and fills in their absent pages, so resident memory can rise over time with no allocation behind it. A library cannot change that policy for its host; `madvise` (the Ubuntu default) or `never` avoids it.
