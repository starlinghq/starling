# G4 soak: multilingual-e5-small/onnx-fp32-cuda

**Verdict:** FAIL: unloaded working_set grows 80 MB (141 MB/h); unloaded rss_anon grows 80 MB (140 MB/h); loaded working_set grows 84 MB (147 MB/h); loaded rss_anon grows 83 MB (146 MB/h); trimmed working_set grows 80 MB (141 MB/h); trimmed rss_anon grows 80 MB (140 MB/h); 1 unexpected errors

## Run

| Item | Value |
| --- | --- |
| Model lock | intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 7200 s / 2572 s |
| Load/unload cycles | 1286 |
| Requests per cycle, concurrency | 200, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 231480; wrong outputs 0; unexpected errors 1 |
| Load ms (median, max) | 1124, 2708 |
| Unload ms (median, max) | 24.9, 76.6 |
| Request p50 ms (median over cycles) | 2.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3528 | 22192 | 9.37 | 18.36 | 36.26 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 1031 | 48.9 | 40.9 | 42.0 | 40.3 | 60.7 | 3.1 | 1.8 | -1.7 | 0.6 | 32.0 | plateau |
| unloaded | working_set | yes | 1031 | 967.3 | 1041.2 | 1019.3 | 962.1 | 1075.8 | 140.6 | 80.3 | 56.5 | 36.3 | 32.0 | growth |
| unloaded | private | no | 1031 | 1109.3 | 1224.2 | 1171.5 | 1088.0 | 1301.9 | 194.7 | 111.2 | 49.0 | 89.4 | 32.0 | growth |
| unloaded | rss_anon | yes | 1031 | 611.8 | 685.1 | 663.3 | 606.5 | 719.7 | 139.6 | 79.7 | 56.3 | 35.9 | 32.0 | growth |
| unloaded | gpu | yes | 1031 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 1031 | 60.4 | 52.4 | 53.5 | 51.8 | 72.2 | 3.1 | 1.8 | -1.7 | 0.6 | 32.0 | plateau |
| loaded | working_set | yes | 1031 | 1074.7 | 1157.1 | 1135.9 | 1074.1 | 1194.9 | 146.6 | 83.7 | 61.2 | 36.7 | 32.0 | growth |
| loaded | private | no | 1031 | 1218.3 | 1307.1 | 1277.7 | 1200.2 | 1360.8 | 187.2 | 106.9 | 54.2 | 76.6 | 32.0 | growth |
| loaded | rss_anon | yes | 1031 | 717.2 | 799.0 | 778.0 | 716.6 | 836.7 | 145.6 | 83.2 | 60.9 | 36.3 | 32.0 | growth |
| loaded | gpu | yes | 1031 | 950.0 | 1400.0 | 1398.0 | 950.0 | 1464.0 | 38.4 | 21.9 | 79.3 | -13.6 | 32.0 | plateau |
| trimmed | managed_heap | yes | 1031 | 49.2 | 41.2 | 42.1 | 40.4 | 60.9 | 3.1 | 1.8 | -1.7 | 0.6 | 32.0 | plateau |
| trimmed | working_set | yes | 1031 | 967.3 | 1041.2 | 1019.2 | 962.0 | 1075.8 | 140.6 | 80.3 | 56.5 | 36.3 | 32.0 | growth |
| trimmed | private | no | 1031 | 1109.3 | 1224.2 | 1162.0 | 1088.0 | 1294.5 | 206.9 | 118.1 | 48.6 | 97.6 | 32.0 | growth |
| trimmed | rss_anon | yes | 1031 | 611.7 | 685.1 | 663.3 | 606.5 | 719.7 | 139.6 | 79.8 | 56.3 | 35.9 | 32.0 | growth |
| trimmed | gpu | yes | 1031 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Errors and wrong outputs (first 50)

- cycle 1286 load: OnnxProviderUnavailableException: The CUDA execution provider could not be initialized: [ErrorCode:Fail] Exception during initialization: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:360 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) Failed to allocate memory for requested buffer of size 384056832
 It needs libonnxruntime_providers_cuda from Microsoft.ML.OnnxRuntime.Gpu.Linux next to libonnxruntime, the CUDA 13 runtime (libcudart, libcublas, libcublasLt) and cuDNN 9 on the native loader path (LD_LIBRARY_PATH or the ldconfig cache on Linux), a compatible NVIDIA driver, and a visible device. No CPU fallback was attempted.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz |
| logical_cpus | 12 |
| ram_total_mb | 96442 |
| os | Omarchy |
| kernel | 7.2.8-9-omarchy |
| runtime | .NET 10.0.14 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 615.78.08, 24576 MiB; 1, NVIDIA GeForce RTX 4060 Ti, 615.78.08, 16380 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | e5 --cuda --duration 2h --out <local>/out-e5-main-2h-trim/e5 --note local RTX 3090, main, two hours, malloc-trim diagnostic --malloc-trim |
| machine_note | local RTX 3090, main, two hours, malloc-trim diagnostic |
| load before (loadavg, MemAvailable MB) | [1.01,1.04,1.02], 76660 |
| load after (loadavg, MemAvailable MB) | [4.19,3.62,2.68], 71471 |
