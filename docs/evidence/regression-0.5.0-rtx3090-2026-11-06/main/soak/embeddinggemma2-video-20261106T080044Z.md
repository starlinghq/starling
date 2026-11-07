# G4 soak: embeddinggemma-2/video-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off; multimodal text: cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 600 s / 604 s |
| Load/unload cycles | 129 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 387; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 4003, 4660 |
| Unload ms (median, max) | 69.7, 82.8 |
| Request p50 ms (median over cycles) | 228.4 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 129 | 0 | 51.41 | 94.92 | 132.67 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 104 | 201.8 | 201.8 | 201.9 | 201.8 | 202.3 | 1.9 | 0.3 | 0.1 | 0.1 | 32.0 | plateau |
| unloaded | working_set | yes | 104 | 1062.2 | 1065.7 | 1065.8 | 1062.2 | 1074.4 | -3.1 | -0.4 | 1.8 | -2.4 | 32.0 | plateau |
| unloaded | private | no | 104 | 1348.0 | 1341.7 | 1349.2 | 1320.7 | 1385.9 | -80.7 | -10.8 | -1.9 | 0.0 | 32.0 | plateau |
| unloaded | rss_anon | yes | 104 | 704.1 | 706.0 | 706.2 | 703.7 | 715.2 | -11.0 | -1.5 | 0.6 | -2.6 | 32.0 | plateau |
| unloaded | gpu | yes | 104 | 282.0 | 282.0 | 282.0 | 282.0 | 282.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 104 | 354.2 | 354.3 | 354.2 | 348.3 | 363.9 | -14.1 | -1.9 | 1.2 | -2.9 | 32.0 | plateau |
| loaded | working_set | yes | 104 | 1292.7 | 1297.1 | 1298.5 | 1268.4 | 1341.0 | -4.1 | -0.5 | 7.3 | -12.4 | 32.0 | plateau |
| loaded | private | no | 104 | 1638.4 | 1599.1 | 1619.0 | 1557.9 | 1686.4 | -76.4 | -10.2 | 4.6 | -7.5 | 32.4 | plateau |
| loaded | rss_anon | yes | 104 | 932.7 | 935.4 | 936.8 | 907.2 | 979.3 | -12.0 | -1.6 | 6.1 | -12.5 | 32.0 | plateau |
| loaded | gpu | yes | 104 | 4464.0 | 4464.0 | 4465.0 | 3566.0 | 5494.0 | -620.0 | -82.6 | 114.5 | -295.5 | 89.3 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-video --cuda --duration 10m --out <local>/out-main/soak/embeddinggemma2-video --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit 3a021dc0b7f8, CUDA libraries from the pinned eg2 requirements-cuda wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit 3a021dc0b7f8, CUDA libraries from the pinned eg2 requirements-cuda wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| load before (loadavg, MemAvailable MB) | [2.44,1.94,2.22], 74099 |
| load after (loadavg, MemAvailable MB) | [1.44,1.6,1.9], 73811 |
