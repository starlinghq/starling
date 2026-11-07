# G4 soak: embeddinggemma-2/video-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off; multimodal text: cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 600 s / 605 s |
| Load/unload cycles | 53 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 159; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 10208, 18736 |
| Unload ms (median, max) | 98.0, 171.7 |
| Request p50 ms (median over cycles) | 253.7 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 53 | 0 | 46.26 | 80.34 | 122.28 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 45 | 201.7 | 202.0 | 201.8 | 201.7 | 202.2 | 1.3 | 0.2 | 0.0 | 0.0 | 32.0 | plateau |
| unloaded | working_set | yes | 45 | 1091.6 | 1063.8 | 1067.6 | 1063.7 | 1095.5 | -240.9 | -32.3 | -31.4 | -0.6 | 32.0 | plateau |
| unloaded | private | no | 45 | 1325.5 | 1291.5 | 1298.6 | 1279.9 | 1355.3 | -291.4 | -39.0 | -30.7 | 12.3 | 32.0 | plateau |
| unloaded | rss_anon | yes | 45 | 734.4 | 704.6 | 708.6 | 704.5 | 737.6 | -252.0 | -33.7 | -33.2 | -0.7 | 32.0 | plateau |
| unloaded | gpu | yes | 45 | 282.0 | 282.0 | 282.0 | 282.0 | 282.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 45 | 360.6 | 351.1 | 350.7 | 340.7 | 361.1 | -14.0 | -1.9 | -2.5 | 3.6 | 32.0 | plateau |
| loaded | working_set | yes | 45 | 1297.8 | 1261.7 | 1289.7 | 1237.0 | 1335.0 | -35.2 | -4.7 | -7.0 | 15.1 | 32.0 | plateau |
| loaded | private | no | 45 | 1559.3 | 1538.0 | 1553.8 | 1495.4 | 1595.0 | -8.2 | -1.1 | -11.5 | 36.0 | 32.0 | plateau |
| loaded | rss_anon | yes | 45 | 938.6 | 900.4 | 929.3 | 875.9 | 974.4 | -47.6 | -6.4 | -8.4 | 14.9 | 32.0 | plateau |
| loaded | gpu | yes | 45 | 4592.0 | 4470.0 | 4460.0 | 3564.0 | 5506.0 | 217.0 | 29.1 | 179.5 | 76.5 | 89.2 | plateau |

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
| command_line | embeddinggemma2-video --cuda --duration 10m --out <local>/out-0.5.0/soak/embeddinggemma2-video --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit ce5225854fa7, CUDA libraries from the pinned eg2 requirements-cuda wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit ce5225854fa7, CUDA libraries from the pinned eg2 requirements-cuda wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| load before (loadavg, MemAvailable MB) | [43.21,35.57,28.68], 56981 |
| load after (loadavg, MemAvailable MB) | [51.69,51.82,40.8], 62679 |
