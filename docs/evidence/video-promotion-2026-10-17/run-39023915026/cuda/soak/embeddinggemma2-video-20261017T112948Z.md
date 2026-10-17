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
| Load/unload cycles | 86 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 258; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 6156, 8387 |
| Unload ms (median, max) | 86.9, 377.3 |
| Request p50 ms (median over cycles) | 239.7 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 86 | 0 | 54.87 | 90.08 | 116.12 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 70 | 169.7 | 169.8 | 169.9 | 169.7 | 170.3 | 1.8 | 0.2 | 0.1 | 0.2 | 32.0 | plateau |
| unloaded | working_set | yes | 70 | 942.3 | 942.4 | 943.6 | 939.8 | 960.0 | 22.5 | 3.0 | 2.4 | 0.8 | 32.0 | plateau |
| unloaded | private | no | 70 | 1412.1 | 1406.2 | 1415.6 | 1389.3 | 1444.6 | 36.9 | 4.9 | 9.2 | -5.2 | 32.0 | plateau |
| unloaded | rss_anon | yes | 70 | 656.3 | 654.8 | 656.8 | 653.4 | 672.8 | 13.2 | 1.8 | 1.3 | 0.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 70 | 321.5 | 319.8 | 322.5 | 317.0 | 330.8 | 13.3 | 1.8 | 1.9 | -0.4 | 32.0 | plateau |
| loaded | working_set | yes | 70 | 1165.1 | 1166.0 | 1167.2 | 1136.0 | 1210.5 | 41.4 | 5.5 | 0.7 | -18.0 | 32.0 | plateau |
| loaded | private | no | 70 | 2098.9 | 2085.5 | 2100.6 | 2040.1 | 2153.1 | 57.7 | 7.7 | 1.9 | -19.7 | 42.0 | plateau |
| loaded | rss_anon | yes | 70 | 877.1 | 876.3 | 877.8 | 846.5 | 921.0 | 31.7 | 4.2 | -0.6 | -18.4 | 32.0 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 7C13 64-Core Processor |
| logical_cpus | 32 |
| ram_total_mb | 515717 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.8.0-110-generic |
| runtime | .NET 10.0.0 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 580.126.20, 24576 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-video --cuda --duration 10m --out /root/ci-out/soak/embeddinggemma2-video --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit 6ea1bfa21855, CUDA libraries from the pinned eg2 requirements-cuda wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit 6ea1bfa21855, CUDA libraries from the pinned eg2 requirements-cuda wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| load before (loadavg, MemAvailable MB) | [13.33,28.28,30.73], 459534 |
| load after (loadavg, MemAvailable MB) | [5.31,23.51,29.11], 457492 |
