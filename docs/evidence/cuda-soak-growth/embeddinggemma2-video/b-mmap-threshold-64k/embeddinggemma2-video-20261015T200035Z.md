# G4 soak: embeddinggemma-2/video-fp32-cuda

> **Short development run (720 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 720 s / 723 s |
| Load/unload cycles | 122 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 366; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 5392, 8273 |
| Unload ms (median, max) | 76.5, 89.8 |
| Request p50 ms (median over cycles) | 140.3 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 122 | 0 | 8.28 | 26.21 | 49.83 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 98 | 202.6 | 202.5 | 202.6 | 202.5 | 203.1 | 1.7 | 0.3 | 0.1 | 0.3 | 32.0 | plateau |
| unloaded | working_set | yes | 98 | 979.9 | 992.9 | 987.6 | 979.9 | 993.1 | 71.7 | 11.4 | 5.1 | 5.0 | 32.0 | plateau |
| unloaded | private | no | 98 | 1230.0 | 1241.5 | 1238.0 | 1223.7 | 1264.6 | -4.9 | -0.8 | 5.8 | 10.2 | 32.0 | plateau |
| unloaded | rss_anon | yes | 98 | 697.0 | 708.7 | 703.5 | 697.0 | 708.9 | 65.9 | 10.5 | 4.1 | 4.8 | 32.0 | plateau |
| loaded | managed_heap | yes | 98 | 281.4 | 281.2 | 281.4 | 280.7 | 281.8 | 1.9 | 0.3 | 0.1 | 0.3 | 32.0 | plateau |
| loaded | working_set | yes | 98 | 1111.4 | 1153.3 | 1142.6 | 1083.5 | 1153.4 | 60.8 | 9.7 | 11.1 | 3.8 | 32.0 | plateau |
| loaded | private | no | 98 | 1825.0 | 1865.4 | 1861.3 | 1816.6 | 1896.9 | 4.4 | 0.7 | 14.1 | 7.5 | 37.2 | plateau |
| loaded | rss_anon | yes | 98 | 826.5 | 867.1 | 857.1 | 797.9 | 867.2 | 54.9 | 8.8 | 10.1 | 3.6 | 32.0 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 7763 64-Core Processor |
| logical_cpus | 32 |
| ram_total_mb | 1031813 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.8.0-110-generic |
| runtime | .NET 10.0.0 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 4090, 580.126.20, 24564 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-video --cuda --duration 12m --out /workspace/out/starling-q-c2-fix/b-mmap-threshold-64k --note CUDA soak growth diagnostic b-mmap-threshold-64k, RunPod starling-q-c2-fix, NVIDIA GeForce RTX 4090, 580.126.20, CPU AMD EPYC 7763 64-Core Processor, CUDA libraries from the pinned wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| machine_note | CUDA soak growth diagnostic b-mmap-threshold-64k, RunPod starling-q-c2-fix, NVIDIA GeForce RTX 4090, 580.126.20, CPU AMD EPYC 7763 64-Core Processor, CUDA libraries from the pinned wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| load before (loadavg, MemAvailable MB) | [19.02,26.49,29.68], 935166 |
| load after (loadavg, MemAvailable MB) | [39.65,26.16,25.16], 945984 |
