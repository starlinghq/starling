# G4 soak: embeddinggemma-2/video-fp32-cuda

> **Short development run (720 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** FAIL: loaded working_set grows 51 MB (319 MB/h); loaded rss_anon grows 50 MB (316 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 720 s / 722 s |
| Load/unload cycles | 214 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 642; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2870, 6215 |
| Unload ms (median, max) | 68.6, 142.1 |
| Request p50 ms (median over cycles) | 139.9 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 214 | 0 | 8.55 | 30.42 | 44.48 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 173 | 202.6 | 202.5 | 202.6 | 196.2 | 218.9 | 24.9 | 4.0 | 0.4 | 3.5 | 32.0 | plateau |
| unloaded | working_set | yes | 173 | 1816.6 | 1872.5 | 1849.5 | 1789.8 | 1921.5 | 249.6 | 39.9 | 4.7 | 31.7 | 37.0 | plateau |
| unloaded | private | no | 173 | 1979.1 | 2026.9 | 1993.0 | 1936.0 | 2068.1 | 232.0 | 37.1 | -2.5 | 26.6 | 39.9 | plateau |
| unloaded | rss_anon | yes | 173 | 1533.5 | 1588.6 | 1565.8 | 1506.6 | 1637.7 | 246.4 | 39.4 | 4.1 | 31.6 | 32.0 | plateau |
| loaded | managed_heap | yes | 173 | 281.4 | 281.3 | 281.4 | 274.9 | 297.7 | 24.9 | 4.0 | 0.4 | 3.5 | 32.0 | plateau |
| loaded | working_set | yes | 173 | 1883.1 | 1906.9 | 1933.3 | 1835.8 | 2037.9 | 318.7 | 50.9 | 34.9 | 32.5 | 38.7 | growth |
| loaded | private | no | 173 | 2507.1 | 2522.9 | 2545.0 | 2459.1 | 2638.2 | 299.3 | 47.8 | 30.2 | 28.5 | 50.9 | plateau |
| loaded | rss_anon | yes | 173 | 1597.9 | 1621.0 | 1647.5 | 1550.4 | 1752.1 | 315.6 | 50.4 | 34.4 | 32.4 | 33.0 | growth |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-video --cuda --duration 12m --out /workspace/out/starling-q-c2-fix/a-baseline --note CUDA soak growth diagnostic a-baseline, RunPod starling-q-c2-fix, NVIDIA GeForce RTX 4090, 580.126.20, CPU AMD EPYC 7763 64-Core Processor, CUDA libraries from the pinned wheels |
| machine_note | CUDA soak growth diagnostic a-baseline, RunPod starling-q-c2-fix, NVIDIA GeForce RTX 4090, 580.126.20, CPU AMD EPYC 7763 64-Core Processor, CUDA libraries from the pinned wheels |
| load before (loadavg, MemAvailable MB) | [24.72,32.63,32.06], 921115 |
| load after (loadavg, MemAvailable MB) | [22.28,27.46,30.04], 942186 |
