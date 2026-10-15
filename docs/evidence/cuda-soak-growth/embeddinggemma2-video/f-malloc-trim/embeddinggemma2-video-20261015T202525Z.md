# G4 soak: embeddinggemma-2/video-fp32-cuda

> **Short development run (720 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 720 s / 724 s |
| Load/unload cycles | 183 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 549; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3318, 6831 |
| Unload ms (median, max) | 70.0, 112.7 |
| Request p50 ms (median over cycles) | 140.9 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 183 | 0 | 7.52 | 27.03 | 48.69 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 147 | 202.6 | 202.6 | 202.6 | 202.5 | 203.6 | 3.4 | 0.5 | 0.2 | 0.5 | 32.0 | plateau |
| unloaded | working_set | yes | 147 | 1787.5 | 1777.9 | 1784.9 | 1758.7 | 1832.6 | 33.9 | 5.4 | -1.9 | 0.4 | 35.7 | plateau |
| unloaded | private | no | 147 | 1960.3 | 1971.8 | 1972.5 | 1936.1 | 2035.5 | 141.7 | 22.7 | 13.8 | 20.8 | 39.5 | plateau |
| unloaded | rss_anon | yes | 147 | 1504.3 | 1494.0 | 1501.3 | 1475.6 | 1548.8 | 30.4 | 4.9 | -2.6 | 0.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 147 | 281.4 | 281.3 | 281.4 | 281.2 | 282.3 | 3.4 | 0.5 | 0.2 | 0.5 | 32.0 | plateau |
| loaded | working_set | yes | 147 | 1826.0 | 1816.4 | 1887.1 | 1801.3 | 1938.3 | 87.1 | 13.9 | 19.0 | -13.7 | 37.7 | plateau |
| loaded | private | no | 147 | 2468.3 | 2479.8 | 2543.3 | 2437.9 | 2603.7 | 201.6 | 32.2 | 31.7 | 6.3 | 50.9 | plateau |
| loaded | rss_anon | yes | 147 | 1540.8 | 1530.5 | 1601.9 | 1515.7 | 1652.4 | 83.6 | 13.4 | 18.3 | -13.8 | 32.0 | plateau |
| trimmed | managed_heap | yes | 147 | 202.7 | 202.6 | 202.7 | 202.5 | 203.8 | 3.4 | 0.5 | 0.2 | 0.5 | 32.0 | plateau |
| trimmed | working_set | yes | 147 | 1042.7 | 1051.5 | 1052.3 | 1042.7 | 1059.6 | 80.3 | 12.8 | 6.1 | 0.4 | 32.0 | plateau |
| trimmed | private | no | 147 | 1933.2 | 1962.7 | 1954.9 | 1921.2 | 2006.6 | 194.1 | 31.1 | 20.4 | 21.3 | 39.1 | plateau |
| trimmed | rss_anon | yes | 147 | 759.5 | 767.6 | 768.5 | 759.5 | 775.8 | 76.8 | 12.3 | 5.4 | 0.3 | 32.0 | plateau |

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
| command_line | embeddinggemma2-video --cuda --duration 12m --malloc-trim --out /workspace/out/starling-q-c2-fix/f-malloc-trim --note CUDA soak growth diagnostic f-malloc-trim, RunPod starling-q-c2-fix, NVIDIA GeForce RTX 4090, 580.126.20, CPU AMD EPYC 7763 64-Core Processor, CUDA libraries from the pinned wheels |
| machine_note | CUDA soak growth diagnostic f-malloc-trim, RunPod starling-q-c2-fix, NVIDIA GeForce RTX 4090, 580.126.20, CPU AMD EPYC 7763 64-Core Processor, CUDA libraries from the pinned wheels |
| load before (loadavg, MemAvailable MB) | [33.98,24.16,25.2], 931007 |
| load after (loadavg, MemAvailable MB) | [40.65,22.61,21.51], 947666 |
