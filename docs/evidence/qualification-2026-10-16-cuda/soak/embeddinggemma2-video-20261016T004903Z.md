# G4 soak: embeddinggemma-2/video-fp32-cuda

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 57 MB; loaded private (address space, report-only) grows 57 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3604 s |
| Load/unload cycles | 553 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 1659; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 5792, 8073 |
| Unload ms (median, max) | 78.5, 560.3 |
| Request p50 ms (median over cycles) | 225.1 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 553 | 0 | 52.49 | 83.29 | 151.69 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 441 | 212.6 | 212.9 | 212.5 | 210.5 | 228.7 | 2.8 | 2.2 | 0.9 | 1.9 | 32.0 | plateau |
| unloaded | working_set | yes | 441 | 996.4 | 1009.2 | 998.0 | 994.6 | 1009.2 | 5.8 | 4.7 | 3.8 | 2.3 | 32.0 | plateau |
| unloaded | private | no | 441 | 1445.7 | 1512.2 | 1455.3 | 1404.9 | 1512.2 | 70.7 | 56.5 | 35.8 | 22.6 | 32.0 | growth |
| unloaded | rss_anon | yes | 441 | 709.3 | 722.1 | 710.9 | 708.0 | 722.1 | 5.1 | 4.1 | 3.4 | 2.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 441 | 291.3 | 291.6 | 291.3 | 288.7 | 307.4 | 2.8 | 2.2 | 0.9 | 1.9 | 32.0 | plateau |
| loaded | working_set | yes | 441 | 1151.3 | 1137.6 | 1150.5 | 1119.8 | 1187.7 | 8.2 | 6.6 | 3.7 | 5.7 | 32.0 | plateau |
| loaded | private | no | 441 | 2039.9 | 2080.5 | 2053.8 | 1971.6 | 2125.0 | 70.8 | 56.7 | 39.0 | 23.2 | 41.1 | growth |
| loaded | rss_anon | yes | 441 | 862.7 | 848.5 | 861.5 | 831.0 | 898.5 | 7.5 | 6.0 | 3.3 | 5.7 | 32.0 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 7C13 64-Core Processor |
| logical_cpus | 32 |
| ram_total_mb | 1031831 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.8.0-111-generic |
| runtime | .NET 10.0.0 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 580.159.03, 24576 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-video --cuda --duration 1h --out /workspace/b/out/starling-q-b3/soak-embeddinggemma2-video --note G4 one-hour binding CUDA soak 2026-10-16 at e404702fa0fdbf91fd506757f1a5c15522d023dc, RunPod starling-q-b3, NVIDIA GeForce RTX 3090, 580.159.03, CPU AMD EPYC 7C13 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time on the pod, THP madvise, CUDA libraries from the pinned wheels (eg2 requirements-cuda), TF32 off, MALLOC_MMAP_THRESHOLD_=65536 |
| machine_note | G4 one-hour binding CUDA soak 2026-10-16 at e404702fa0fdbf91fd506757f1a5c15522d023dc, RunPod starling-q-b3, NVIDIA GeForce RTX 3090, 580.159.03, CPU AMD EPYC 7C13 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time on the pod, THP madvise, CUDA libraries from the pinned wheels (eg2 requirements-cuda), TF32 off, MALLOC_MMAP_THRESHOLD_=65536 |
| load before (loadavg, MemAvailable MB) | [20.4,27.24,27.2], 948962 |
| load after (loadavg, MemAvailable MB) | [56.57,58.12,57.15], 943536 |
