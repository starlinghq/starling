# G4 soak: embeddinggemma-2/image-fp32-cuda

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 226 MB; loaded private (address space, report-only) grows 138 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 799 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 3196; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3350, 6624 |
| Unload ms (median, max) | 201.9, 295.0 |
| Request p50 ms (median over cycles) | 270.8 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 799 | 0 | 64.51 | 104.22 | 109.21 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 638 | 212.1 | 206.8 | 202.0 | 181.8 | 224.3 | 8.1 | 6.5 | 1.4 | 4.3 | 32.0 | plateau |
| unloaded | working_set | yes | 638 | 1126.7 | 1134.5 | 1133.1 | 1074.3 | 1159.8 | 16.5 | 13.2 | 10.3 | 14.5 | 32.0 | plateau |
| unloaded | private | no | 638 | 2091.0 | 2304.5 | 2198.6 | 2058.0 | 2340.0 | 282.8 | 226.3 | 62.7 | 130.3 | 44.0 | growth |
| unloaded | rss_anon | yes | 638 | 840.6 | 847.8 | 846.7 | 787.6 | 873.2 | 15.7 | 12.6 | 9.9 | 14.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 638 | 290.8 | 285.4 | 280.8 | 260.0 | 303.0 | 8.1 | 6.5 | 1.4 | 4.2 | 32.0 | plateau |
| loaded | working_set | yes | 638 | 2065.1 | 2043.5 | 2035.8 | 1877.8 | 2122.5 | 37.5 | 30.0 | 36.5 | 16.9 | 40.7 | plateau |
| loaded | private | no | 638 | 2845.7 | 2926.6 | 2874.8 | 2674.3 | 3016.1 | 171.9 | 137.5 | 93.2 | 64.1 | 57.5 | growth |
| loaded | rss_anon | yes | 638 | 1776.9 | 1754.9 | 1747.4 | 1589.6 | 1833.9 | 36.8 | 29.4 | 36.1 | 16.9 | 34.9 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 7H12 64-Core Processor |
| logical_cpus | 32 |
| ram_total_mb | 1031828 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.8.0-107-generic |
| runtime | .NET 10.0.0 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 580.126.20, 24576 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-image --cuda --duration 1h --out /workspace/b/out/starling-q-b4/soak-embeddinggemma2-image --note G4 one-hour binding CUDA soak 2026-10-16 at e404702fa0fdbf91fd506757f1a5c15522d023dc, RunPod starling-q-b4, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time on the pod, THP madvise, CUDA libraries from the pinned wheels (eg2 requirements-cuda), TF32 off |
| machine_note | G4 one-hour binding CUDA soak 2026-10-16 at e404702fa0fdbf91fd506757f1a5c15522d023dc, RunPod starling-q-b4, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time on the pod, THP madvise, CUDA libraries from the pinned wheels (eg2 requirements-cuda), TF32 off |
| load before (loadavg, MemAvailable MB) | [13.09,14.99,13.36], 861188 |
| load after (loadavg, MemAvailable MB) | [10.53,10.33,10.33], 898544 |
