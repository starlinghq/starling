# G4 soak: embeddinggemma-2/image-fp32-cuda

**Verdict:** FAIL: unloaded working_set grows 69 MB (86 MB/h); unloaded rss_anon grows 68 MB (85 MB/h); loaded working_set grows 63 MB (78 MB/h); loaded rss_anon grows 62 MB (77 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 893 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 3572; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3177, 7109 |
| Unload ms (median, max) | 103.4, 178.4 |
| Request p50 ms (median over cycles) | 257.9 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 893 | 0 | 5.30 | 56.96 | 63.88 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 714 | 180.4 | 184.8 | 181.5 | 180.4 | 194.4 | 2.5 | 2.0 | 0.9 | 1.6 | 32.0 | plateau |
| unloaded | working_set | yes | 714 | 1801.5 | 1950.0 | 1870.3 | 1801.5 | 1997.4 | 86.0 | 68.9 | 38.5 | 35.6 | 37.4 | growth |
| unloaded | private | no | 714 | 1973.4 | 2124.3 | 2034.5 | 1950.3 | 2149.4 | 88.6 | 70.9 | 37.0 | 40.8 | 40.7 | retention |
| unloaded | rss_anon | yes | 714 | 1515.5 | 1663.6 | 1583.9 | 1515.5 | 1710.9 | 85.4 | 68.4 | 38.1 | 35.6 | 32.0 | growth |
| loaded | managed_heap | yes | 714 | 259.1 | 263.5 | 260.2 | 258.5 | 273.1 | 2.5 | 2.0 | 0.9 | 1.6 | 32.0 | plateau |
| loaded | working_set | yes | 714 | 2086.1 | 2050.0 | 2073.8 | 1921.5 | 2178.9 | 78.1 | 62.5 | 50.0 | 28.4 | 41.5 | growth |
| loaded | private | no | 714 | 4254.1 | 4220.3 | 4235.0 | 4075.7 | 4352.5 | 79.8 | 63.9 | 47.4 | 33.3 | 84.7 | plateau |
| loaded | rss_anon | yes | 714 | 1798.1 | 1761.5 | 1785.6 | 1633.4 | 1890.5 | 77.4 | 62.0 | 49.6 | 28.4 | 35.7 | growth |

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
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 580.159.03, 24576 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-image --cuda --duration 1h --out /workspace/out/starling-q-w2/soak-embeddinggemma2-image --note G4 one-hour qualification soak, cuda, RunPod starling-q-w2, NVIDIA GeForce RTX 3090, 580.159.03, CPU AMD EPYC 7763 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time, CUDA libraries from the pinned wheels (eg2 requirements-cuda) |
| machine_note | G4 one-hour qualification soak, cuda, RunPod starling-q-w2, NVIDIA GeForce RTX 3090, 580.159.03, CPU AMD EPYC 7763 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time, CUDA libraries from the pinned wheels (eg2 requirements-cuda) |
| load before (loadavg, MemAvailable MB) | [8.12,8.16,13.11], 968376 |
| load after (loadavg, MemAvailable MB) | [15.95,17.39,17.64], 960193 |
