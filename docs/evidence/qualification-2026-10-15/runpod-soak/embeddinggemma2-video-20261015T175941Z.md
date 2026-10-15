# G4 soak: embeddinggemma-2/video-fp32-cuda

**Verdict:** FAIL: unloaded working_set grows 84 MB (105 MB/h); unloaded rss_anon grows 84 MB (105 MB/h); loaded working_set grows 83 MB (104 MB/h); loaded rss_anon grows 82 MB (103 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 975 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 2925; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3183, 9009 |
| Unload ms (median, max) | 84.8, 145.7 |
| Request p50 ms (median over cycles) | 140.9 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 975 | 0 | 7.29 | 30.72 | 53.99 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 780 | 202.5 | 207.0 | 203.6 | 196.3 | 222.8 | 3.6 | 2.9 | 1.3 | 0.9 | 32.0 | plateau |
| unloaded | working_set | yes | 780 | 1968.7 | 2058.8 | 1991.6 | 1911.8 | 2106.2 | 105.2 | 84.2 | 30.6 | 64.4 | 39.8 | growth |
| unloaded | private | no | 780 | 2121.4 | 2229.9 | 2155.4 | 2062.7 | 2277.5 | 85.6 | 68.5 | 20.6 | 56.0 | 43.1 | retention |
| unloaded | rss_anon | yes | 780 | 1683.6 | 1773.2 | 1706.4 | 1626.7 | 1820.7 | 104.6 | 83.7 | 30.2 | 64.4 | 34.1 | growth |
| loaded | managed_heap | yes | 780 | 281.3 | 285.7 | 282.3 | 274.6 | 301.5 | 3.6 | 2.9 | 1.3 | 0.9 | 32.0 | plateau |
| loaded | working_set | yes | 780 | 2096.6 | 2126.7 | 2103.6 | 1987.8 | 2207.2 | 103.7 | 82.9 | 15.3 | 78.0 | 42.1 | growth |
| loaded | private | no | 780 | 4245.4 | 4293.9 | 4268.0 | 4142.2 | 4375.7 | 88.0 | 70.4 | 5.1 | 70.9 | 85.4 | plateau |
| loaded | rss_anon | yes | 780 | 1809.4 | 1839.2 | 1816.0 | 1700.5 | 1919.7 | 103.0 | 82.4 | 14.9 | 78.0 | 36.3 | growth |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 7763 64-Core Processor |
| logical_cpus | 32 |
| ram_total_mb | 1031813 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.8.0-111-generic |
| runtime | .NET 10.0.0 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 4090, 580.159.03, 24564 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-video --cuda --duration 1h --out /workspace/out/starling-q-w3/soak-embeddinggemma2-video --note G4 one-hour qualification soak, cuda, RunPod starling-q-w3, NVIDIA GeForce RTX 4090, 580.159.03, CPU AMD EPYC 7763 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time, CUDA libraries from the pinned wheels (eg2 requirements-cuda) |
| machine_note | G4 one-hour qualification soak, cuda, RunPod starling-q-w3, NVIDIA GeForce RTX 4090, 580.159.03, CPU AMD EPYC 7763 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time, CUDA libraries from the pinned wheels (eg2 requirements-cuda) |
| load before (loadavg, MemAvailable MB) | [3.29,23.27,32.57], 896392 |
| load after (loadavg, MemAvailable MB) | [8.54,28.16,39.84], 950653 |
