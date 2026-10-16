# G4 soak: embeddinggemma-2/onnx-fp32-cuda

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 760 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 41040; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2608, 4663 |
| Unload ms (median, max) | 130.9, 602.7 |
| Request p50 ms (median over cycles) | 18.6 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 4560 | 0 | 146.04 | 211.35 | 247.02 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 609 | 223.2 | 227.7 | 224.3 | 223.2 | 260.3 | 4.2 | 3.4 | 4.8 | 3.8 | 32.0 | plateau |
| unloaded | working_set | yes | 609 | 915.4 | 917.5 | 921.8 | 909.0 | 944.9 | -12.2 | -9.7 | 3.5 | -1.8 | 32.0 | plateau |
| unloaded | private | no | 609 | 1566.8 | 1575.9 | 1577.1 | 1566.8 | 1778.8 | 14.2 | 11.3 | 12.8 | 5.9 | 32.0 | plateau |
| unloaded | rss_anon | yes | 609 | 759.3 | 759.2 | 760.4 | 757.0 | 782.2 | -2.3 | -1.8 | 6.9 | 3.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 609 | 301.9 | 306.4 | 303.0 | 298.6 | 339.0 | 4.1 | 3.3 | 4.8 | 3.7 | 32.0 | plateau |
| loaded | working_set | yes | 609 | 1562.1 | 1549.4 | 1524.8 | 1308.6 | 1606.4 | -12.4 | -9.9 | 5.8 | 4.5 | 32.0 | plateau |
| loaded | private | no | 609 | 2132.4 | 2098.2 | 2091.6 | 1862.9 | 2193.9 | 11.4 | 9.1 | 17.5 | 14.0 | 41.8 | plateau |
| loaded | rss_anon | yes | 609 | 1404.0 | 1389.1 | 1360.3 | 1154.0 | 1441.7 | -2.5 | -2.0 | 9.3 | 9.9 | 32.0 | plateau |

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
| command_line | embeddinggemma2 --cuda --duration 1h --out /root/bout/starling-q-b1/soak-embeddinggemma2 --note G4 one-hour binding CUDA soak 2026-10-16 at e404702fa0fdbf91fd506757f1a5c15522d023dc, RunPod starling-q-b1, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time on the pod, THP madvise, CUDA libraries from the pinned wheels (eg2 requirements-cuda), TF32 off |
| machine_note | G4 one-hour binding CUDA soak 2026-10-16 at e404702fa0fdbf91fd506757f1a5c15522d023dc, RunPod starling-q-b1, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time on the pod, THP madvise, CUDA libraries from the pinned wheels (eg2 requirements-cuda), TF32 off |
| load before (loadavg, MemAvailable MB) | [18.34,24.81,30.94], 932303 |
| load after (loadavg, MemAvailable MB) | [14.86,18.46,15.71], 919095 |
