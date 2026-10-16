# G4 soak: embeddinggemma-2/onnx-fp32-cuda

**Verdict:** INCOMPLETE (checkpoint at 1231 s of 3600 s; the run did not finish): PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 1231 s |
| Load/unload cycles | 262 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 14148; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2583, 4675 |
| Unload ms (median, max) | 122.7, 197.5 |
| Request p50 ms (median over cycles) | 20.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 1572 | 0 | 143.31 | 211.56 | 242.25 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 210 | 223.2 | 223.3 | 223.5 | 223.2 | 224.3 | 2.2 | 0.6 | 0.2 | 0.1 | 32.0 | plateau |
| unloaded | working_set | yes | 210 | 1042.0 | 943.3 | 964.4 | 936.8 | 1065.6 | -624.6 | -170.6 | -15.4 | 4.8 | 32.0 | plateau |
| unloaded | private | no | 210 | 1473.2 | 1511.2 | 1481.5 | 1457.4 | 1762.0 | 99.4 | 27.2 | 9.6 | 13.2 | 32.0 | plateau |
| unloaded | rss_anon | yes | 210 | 754.1 | 777.6 | 773.4 | 754.0 | 781.0 | 40.0 | 10.9 | 9.0 | 2.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 210 | 301.9 | 302.0 | 302.2 | 298.4 | 303.1 | 2.1 | 0.6 | 0.1 | 0.2 | 32.0 | plateau |
| loaded | working_set | yes | 210 | 1703.7 | 1541.0 | 1601.1 | 1421.7 | 1723.7 | -634.3 | -173.3 | -35.0 | 3.1 | 32.0 | plateau |
| loaded | private | no | 210 | 2115.8 | 2089.9 | 2089.9 | 1967.6 | 2155.2 | 51.6 | 14.1 | -7.9 | 0.1 | 41.8 | plateau |
| loaded | rss_anon | yes | 210 | 1413.8 | 1373.3 | 1374.8 | 1254.6 | 1439.4 | 30.3 | 8.3 | -10.7 | 1.1 | 32.0 | plateau |

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
| command_line | embeddinggemma2 --cuda --duration 1h --out /workspace/b/out/starling-q-b1/soak-embeddinggemma2 --note G4 one-hour binding CUDA soak 2026-10-16 at e404702fa0fdbf91fd506757f1a5c15522d023dc, RunPod starling-q-b1, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time on the pod, THP madvise, CUDA libraries from the pinned wheels (eg2 requirements-cuda), TF32 off |
| machine_note | G4 one-hour binding CUDA soak 2026-10-16 at e404702fa0fdbf91fd506757f1a5c15522d023dc, RunPod starling-q-b1, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, EU-CZ-1 secure cloud, files on network volume (page cache prewarmed), one job at a time on the pod, THP madvise, CUDA libraries from the pinned wheels (eg2 requirements-cuda), TF32 off |
| load before (loadavg, MemAvailable MB) | [21.67,54.21,58.16], 927732 |
| load after (loadavg, MemAvailable MB) | [33.75,28.11,32.13], 926044 |
