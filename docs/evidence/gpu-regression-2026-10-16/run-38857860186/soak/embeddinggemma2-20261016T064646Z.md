# G4 soak: embeddinggemma-2/onnx-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 600 s / 602 s |
| Load/unload cycles | 138 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 7452; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2819, 5894 |
| Unload ms (median, max) | 104.7, 580.4 |
| Request p50 ms (median over cycles) | 16.3 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 703 | 125 | 59.24 | 114.82 | 212.22 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 111 | 223.1 | 223.2 | 223.3 | 223.1 | 223.8 | 1.9 | 0.2 | 0.0 | 0.1 | 32.0 | plateau |
| unloaded | working_set | yes | 111 | 1044.4 | 1054.2 | 1050.5 | 1044.4 | 1058.0 | 51.8 | 6.9 | 4.9 | 4.4 | 32.0 | plateau |
| unloaded | private | no | 111 | 1485.0 | 1507.8 | 1505.0 | 1472.0 | 1545.1 | 227.7 | 30.2 | 32.2 | 6.1 | 32.0 | plateau |
| unloaded | rss_anon | yes | 111 | 760.2 | 769.0 | 765.3 | 760.2 | 773.2 | 46.1 | 6.1 | 4.0 | 4.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 111 | 301.9 | 301.9 | 302.0 | 298.9 | 302.5 | 0.8 | 0.1 | 0.0 | 0.1 | 32.0 | plateau |
| loaded | working_set | yes | 111 | 1642.4 | 1601.3 | 1652.8 | 1590.4 | 1713.7 | -27.7 | -3.7 | -9.0 | -6.5 | 33.1 | plateau |
| loaded | private | no | 111 | 2056.7 | 2004.9 | 2078.0 | 1993.9 | 2148.6 | 153.1 | 20.3 | 19.6 | -6.0 | 41.6 | plateau |
| loaded | rss_anon | yes | 111 | 1356.6 | 1314.0 | 1365.6 | 1303.2 | 1426.5 | -34.1 | -4.5 | -10.0 | -6.6 | 32.0 | plateau |

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
| command_line | embeddinggemma2 --cuda --duration 10m --out /root/ci-out/soak/embeddinggemma2 --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 4090, commit 9a18b019a4d8, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 4090, commit 9a18b019a4d8, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| load before (loadavg, MemAvailable MB) | [71.48,45.42,38.31], 919317 |
| load after (loadavg, MemAvailable MB) | [55.68,62.88,51.8], 923997 |
