# G4 soak: embeddinggemma-2/image-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 600 s / 605 s |
| Load/unload cycles | 136 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 544; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3484, 6285 |
| Unload ms (median, max) | 172.7, 504.6 |
| Request p50 ms (median over cycles) | 181.6 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 136 | 0 | 32.89 | 63.67 | 108.02 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 110 | 202.2 | 212.4 | 222.1 | 191.9 | 243.1 | 4.8 | 0.6 | 2.7 | -16.3 | 32.0 | plateau |
| unloaded | working_set | yes | 110 | 1132.2 | 1132.3 | 1136.4 | 1064.9 | 1152.5 | 137.8 | 18.4 | 9.7 | -8.2 | 32.0 | plateau |
| unloaded | private | no | 110 | 2138.4 | 2213.1 | 2194.6 | 2067.6 | 2252.8 | 959.1 | 128.1 | 73.6 | 0.9 | 43.9 | plateau |
| unloaded | rss_anon | yes | 110 | 846.7 | 845.0 | 850.2 | 778.3 | 865.4 | 128.5 | 17.2 | 8.5 | -8.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 110 | 281.0 | 291.1 | 300.8 | 271.5 | 321.8 | 0.6 | 0.1 | 2.8 | -16.3 | 32.0 | plateau |
| loaded | working_set | yes | 110 | 2080.4 | 2021.6 | 2052.9 | 1966.2 | 2140.0 | 75.9 | 10.1 | 16.7 | 0.6 | 41.1 | plateau |
| loaded | private | no | 110 | 2811.1 | 2805.3 | 2828.7 | 2713.0 | 2908.1 | 612.7 | 81.8 | 66.2 | 9.2 | 56.6 | plateau |
| loaded | rss_anon | yes | 110 | 1792.9 | 1732.3 | 1764.1 | 1677.6 | 1850.9 | 66.3 | 8.9 | 15.5 | 0.5 | 35.3 | plateau |

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
| command_line | embeddinggemma2-image --cuda --duration 10m --out /root/ci-out/soak/embeddinggemma2-image --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 4090, commit 9a18b019a4d8, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 4090, commit 9a18b019a4d8, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| load before (loadavg, MemAvailable MB) | [54.98,62.62,51.78], 925594 |
| load after (loadavg, MemAvailable MB) | [83.83,73.68,63.35], 929765 |
