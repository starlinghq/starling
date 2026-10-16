# G4 soak: embeddinggemma-2/video-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 600 s / 605 s |
| Load/unload cycles | 92 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 276; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 5868, 7579 |
| Unload ms (median, max) | 82.7, 470.3 |
| Request p50 ms (median over cycles) | 142.5 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 92 | 0 | 32.38 | 74.84 | 96.08 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 74 | 212.0 | 223.2 | 212.0 | 205.4 | 230.4 | -14.8 | -2.0 | -1.2 | -6.4 | 32.0 | plateau |
| unloaded | working_set | yes | 74 | 1000.3 | 999.4 | 998.3 | 986.5 | 1026.9 | -190.9 | -25.4 | -3.0 | -15.6 | 32.0 | plateau |
| unloaded | private | no | 74 | 1443.3 | 1455.4 | 1423.6 | 1383.9 | 1491.9 | 41.3 | 5.5 | 6.1 | -18.1 | 32.0 | plateau |
| unloaded | rss_anon | yes | 74 | 714.6 | 711.9 | 711.3 | 699.1 | 740.3 | -199.6 | -26.5 | -3.9 | -15.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 74 | 290.7 | 301.9 | 290.7 | 284.1 | 309.1 | -14.7 | -1.9 | -1.4 | -6.4 | 32.0 | plateau |
| loaded | working_set | yes | 74 | 1184.8 | 1152.3 | 1172.1 | 1115.8 | 1213.0 | -207.6 | -27.6 | -5.6 | -1.9 | 32.0 | plateau |
| loaded | private | no | 74 | 2068.4 | 2072.2 | 2042.1 | 1982.5 | 2122.0 | 40.0 | 5.3 | 1.3 | 0.3 | 40.8 | plateau |
| loaded | rss_anon | yes | 74 | 897.2 | 862.9 | 882.7 | 826.3 | 924.4 | -216.6 | -28.8 | -6.7 | -2.1 | 32.0 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-video --cuda --duration 10m --out /root/ci-out/soak/embeddinggemma2-video --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 4090, commit 9a18b019a4d8, CUDA libraries from the pinned eg2 requirements-cuda wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 4090, commit 9a18b019a4d8, CUDA libraries from the pinned eg2 requirements-cuda wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| load before (loadavg, MemAvailable MB) | [81.68,73.4,63.32], 929577 |
| load after (loadavg, MemAvailable MB) | [65.08,61.95,60.79], 911257 |
