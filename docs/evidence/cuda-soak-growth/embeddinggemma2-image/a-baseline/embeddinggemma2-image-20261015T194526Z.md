# G4 soak: embeddinggemma-2/image-fp32-cuda

> **Short development run (720 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 720 s / 722 s |
| Load/unload cycles | 179 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 716; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3105, 6957 |
| Unload ms (median, max) | 111.4, 162.7 |
| Request p50 ms (median over cycles) | 264.3 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 179 | 0 | 6.65 | 61.55 | 64.15 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 145 | 180.5 | 180.4 | 180.6 | 180.3 | 190.5 | 2.4 | 0.4 | 0.2 | -2.8 | 32.0 | plateau |
| unloaded | working_set | yes | 145 | 1698.3 | 1808.1 | 1796.9 | 1698.3 | 1952.5 | 968.7 | 154.8 | 137.1 | -22.1 | 35.9 | plateau |
| unloaded | private | no | 145 | 1850.1 | 1937.7 | 1941.2 | 1826.1 | 2106.2 | 963.2 | 153.9 | 129.0 | -14.8 | 38.8 | plateau |
| unloaded | rss_anon | yes | 145 | 1412.6 | 1521.8 | 1510.7 | 1412.6 | 1666.1 | 965.4 | 154.2 | 136.5 | -22.1 | 32.0 | plateau |
| loaded | managed_heap | yes | 145 | 259.2 | 259.1 | 259.3 | 258.9 | 269.2 | 2.5 | 0.4 | 0.2 | -2.8 | 32.0 | plateau |
| loaded | working_set | yes | 145 | 2005.7 | 2083.2 | 2042.6 | 1907.4 | 2116.1 | 388.3 | 62.0 | 98.6 | -45.9 | 40.9 | plateau |
| loaded | private | no | 145 | 2619.1 | 2698.4 | 2650.6 | 2514.4 | 2732.1 | 376.9 | 60.2 | 93.6 | -42.1 | 53.0 | plateau |
| loaded | rss_anon | yes | 145 | 1718.0 | 1794.8 | 1754.3 | 1619.1 | 1827.8 | 384.9 | 61.5 | 97.9 | -45.9 | 35.1 | plateau |

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
| command_line | embeddinggemma2-image --cuda --duration 12m --out /workspace/out/starling-q-c3-fix/a-baseline --note CUDA soak growth diagnostic a-baseline, RunPod starling-q-c3-fix, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, CUDA libraries from the pinned wheels |
| machine_note | CUDA soak growth diagnostic a-baseline, RunPod starling-q-c3-fix, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, CUDA libraries from the pinned wheels |
| load before (loadavg, MemAvailable MB) | [51.5,54.71,60.27], 932034 |
| load after (loadavg, MemAvailable MB) | [49.35,56.8,57.94], 925430 |
