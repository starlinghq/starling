# G4 soak: embeddinggemma-2/image-fp32-cuda

> **Short development run (720 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 720 s / 724 s |
| Load/unload cycles | 125 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 500; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 4780, 8152 |
| Unload ms (median, max) | 108.9, 162.6 |
| Request p50 ms (median over cycles) | 264.8 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 125 | 0 | 4.35 | 55.51 | 63.00 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 101 | 180.5 | 180.9 | 180.5 | 180.3 | 190.2 | 4.7 | 0.8 | 2.9 | -0.9 | 32.0 | plateau |
| unloaded | working_set | yes | 101 | 974.1 | 949.6 | 974.9 | 947.4 | 986.0 | -229.1 | -36.7 | 4.0 | -33.2 | 32.0 | plateau |
| unloaded | private | no | 101 | 1246.8 | 1209.7 | 1216.2 | 1183.9 | 1255.0 | -279.1 | -44.7 | 3.7 | -28.2 | 32.0 | plateau |
| unloaded | rss_anon | yes | 101 | 689.7 | 663.9 | 689.7 | 661.8 | 700.8 | -235.0 | -37.6 | 3.0 | -33.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 101 | 259.2 | 259.6 | 259.2 | 258.8 | 268.9 | 4.8 | 0.8 | 2.8 | -0.9 | 32.0 | plateau |
| loaded | working_set | yes | 101 | 1167.3 | 1119.1 | 1174.1 | 1117.6 | 1207.7 | -262.1 | -42.0 | 17.9 | -29.9 | 32.0 | plateau |
| loaded | private | no | 101 | 1903.6 | 1842.8 | 1897.3 | 1840.7 | 1943.9 | -316.1 | -50.6 | 21.9 | -31.5 | 37.9 | plateau |
| loaded | rss_anon | yes | 101 | 880.8 | 831.4 | 886.4 | 829.9 | 920.2 | -268.0 | -42.9 | 16.9 | -30.0 | 32.0 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-image --cuda --duration 12m --out /workspace/out/starling-q-c3-fix/b-mmap-threshold-64k --note CUDA soak growth diagnostic b-mmap-threshold-64k, RunPod starling-q-c3-fix, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, CUDA libraries from the pinned wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| machine_note | CUDA soak growth diagnostic b-mmap-threshold-64k, RunPod starling-q-c3-fix, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, CUDA libraries from the pinned wheels, MALLOC_MMAP_THRESHOLD_=65536 |
| load before (loadavg, MemAvailable MB) | [46.46,55.66,57.54], 933875 |
| load after (loadavg, MemAvailable MB) | [43.72,46.87,51.58], 944916 |
