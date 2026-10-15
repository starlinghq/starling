# G4 soak: embeddinggemma-2/image-fp32-cuda

> **Short development run (720 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 720 s / 721 s |
| Load/unload cycles | 1025 |
| Keep loaded (diagnostic) | yes: one load, no unloads; the `unloaded` series is a second sample with the model loaded |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 4100; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 0, 7228 |
| Unload ms (median, max) | 0.0, 0.0 |
| Request p50 ms (median over cycles) | 229.0 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 1025 | 0 | 5.60 | 56.14 | 65.05 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 830 | 102.2 | 102.0 | 102.7 | 101.6 | 113.6 | 17.1 | 2.7 | 0.5 | -0.8 | 32.0 | plateau |
| unloaded | working_set | yes | 830 | 1381.1 | 1380.3 | 1380.3 | 1361.3 | 1400.1 | 32.4 | 5.2 | -16.4 | 4.1 | 32.0 | plateau |
| unloaded | private | no | 830 | 1990.7 | 1989.5 | 1989.6 | 1969.3 | 2017.3 | 25.0 | 4.0 | -17.3 | 4.3 | 39.8 | plateau |
| unloaded | rss_anon | yes | 830 | 1095.1 | 1093.8 | 1093.8 | 1075.2 | 1113.9 | 29.2 | 4.7 | -16.8 | 3.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 830 | 102.2 | 102.0 | 102.7 | 101.6 | 113.6 | 17.1 | 2.7 | 0.5 | -0.8 | 32.0 | plateau |
| loaded | working_set | yes | 830 | 1381.1 | 1380.3 | 1380.3 | 1361.3 | 1400.1 | 32.5 | 5.2 | -16.4 | 4.0 | 32.0 | plateau |
| loaded | private | no | 830 | 1990.7 | 1989.5 | 1989.6 | 1969.3 | 2017.3 | 25.1 | 4.0 | -17.3 | 4.3 | 39.8 | plateau |
| loaded | rss_anon | yes | 830 | 1095.1 | 1093.8 | 1093.8 | 1075.2 | 1113.9 | 29.4 | 4.7 | -16.8 | 3.8 | 32.0 | plateau |

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
| command_line | embeddinggemma2-image --cuda --duration 12m --keep-loaded --out /workspace/out/starling-q-c3-fix/d-requests-only --note CUDA soak growth diagnostic d-requests-only, RunPod starling-q-c3-fix, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, CUDA libraries from the pinned wheels |
| machine_note | CUDA soak growth diagnostic d-requests-only, RunPod starling-q-c3-fix, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, CUDA libraries from the pinned wheels |
| load before (loadavg, MemAvailable MB) | [43.23,46.62,51.42], 948529 |
| load after (loadavg, MemAvailable MB) | [36.2,42.63,46.94], 949321 |
