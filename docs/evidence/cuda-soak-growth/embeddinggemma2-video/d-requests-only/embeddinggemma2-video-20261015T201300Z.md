# G4 soak: embeddinggemma-2/video-fp32-cuda

> **Short development run (720 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 720 s / 721 s |
| Load/unload cycles | 2085 |
| Keep loaded (diagnostic) | yes: one load, no unloads; the `unloaded` series is a second sample with the model loaded |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 6255; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 0, 6614 |
| Unload ms (median, max) | 0.0, 0.0 |
| Request p50 ms (median over cycles) | 97.8 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 2085 | 0 | 4.94 | 24.85 | 39.56 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 1687 | 124.9 | 140.4 | 124.5 | 123.9 | 140.4 | 14.9 | 2.4 | 1.7 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 1687 | 1305.5 | 1354.5 | 1348.8 | 1305.5 | 1362.6 | 274.5 | 43.9 | 56.5 | 2.7 | 32.0 | plateau |
| unloaded | private | no | 1687 | 1925.5 | 1974.7 | 1968.5 | 1925.5 | 1982.0 | 270.9 | 43.3 | 56.2 | 2.4 | 39.4 | plateau |
| unloaded | rss_anon | yes | 1687 | 1021.5 | 1070.1 | 1064.5 | 1021.5 | 1078.1 | 272.0 | 43.5 | 56.2 | 2.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 1687 | 124.9 | 140.4 | 124.5 | 123.9 | 140.4 | 14.9 | 2.4 | 1.7 | 0.4 | 32.0 | plateau |
| loaded | working_set | yes | 1687 | 1305.5 | 1354.5 | 1348.8 | 1305.5 | 1362.6 | 274.5 | 43.9 | 56.5 | 2.7 | 32.0 | plateau |
| loaded | private | no | 1687 | 1925.5 | 1974.7 | 1968.5 | 1925.5 | 1982.0 | 270.9 | 43.3 | 56.2 | 2.4 | 39.4 | plateau |
| loaded | rss_anon | yes | 1687 | 1021.5 | 1070.1 | 1064.5 | 1021.5 | 1078.1 | 272.0 | 43.5 | 56.2 | 2.5 | 32.0 | plateau |

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
| gpu_identity | 0, NVIDIA GeForce RTX 4090, 580.126.20, 24564 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-video --cuda --duration 12m --keep-loaded --out /workspace/out/starling-q-c2-fix/d-requests-only --note CUDA soak growth diagnostic d-requests-only, RunPod starling-q-c2-fix, NVIDIA GeForce RTX 4090, 580.126.20, CPU AMD EPYC 7763 64-Core Processor, CUDA libraries from the pinned wheels |
| machine_note | CUDA soak growth diagnostic d-requests-only, RunPod starling-q-c2-fix, NVIDIA GeForce RTX 4090, 580.126.20, CPU AMD EPYC 7763 64-Core Processor, CUDA libraries from the pinned wheels |
| load before (loadavg, MemAvailable MB) | [48.49,29.06,26.13], 942923 |
| load after (loadavg, MemAvailable MB) | [18.97,21.14,24.29], 929489 |
