# G4 soak: embeddinggemma-2/image-fp32-cuda

> **Short development run (720 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 720 s / 725 s |
| Load/unload cycles | 161 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 644; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3407, 6769 |
| Unload ms (median, max) | 99.1, 158.7 |
| Request p50 ms (median over cycles) | 265.8 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 161 | 0 | 6.50 | 57.80 | 64.23 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 130 | 180.5 | 180.4 | 180.6 | 180.3 | 190.5 | -6.0 | -1.0 | -1.1 | 1.4 | 32.0 | plateau |
| unloaded | working_set | yes | 130 | 1802.6 | 1802.0 | 1787.3 | 1752.5 | 1852.1 | 144.6 | 23.2 | 26.7 | 2.3 | 35.7 | plateau |
| unloaded | private | no | 130 | 1970.3 | 1993.6 | 1977.8 | 1929.0 | 2045.9 | 307.0 | 49.2 | 48.8 | 12.1 | 39.6 | retention |
| unloaded | rss_anon | yes | 130 | 1517.7 | 1516.4 | 1501.8 | 1466.9 | 1566.5 | 140.6 | 22.5 | 26.0 | 2.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 130 | 259.2 | 259.1 | 259.3 | 259.0 | 269.2 | -6.2 | -1.0 | -1.1 | 1.3 | 32.0 | plateau |
| loaded | working_set | yes | 130 | 1901.1 | 1932.5 | 1933.5 | 1866.2 | 2009.1 | -21.6 | -3.5 | 6.0 | -19.2 | 38.7 | plateau |
| loaded | private | no | 130 | 2530.4 | 2585.7 | 2596.8 | 2505.4 | 2679.2 | 166.7 | 26.7 | 30.5 | -6.9 | 51.9 | plateau |
| loaded | rss_anon | yes | 130 | 1614.2 | 1644.9 | 1646.0 | 1578.6 | 1721.7 | -25.6 | -4.1 | 5.2 | -19.4 | 32.9 | plateau |
| trimmed | managed_heap | yes | 130 | 180.6 | 180.5 | 180.7 | 180.4 | 190.6 | -6.1 | -1.0 | -1.1 | 1.4 | 32.0 | plateau |
| trimmed | working_set | yes | 130 | 1050.8 | 1043.5 | 1058.6 | 1042.8 | 1063.5 | 8.9 | 1.4 | 11.9 | -19.7 | 32.0 | plateau |
| trimmed | private | no | 130 | 1931.2 | 1945.5 | 1967.7 | 1907.2 | 2012.7 | 183.9 | 29.4 | 49.6 | -15.5 | 39.4 | plateau |
| trimmed | rss_anon | yes | 130 | 765.9 | 757.9 | 773.1 | 757.1 | 777.9 | 4.8 | 0.8 | 11.2 | -19.8 | 32.0 | plateau |

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
| command_line | embeddinggemma2-image --cuda --duration 12m --malloc-trim --out /workspace/out/starling-q-c3-fix/f-malloc-trim --note CUDA soak growth diagnostic f-malloc-trim, RunPod starling-q-c3-fix, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, CUDA libraries from the pinned wheels |
| machine_note | CUDA soak growth diagnostic f-malloc-trim, RunPod starling-q-c3-fix, NVIDIA GeForce RTX 3090, 580.126.20, CPU AMD EPYC 7H12 64-Core Processor, CUDA libraries from the pinned wheels |
| load before (loadavg, MemAvailable MB) | [38.44,42.75,46.88], 950750 |
| load after (loadavg, MemAvailable MB) | [56.23,47.07,46.69], 946470 |
