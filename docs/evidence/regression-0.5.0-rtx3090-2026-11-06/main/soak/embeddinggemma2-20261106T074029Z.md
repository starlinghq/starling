# G4 soak: embeddinggemma-2/onnx-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 600 s / 602 s |
| Load/unload cycles | 140 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 7560; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2423, 3303 |
| Unload ms (median, max) | 89.6, 102.1 |
| Request p50 ms (median over cycles) | 16.2 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 840 | 0 | 144.79 | 212.18 | 235.31 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 113 | 287.8 | 255.2 | 255.7 | 255.2 | 288.3 | -13.7 | -1.8 | 10.9 | 7.4 | 32.0 | plateau |
| unloaded | working_set | yes | 113 | 1185.1 | 1176.3 | 1176.3 | 1155.3 | 1225.5 | 69.5 | 9.3 | 17.5 | 2.9 | 32.0 | plateau |
| unloaded | private | no | 113 | 1728.8 | 1703.0 | 1718.0 | 1685.2 | 1851.5 | 65.5 | 8.7 | 8.9 | -18.9 | 34.4 | plateau |
| unloaded | rss_anon | yes | 113 | 829.1 | 818.8 | 818.8 | 798.6 | 867.9 | 62.6 | 8.3 | 16.5 | 2.7 | 32.0 | plateau |
| unloaded | gpu | yes | 113 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 113 | 366.5 | 334.0 | 334.4 | 333.9 | 367.1 | -13.8 | -1.8 | 10.9 | 7.4 | 32.0 | plateau |
| loaded | working_set | yes | 113 | 1905.2 | 1895.8 | 1882.8 | 1844.7 | 1945.9 | 63.1 | 8.4 | 13.3 | 13.4 | 37.7 | plateau |
| loaded | private | no | 113 | 2008.4 | 1982.7 | 1992.7 | 1944.4 | 2062.0 | 42.5 | 5.7 | 18.8 | 11.6 | 39.9 | plateau |
| loaded | rss_anon | yes | 113 | 1547.2 | 1536.2 | 1523.4 | 1485.2 | 1586.4 | 55.5 | 7.4 | 12.2 | 13.3 | 32.0 | plateau |
| loaded | gpu | yes | 113 | 3638.0 | 3638.0 | 3386.0 | 3382.0 | 3640.0 | -40.3 | -5.4 | 38.5 | 44.9 | 67.7 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz |
| logical_cpus | 12 |
| ram_total_mb | 96442 |
| os | Omarchy |
| kernel | 7.2.8-9-omarchy |
| runtime | .NET 10.0.14 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 615.78.08, 24576 MiB; 1, NVIDIA GeForce RTX 4060 Ti, 615.78.08, 16380 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2 --cuda --duration 10m --out <local>/out-main/soak/embeddinggemma2 --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit 3a021dc0b7f8, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit 3a021dc0b7f8, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| load before (loadavg, MemAvailable MB) | [2.13,1.74,3.41], 75166 |
| load after (loadavg, MemAvailable MB) | [1.86,1.74,2.59], 73646 |
