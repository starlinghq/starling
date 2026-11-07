# G4 soak: embeddinggemma-2/onnx-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 600 s / 605 s |
| Load/unload cycles | 87 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 4698; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 4241, 8422 |
| Unload ms (median, max) | 137.0, 253.4 |
| Request p50 ms (median over cycles) | 31.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 522 | 0 | 149.54 | 213.19 | 238.29 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 73 | 287.8 | 320.4 | 288.3 | 255.2 | 353.1 | -14.8 | -2.0 | 3.3 | -35.6 | 32.0 | plateau |
| unloaded | working_set | yes | 73 | 1192.7 | 1217.6 | 1210.9 | 1192.2 | 1232.6 | 50.1 | 6.6 | 2.6 | -13.8 | 32.0 | plateau |
| unloaded | private | no | 73 | 1748.8 | 1769.2 | 1781.3 | 1726.4 | 1925.0 | -713.5 | -94.5 | 98.3 | -72.0 | 35.6 | plateau |
| unloaded | rss_anon | yes | 73 | 837.6 | 860.5 | 854.6 | 836.6 | 876.0 | 38.4 | 5.1 | 1.3 | -14.4 | 32.0 | plateau |
| unloaded | gpu | yes | 73 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 73 | 366.5 | 399.2 | 367.1 | 333.9 | 431.8 | -4.3 | -0.6 | 3.4 | -35.5 | 32.0 | plateau |
| loaded | working_set | yes | 73 | 1874.2 | 1898.1 | 1904.7 | 1859.2 | 1941.9 | 46.6 | 6.2 | 1.6 | -18.5 | 38.1 | plateau |
| loaded | private | no | 73 | 1981.4 | 2049.8 | 2040.4 | 1981.4 | 2119.3 | 58.8 | 7.8 | 8.7 | -44.9 | 40.8 | plateau |
| loaded | rss_anon | yes | 73 | 1517.2 | 1539.0 | 1546.1 | 1500.2 | 1583.3 | 34.4 | 4.6 | 0.2 | -19.1 | 32.0 | plateau |
| loaded | gpu | yes | 73 | 3384.0 | 3382.0 | 3386.0 | 3382.0 | 3640.0 | -204.8 | -27.1 | 82.2 | -33.1 | 67.7 | plateau |

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
| command_line | embeddinggemma2 --cuda --duration 10m --out <local>/out-0.5.0/soak/embeddinggemma2 --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit ce5225854fa7, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit ce5225854fa7, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| load before (loadavg, MemAvailable MB) | [55.97,27.44,23.02], 54281 |
| load after (loadavg, MemAvailable MB) | [38.23,23.37,21.96], 62641 |
