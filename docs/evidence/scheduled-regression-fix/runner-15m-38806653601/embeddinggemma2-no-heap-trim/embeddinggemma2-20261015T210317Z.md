# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 903 s |
| Load/unload cycles | 335 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 18090; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 572, 905 |
| Unload ms (median, max) | 10.0, 16.8 |
| Request p50 ms (median over cycles) | 40.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 2010 | 0 | 209.41 | 289.57 | 345.19 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 267 | 288.4 | 288.5 | 288.4 | 224.2 | 321.1 | -6.4 | -1.3 | -9.6 | 2.1 | 32.0 | plateau |
| unloaded | working_set | yes | 267 | 1558.7 | 1598.3 | 1571.8 | 1520.1 | 1620.3 | 304.1 | 60.8 | 30.3 | 5.8 | 32.0 | plateau |
| unloaded | private | no | 267 | 1866.2 | 1928.0 | 1874.2 | 1838.1 | 1953.7 | 273.6 | 54.7 | 17.7 | 19.7 | 37.5 | retention |
| unloaded | rss_anon | yes | 267 | 1485.0 | 1523.7 | 1497.2 | 1446.1 | 1545.7 | 300.4 | 60.1 | 29.9 | 5.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 267 | 367.1 | 367.3 | 367.2 | 302.9 | 399.9 | -6.4 | -1.3 | -9.6 | 2.1 | 32.0 | plateau |
| loaded | working_set | yes | 267 | 2344.8 | 2384.7 | 2361.6 | 2306.2 | 2406.4 | 305.8 | 61.2 | 28.5 | 8.8 | 47.2 | plateau |
| loaded | private | no | 267 | 2890.2 | 2954.0 | 2898.5 | 2862.1 | 2977.7 | 279.4 | 55.9 | 18.7 | 19.4 | 58.0 | plateau |
| loaded | rss_anon | yes | 267 | 2271.1 | 2310.1 | 2287.0 | 2232.2 | 2331.8 | 302.0 | 60.4 | 28.2 | 8.4 | 45.7 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V45 96-Core Processor |
| logical_cpus | 4 |
| ram_total_mb | 15990 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.17.0-1022-azure |
| runtime | .NET 10.0.13 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI/0.17.1 |
| gpu_identity | n/a |
| cpu_governor | n/a |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"true","DOTNET_MULTILEVEL_LOOKUP":"0","DOTNET_NOLOGO":"true","DOTNET_ROOT_X64":"/usr/share/dotnet","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1"} |
| git_commit | ff58b212568b4d484c6f849b75b0094a2e597636 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601, variant no-heap-trim --no-heap-trim |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601, variant no-heap-trim |
| load before (loadavg, MemAvailable MB) | [2.59,1.04,0.39], 14309 |
| load after (loadavg, MemAvailable MB) | [2.81,2.54,1.78], 13195 |
