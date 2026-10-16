# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 909 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 49086; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1310, 1806 |
| Unload ms (median, max) | 105.3, 147.4 |
| Request p50 ms (median over cycles) | 46.2 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 5454 | 0 | 301.99 | 386.20 | 525.24 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 731 | 288.5 | 292.9 | 288.6 | 224.3 | 292.9 | 2.2 | 1.8 | -10.1 | 4.8 | 32.0 | plateau |
| unloaded | working_set | yes | 731 | 399.0 | 402.6 | 404.8 | 393.6 | 438.0 | -0.1 | -0.1 | 42.6 | -15.1 | 32.0 | plateau |
| unloaded | private | no | 731 | 1640.2 | 1634.4 | 1635.2 | 1567.2 | 1684.4 | 4.2 | 3.4 | 35.9 | -13.4 | 32.7 | plateau |
| unloaded | rss_anon | yes | 731 | 324.5 | 327.5 | 330.0 | 318.6 | 363.2 | -0.9 | -0.7 | 42.1 | -15.1 | 32.0 | plateau |
| loaded | managed_heap | yes | 731 | 367.2 | 371.6 | 367.3 | 303.0 | 371.6 | 2.2 | 1.8 | -9.6 | 4.7 | 32.0 | plateau |
| loaded | working_set | yes | 731 | 1817.9 | 1814.8 | 1818.3 | 1792.5 | 1859.4 | 0.9 | 0.7 | 40.2 | -12.2 | 36.4 | plateau |
| loaded | private | no | 731 | 2176.3 | 2182.2 | 2176.1 | 2105.4 | 2240.4 | 4.8 | 3.8 | 34.5 | -13.1 | 43.5 | plateau |
| loaded | rss_anon | yes | 731 | 1743.4 | 1739.7 | 1743.4 | 1717.8 | 1784.3 | 0.1 | 0.1 | 39.7 | -12.2 | 34.9 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558 |
| load before (loadavg, MemAvailable MB) | [2.43,0.76,0.27], 14164 |
| load after (loadavg, MemAvailable MB) | [2.39,2.27,2.2], 14042 |
