# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 221 MB; loaded private (address space, report-only) grows 240 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3620 s |
| Load/unload cycles | 88 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 352; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3120, 4047 |
| Unload ms (median, max) | 133.4, 143.9 |
| Request p50 ms (median over cycles) | 17976.5 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 88 | 0 | 15.52 | 57.48 | 104.41 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 73 | 182.0 | 182.2 | 181.6 | 180.1 | 183.1 | 0.4 | 0.3 | 0.3 | 0.3 | 32.0 | plateau |
| unloaded | working_set | yes | 73 | 395.9 | 441.1 | 407.1 | 371.2 | 498.1 | 26.9 | 21.5 | 29.1 | -41.4 | 32.0 | plateau |
| unloaded | private | no | 73 | 2396.4 | 2625.1 | 2526.0 | 2371.8 | 2655.5 | 277.3 | 221.3 | 126.0 | 88.1 | 50.5 | growth |
| unloaded | rss_anon | yes | 73 | 321.8 | 365.0 | 332.1 | 295.2 | 422.1 | 24.7 | 19.7 | 27.9 | -42.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 73 | 260.5 | 260.9 | 260.3 | 258.8 | 261.4 | 0.4 | 0.3 | 0.2 | 0.3 | 32.0 | plateau |
| loaded | working_set | yes | 73 | 2089.1 | 2134.6 | 2113.3 | 2071.4 | 2194.5 | 35.8 | 28.5 | 38.2 | -42.0 | 42.3 | plateau |
| loaded | private | no | 73 | 2972.4 | 3217.1 | 3121.0 | 2972.4 | 3251.7 | 300.4 | 239.7 | 143.1 | 101.6 | 62.4 | growth |
| loaded | rss_anon | yes | 73 | 2015.0 | 2058.5 | 2037.2 | 1995.4 | 2119.0 | 33.5 | 26.7 | 36.8 | -42.6 | 40.7 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 7763 64-Core Processor |
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
| git_commit | d5798090b13dddada74fe920a81bd346f7dfd974 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [1.61,0.53,0.18], 14085 |
| load after (loadavg, MemAvailable MB) | [6.05,6.24,6.21], 14236 |
