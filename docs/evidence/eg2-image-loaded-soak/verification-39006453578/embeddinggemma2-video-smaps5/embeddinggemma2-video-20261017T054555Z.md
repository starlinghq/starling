# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3606 s |
| Load/unload cycles | 126 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 378; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3273, 4589 |
| Unload ms (median, max) | 135.3, 141.5 |
| Request p50 ms (median over cycles) | 13153.9 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 126 | 0 | 15.95 | 49.23 | 100.15 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 105 | 170.5 | 171.3 | 170.7 | 169.7 | 172.3 | 1.3 | 1.1 | 0.5 | 0.8 | 32.0 | plateau |
| unloaded | working_set | yes | 105 | 367.9 | 455.4 | 452.1 | 367.7 | 506.7 | 36.6 | 29.2 | 42.5 | -2.5 | 32.0 | plateau |
| unloaded | private | no | 105 | 2396.5 | 2482.3 | 2467.9 | 2367.8 | 2545.4 | 57.7 | 45.9 | 40.2 | 6.9 | 49.4 | plateau |
| unloaded | rss_anon | yes | 105 | 293.5 | 379.3 | 376.3 | 292.6 | 431.3 | 35.0 | 27.9 | 41.5 | -2.8 | 32.0 | plateau |
| loaded | managed_heap | yes | 105 | 328.4 | 329.3 | 328.8 | 327.6 | 332.0 | 1.4 | 1.1 | 0.8 | 0.5 | 32.0 | plateau |
| loaded | working_set | yes | 105 | 2202.0 | 2217.8 | 2213.2 | 2177.5 | 2233.0 | 8.0 | 6.4 | 4.6 | 3.3 | 44.3 | plateau |
| loaded | private | no | 105 | 3120.4 | 3128.0 | 3129.1 | 3079.4 | 3171.2 | 24.9 | 19.8 | 8.3 | 1.5 | 62.6 | plateau |
| loaded | rss_anon | yes | 105 | 2127.4 | 2141.8 | 2137.5 | 2102.1 | 2157.3 | 6.3 | 5.0 | 3.4 | 2.9 | 42.8 | plateau |

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
| git_commit | 03570797591f1b66f32804bcdb8207aa58feb119 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [1.95,0.71,0.26], 14180 |
| load after (loadavg, MemAvailable MB) | [5.89,6.1,6.06], 14417 |
