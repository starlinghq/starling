# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 128 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 384; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3203, 4115 |
| Unload ms (median, max) | 131.0, 135.6 |
| Request p50 ms (median over cycles) | 13044.1 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 128 | 0 | 16.44 | 46.03 | 77.10 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 106 | 170.0 | 170.8 | 170.2 | 169.7 | 170.8 | 0.9 | 0.7 | 0.3 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 106 | 410.3 | 417.1 | 413.7 | 409.5 | 418.9 | 6.5 | 5.2 | 3.6 | 1.1 | 32.0 | plateau |
| unloaded | private | no | 106 | 2407.8 | 2392.0 | 2397.2 | 2375.8 | 2434.7 | -0.2 | -0.2 | -4.5 | -7.0 | 47.9 | plateau |
| unloaded | rss_anon | yes | 106 | 336.0 | 341.3 | 338.1 | 335.2 | 343.0 | 5.0 | 4.0 | 2.7 | 0.7 | 32.0 | plateau |
| loaded | managed_heap | yes | 106 | 328.1 | 329.3 | 328.5 | 327.6 | 330.2 | 1.0 | 0.8 | 0.1 | 0.5 | 32.0 | plateau |
| loaded | working_set | yes | 106 | 2175.7 | 2178.6 | 2177.6 | 2142.3 | 2182.5 | 19.7 | 15.7 | -8.5 | -3.7 | 43.6 | plateau |
| loaded | private | no | 106 | 3056.4 | 3052.6 | 3052.7 | 3002.7 | 3109.8 | 9.2 | 7.3 | -14.8 | -17.8 | 61.1 | plateau |
| loaded | rss_anon | yes | 106 | 2101.5 | 2102.8 | 2102.3 | 2067.4 | 2106.7 | 18.1 | 14.4 | -9.6 | -4.0 | 42.0 | plateau |

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
| git_commit | 6ea1bfa218555d4360a81afb48c4050f46e71824 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39023915026 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39023915026 |
| load before (loadavg, MemAvailable MB) | [3.47,0.99,0.34], 14171 |
| load after (loadavg, MemAvailable MB) | [7.06,6.76,6.51], 14446 |
