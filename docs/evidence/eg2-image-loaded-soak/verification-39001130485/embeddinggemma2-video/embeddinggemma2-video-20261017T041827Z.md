# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3609 s |
| Load/unload cycles | 129 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 387; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3146, 4126 |
| Unload ms (median, max) | 130.4, 135.1 |
| Request p50 ms (median over cycles) | 12989.4 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 129 | 0 | 16.77 | 54.44 | 73.06 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 107 | 170.0 | 170.8 | 170.2 | 169.7 | 170.8 | 1.0 | 0.8 | 0.3 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 107 | 424.4 | 431.4 | 429.7 | 423.4 | 437.6 | 11.3 | 9.0 | 5.8 | 1.7 | 32.0 | plateau |
| unloaded | private | no | 107 | 2401.7 | 2390.9 | 2395.4 | 2369.6 | 2450.0 | 24.8 | 19.8 | 8.7 | 11.0 | 47.9 | plateau |
| unloaded | rss_anon | yes | 107 | 350.0 | 355.4 | 354.0 | 349.0 | 361.6 | 9.8 | 7.8 | 4.9 | 1.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 107 | 328.7 | 329.0 | 328.5 | 327.6 | 330.5 | 1.0 | 0.8 | 0.2 | 0.6 | 32.0 | plateau |
| loaded | working_set | yes | 107 | 2188.8 | 2196.3 | 2193.9 | 2157.3 | 2198.8 | 32.2 | 25.7 | 18.9 | 9.8 | 43.9 | plateau |
| loaded | private | no | 107 | 3049.8 | 3057.3 | 3056.5 | 2997.3 | 3114.0 | 51.6 | 41.2 | 27.3 | 27.2 | 61.1 | plateau |
| loaded | rss_anon | yes | 107 | 2114.5 | 2120.4 | 2118.2 | 2082.0 | 2122.9 | 30.5 | 24.4 | 17.7 | 9.5 | 42.4 | plateau |

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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485 |
| load before (loadavg, MemAvailable MB) | [3.44,0.95,0.33], 14081 |
| load after (loadavg, MemAvailable MB) | [6.84,6.39,6.2], 14408 |
