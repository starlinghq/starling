# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3612 s |
| Load/unload cycles | 129 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 387; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3115, 4114 |
| Unload ms (median, max) | 130.2, 133.6 |
| Request p50 ms (median over cycles) | 13018.9 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 129 | 0 | 15.29 | 50.96 | 61.74 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 107 | 170.0 | 170.8 | 170.2 | 169.7 | 170.8 | 1.0 | 0.8 | 0.3 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 107 | 419.1 | 425.2 | 424.1 | 418.8 | 432.3 | 9.1 | 7.3 | 5.8 | 2.1 | 32.0 | plateau |
| unloaded | private | no | 107 | 2400.4 | 2391.5 | 2405.3 | 2373.8 | 2446.2 | -3.6 | -2.9 | 4.5 | 8.1 | 48.1 | plateau |
| unloaded | rss_anon | yes | 107 | 344.7 | 349.3 | 348.4 | 344.3 | 356.4 | 7.6 | 6.1 | 4.9 | 1.7 | 32.0 | plateau |
| loaded | managed_heap | yes | 107 | 328.5 | 330.2 | 328.5 | 327.6 | 330.2 | 0.9 | 0.8 | 0.3 | 0.3 | 32.0 | plateau |
| loaded | working_set | yes | 107 | 2153.8 | 2191.6 | 2189.0 | 2153.2 | 2194.5 | 26.0 | 20.8 | 12.8 | 11.2 | 43.8 | plateau |
| loaded | private | no | 107 | 3018.5 | 3068.4 | 3068.8 | 3014.9 | 3104.0 | 14.2 | 11.3 | 12.2 | 9.1 | 61.4 | plateau |
| loaded | rss_anon | yes | 107 | 2079.5 | 2115.6 | 2113.6 | 2078.2 | 2118.6 | 24.3 | 19.4 | 11.6 | 10.8 | 42.3 | plateau |

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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578 |
| load before (loadavg, MemAvailable MB) | [2.91,0.96,0.34], 14182 |
| load after (loadavg, MemAvailable MB) | [6.31,6.28,6.2], 14437 |
