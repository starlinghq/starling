# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3605 s |
| Load/unload cycles | 215 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 645; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2781, 3395 |
| Unload ms (median, max) | 67.3, 86.3 |
| Request p50 ms (median over cycles) | 7124.9 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 215 | 0 | 8.40 | 64.52 | 78.45 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 176 | 169.7 | 170.9 | 169.9 | 169.7 | 170.9 | 0.7 | 0.6 | 0.4 | 0.1 | 32.0 | plateau |
| unloaded | working_set | yes | 176 | 419.4 | 437.1 | 422.6 | 414.9 | 443.3 | 13.2 | 10.5 | 3.4 | 13.0 | 32.0 | plateau |
| unloaded | private | no | 176 | 2424.9 | 2497.5 | 2424.8 | 2400.8 | 2497.5 | 44.6 | 35.6 | 1.0 | 54.5 | 48.5 | plateau |
| unloaded | rss_anon | yes | 176 | 344.5 | 361.1 | 346.6 | 339.8 | 367.3 | 12.1 | 9.6 | 2.7 | 13.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 176 | 328.2 | 329.5 | 328.6 | 327.6 | 330.5 | 0.7 | 0.5 | 0.4 | 0.0 | 32.0 | plateau |
| loaded | working_set | yes | 176 | 2152.6 | 2201.4 | 2186.1 | 2148.6 | 2207.7 | 18.8 | 15.0 | 14.5 | 11.9 | 43.7 | plateau |
| loaded | private | no | 176 | 2977.3 | 3083.5 | 3018.5 | 2968.4 | 3099.8 | 45.4 | 36.2 | 7.7 | 51.8 | 60.4 | plateau |
| loaded | rss_anon | yes | 176 | 2077.7 | 2125.4 | 2110.4 | 2073.3 | 2132.1 | 17.7 | 14.1 | 13.7 | 11.9 | 42.2 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | INTEL(R) XEON(R) PLATINUM 8573C |
| logical_cpus | 4 |
| ram_total_mb | 15989 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.17.0-1022-azure |
| runtime | .NET 10.0.13 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI/0.17.1 |
| gpu_identity | n/a |
| cpu_governor | performance |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"true","DOTNET_MULTILEVEL_LOOKUP":"0","DOTNET_NOLOGO":"true","DOTNET_ROOT_X64":"/usr/share/dotnet","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1"} |
| git_commit | 03570797591f1b66f32804bcdb8207aa58feb119 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821 |
| load before (loadavg, MemAvailable MB) | [2.05,0.7,0.25], 14074 |
| load after (loadavg, MemAvailable MB) | [2.98,2.92,2.85], 14140 |
