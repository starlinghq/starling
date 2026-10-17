# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3605 s |
| Load/unload cycles | 276 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 828; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1960, 2528 |
| Unload ms (median, max) | 123.7, 186.5 |
| Request p50 ms (median over cycles) | 5629.3 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 276 | 0 | 16.59 | 64.37 | 74.38 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 225 | 171.0 | 170.7 | 170.8 | 169.7 | 172.4 | 0.9 | 0.8 | 0.5 | 0.5 | 32.0 | plateau |
| unloaded | working_set | yes | 225 | 407.8 | 404.3 | 406.0 | 375.2 | 474.6 | -1.5 | -1.2 | -5.6 | 3.2 | 32.0 | plateau |
| unloaded | private | no | 225 | 2488.0 | 2458.3 | 2477.7 | 2416.0 | 2558.5 | 28.0 | 22.4 | 0.6 | 7.1 | 49.6 | plateau |
| unloaded | rss_anon | yes | 225 | 332.4 | 327.9 | 330.1 | 299.2 | 398.3 | -2.4 | -1.9 | -6.2 | 3.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 225 | 329.0 | 328.0 | 329.2 | 327.6 | 333.0 | 1.1 | 0.9 | 0.8 | 0.7 | 32.0 | plateau |
| loaded | working_set | yes | 225 | 2171.3 | 2169.7 | 2174.5 | 2132.9 | 2245.6 | 16.1 | 12.9 | 22.5 | 15.7 | 43.5 | plateau |
| loaded | private | no | 225 | 3071.6 | 3059.5 | 3085.7 | 2995.3 | 3149.5 | 28.4 | 22.7 | 18.9 | 17.4 | 61.7 | plateau |
| loaded | rss_anon | yes | 225 | 2095.6 | 2093.4 | 2098.7 | 2056.9 | 2169.6 | 15.2 | 12.2 | 21.9 | 15.7 | 42.0 | plateau |

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
| git_commit | 03570797591f1b66f32804bcdb8207aa58feb119 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [1.71,0.6,0.22], 14137 |
| load after (loadavg, MemAvailable MB) | [2.74,2.87,2.77], 14225 |
