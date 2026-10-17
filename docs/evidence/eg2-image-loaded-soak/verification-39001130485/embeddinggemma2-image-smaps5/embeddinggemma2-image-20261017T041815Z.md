# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3609 s |
| Load/unload cycles | 87 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 348; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3121, 4180 |
| Unload ms (median, max) | 134.0, 143.9 |
| Request p50 ms (median over cycles) | 18125.5 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 87 | 0 | 16.45 | 57.47 | 92.44 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 72 | 159.6 | 160.1 | 159.7 | 159.1 | 160.7 | 0.8 | 0.6 | 0.3 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 72 | 386.1 | 396.8 | 388.5 | 367.4 | 465.0 | 24.0 | 19.0 | 16.2 | 20.9 | 32.0 | plateau |
| unloaded | private | no | 72 | 2419.7 | 2454.2 | 2440.7 | 2402.2 | 2512.0 | 38.7 | 30.7 | 43.8 | 27.4 | 48.8 | plateau |
| unloaded | rss_anon | yes | 72 | 311.9 | 320.7 | 312.7 | 292.7 | 389.7 | 21.8 | 17.3 | 15.1 | 19.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 72 | 282.8 | 277.2 | 277.4 | 276.3 | 284.4 | 1.3 | 1.0 | -1.0 | 0.4 | 32.0 | plateau |
| loaded | working_set | yes | 72 | 2092.3 | 2103.6 | 2095.6 | 2073.6 | 2166.4 | 28.3 | 22.4 | 13.2 | 30.1 | 41.9 | plateau |
| loaded | private | no | 72 | 3010.2 | 3044.9 | 3052.2 | 2997.5 | 3122.7 | 40.6 | 32.2 | 43.5 | 28.4 | 61.0 | plateau |
| loaded | rss_anon | yes | 72 | 2018.2 | 2027.5 | 2019.9 | 1998.9 | 2090.3 | 26.0 | 20.7 | 12.0 | 29.1 | 40.4 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [1.65,0.5,0.17], 14210 |
| load after (loadavg, MemAvailable MB) | [7.53,6.76,6.29], 13911 |
