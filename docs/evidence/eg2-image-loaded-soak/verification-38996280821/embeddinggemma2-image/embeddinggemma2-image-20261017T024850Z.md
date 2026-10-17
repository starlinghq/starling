# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3614 s |
| Load/unload cycles | 181 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 724; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1855, 2387 |
| Unload ms (median, max) | 117.9, 148.3 |
| Request p50 ms (median over cycles) | 8612.2 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 181 | 0 | 45.73 | 112.11 | 180.41 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 149 | 159.4 | 160.3 | 159.4 | 159.2 | 160.3 | 0.7 | 0.6 | 0.3 | 0.1 | 32.0 | plateau |
| unloaded | working_set | yes | 149 | 373.6 | 382.7 | 380.5 | 373.6 | 450.1 | 3.9 | 3.2 | 9.6 | -9.9 | 32.0 | plateau |
| unloaded | private | no | 149 | 2456.8 | 2451.6 | 2462.8 | 2432.3 | 2557.7 | 2.5 | 2.0 | 11.0 | 2.4 | 49.3 | plateau |
| unloaded | rss_anon | yes | 149 | 298.7 | 306.4 | 304.3 | 298.5 | 373.9 | 2.5 | 2.0 | 8.5 | -10.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 149 | 276.6 | 277.4 | 277.4 | 276.0 | 289.1 | 1.0 | 0.8 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | working_set | yes | 149 | 2080.7 | 2088.2 | 2087.5 | 2079.9 | 2180.5 | 9.5 | 7.6 | 9.6 | -5.5 | 41.7 | plateau |
| loaded | private | no | 149 | 3000.6 | 2977.3 | 2998.2 | 2972.6 | 3093.9 | -3.2 | -2.6 | 3.8 | 6.9 | 60.0 | plateau |
| loaded | rss_anon | yes | 149 | 2005.8 | 2012.0 | 2011.3 | 2005.0 | 2104.3 | 8.1 | 6.5 | 8.5 | -5.6 | 40.2 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821 |
| load before (loadavg, MemAvailable MB) | [2.96,0.8,0.27], 14187 |
| load after (loadavg, MemAvailable MB) | [2.94,2.96,2.9], 14166 |
