# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3602 s |
| Load/unload cycles | 87 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 348; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3147, 4169 |
| Unload ms (median, max) | 135.1, 140.3 |
| Request p50 ms (median over cycles) | 18089.2 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 87 | 0 | 19.09 | 49.67 | 67.26 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 72 | 159.4 | 159.3 | 159.3 | 159.1 | 160.2 | 0.7 | 0.5 | 0.2 | 0.3 | 32.0 | plateau |
| unloaded | working_set | yes | 72 | 365.5 | 371.7 | 372.7 | 365.5 | 441.6 | 3.3 | 2.6 | 5.7 | 13.7 | 32.0 | plateau |
| unloaded | private | no | 72 | 2483.3 | 2451.4 | 2459.0 | 2411.9 | 2550.0 | 8.8 | 6.9 | 15.5 | -16.4 | 49.2 | plateau |
| unloaded | rss_anon | yes | 72 | 291.4 | 295.7 | 297.7 | 290.5 | 365.6 | 1.0 | 0.8 | 4.6 | 13.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 72 | 276.5 | 278.0 | 280.2 | 276.1 | 283.9 | -0.4 | -0.3 | 1.9 | -0.9 | 32.0 | plateau |
| loaded | working_set | yes | 72 | 2073.0 | 2081.2 | 2080.9 | 2072.5 | 2150.3 | 4.8 | 3.8 | 6.5 | 14.7 | 41.6 | plateau |
| loaded | private | no | 72 | 3074.5 | 3097.9 | 3070.7 | 3018.3 | 3197.7 | 21.9 | 17.3 | 26.2 | -31.2 | 61.4 | plateau |
| loaded | rss_anon | yes | 72 | 1999.0 | 2005.2 | 2005.5 | 1997.8 | 2074.3 | 2.4 | 1.9 | 5.3 | 14.0 | 40.1 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485 |
| load before (loadavg, MemAvailable MB) | [3.18,1.08,0.38], 14245 |
| load after (loadavg, MemAvailable MB) | [6.59,6.32,6.17], 14176 |
