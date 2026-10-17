# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3627 s |
| Load/unload cycles | 88 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 352; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3142, 4038 |
| Unload ms (median, max) | 133.6, 143.6 |
| Request p50 ms (median over cycles) | 17975.6 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 88 | 0 | 17.47 | 50.00 | 92.55 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 73 | 159.6 | 160.2 | 159.7 | 159.1 | 160.7 | 0.8 | 0.6 | 0.3 | 0.5 | 32.0 | plateau |
| unloaded | working_set | yes | 73 | 381.3 | 389.9 | 390.0 | 367.1 | 454.2 | 38.7 | 30.9 | 17.3 | 7.9 | 32.0 | plateau |
| unloaded | private | no | 73 | 2358.0 | 2435.6 | 2447.1 | 2358.0 | 2581.0 | 69.4 | 55.5 | 45.8 | 1.2 | 48.9 | plateau |
| unloaded | rss_anon | yes | 73 | 307.1 | 313.7 | 314.6 | 292.2 | 378.1 | 36.5 | 29.2 | 16.2 | 7.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 73 | 283.0 | 277.4 | 282.8 | 276.3 | 284.4 | 1.1 | 0.9 | -1.9 | -1.0 | 32.0 | plateau |
| loaded | working_set | yes | 73 | 2088.0 | 2097.2 | 2097.7 | 2073.7 | 2186.4 | 38.0 | 30.3 | 10.4 | 11.5 | 42.0 | plateau |
| loaded | private | no | 73 | 2964.4 | 3057.9 | 3057.9 | 2964.4 | 3172.1 | 86.5 | 69.1 | 47.1 | 2.6 | 61.2 | plateau |
| loaded | rss_anon | yes | 73 | 2013.9 | 2021.1 | 2021.9 | 1999.0 | 2110.3 | 35.7 | 28.5 | 9.3 | 10.6 | 40.4 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [2.63,0.82,0.29], 14023 |
| load after (loadavg, MemAvailable MB) | [6.19,6.04,6.02], 14229 |
