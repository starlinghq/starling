# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3624 s |
| Load/unload cycles | 135 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 540; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2782, 3419 |
| Unload ms (median, max) | 94.8, 106.2 |
| Request p50 ms (median over cycles) | 11497.0 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 135 | 0 | 18.04 | 104.11 | 119.08 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 111 | 190.6 | 192.7 | 191.1 | 180.1 | 192.9 | -1.5 | -1.2 | -2.1 | 3.5 | 32.0 | plateau |
| unloaded | working_set | yes | 111 | 489.7 | 472.5 | 459.2 | 410.5 | 541.8 | -1.8 | -1.4 | 19.6 | 4.1 | 32.0 | plateau |
| unloaded | private | no | 111 | 2520.7 | 2569.4 | 2544.0 | 2459.9 | 2599.9 | 100.8 | 80.9 | 36.7 | 6.0 | 50.9 | plateau |
| unloaded | rss_anon | yes | 111 | 415.3 | 396.5 | 383.3 | 334.5 | 466.5 | -3.3 | -2.6 | 18.6 | 3.8 | 32.0 | plateau |
| loaded | managed_heap | yes | 111 | 269.3 | 271.5 | 269.8 | 258.8 | 271.7 | -1.5 | -1.2 | -2.1 | 3.5 | 32.0 | plateau |
| loaded | working_set | yes | 111 | 2749.9 | 2805.8 | 2784.5 | 2694.6 | 2875.2 | 79.5 | 63.8 | 14.5 | -6.1 | 55.7 | plateau |
| loaded | private | no | 111 | 3048.8 | 3081.5 | 3082.0 | 2988.0 | 3185.1 | 87.3 | 70.0 | 16.0 | -3.4 | 61.6 | plateau |
| loaded | rss_anon | yes | 111 | 2675.4 | 2729.8 | 2708.5 | 2619.4 | 2799.6 | 77.9 | 62.5 | 13.4 | -6.4 | 54.2 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [2.81,0.97,0.35], 13978 |
| load after (loadavg, MemAvailable MB) | [2.94,3.02,3], 13973 |
