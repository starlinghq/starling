# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3604 s |
| Load/unload cycles | 130 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 390; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3004, 3983 |
| Unload ms (median, max) | 168.4, 186.3 |
| Request p50 ms (median over cycles) | 12867.9 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 130 | 0 | 18.29 | 55.27 | 72.08 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 108 | 205.2 | 214.4 | 199.6 | 198.0 | 216.4 | -1.4 | -1.1 | 3.9 | 3.1 | 32.0 | plateau |
| unloaded | working_set | yes | 108 | 564.6 | 585.8 | 565.2 | 541.0 | 585.8 | 11.1 | 8.9 | 10.2 | 5.3 | 32.0 | plateau |
| unloaded | private | no | 108 | 2569.2 | 2619.2 | 2586.7 | 2543.5 | 2620.7 | 45.4 | 36.3 | 11.3 | 6.8 | 51.7 | plateau |
| unloaded | rss_anon | yes | 108 | 490.0 | 509.5 | 489.8 | 465.5 | 509.5 | 9.3 | 7.5 | 9.0 | 5.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 108 | 283.9 | 293.1 | 278.3 | 276.7 | 295.1 | -1.4 | -1.1 | 3.9 | 3.1 | 32.0 | plateau |
| loaded | working_set | yes | 108 | 2854.7 | 2860.0 | 2827.1 | 2762.0 | 2872.6 | 29.3 | 23.4 | -15.9 | 10.8 | 56.5 | plateau |
| loaded | private | no | 108 | 3217.0 | 3239.1 | 3199.0 | 3141.9 | 3252.7 | 25.9 | 20.7 | -16.1 | 2.9 | 64.0 | plateau |
| loaded | rss_anon | yes | 108 | 2780.0 | 2783.8 | 2751.6 | 2687.1 | 2796.4 | 27.7 | 22.1 | -17.0 | 10.5 | 55.0 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [2.12,0.71,0.25], 14197 |
| load after (loadavg, MemAvailable MB) | [6.47,6.21,6.1], 14249 |
