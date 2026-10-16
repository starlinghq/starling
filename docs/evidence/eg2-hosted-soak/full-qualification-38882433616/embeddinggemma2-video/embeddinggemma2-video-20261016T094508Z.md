# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded working_set retention 51 MB between loads, reused by the next load; unloaded private (address space, report-only) grows 354 MB; unloaded rss_anon retention 50 MB between loads, reused by the next load; loaded private (address space, report-only) grows 364 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3618 s |
| Load/unload cycles | 131 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 393; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2937, 3862 |
| Unload ms (median, max) | 169.2, 181.4 |
| Request p50 ms (median over cycles) | 12909.7 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 131 | 0 | 10.15 | 42.96 | 78.18 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 108 | 214.7 | 215.1 | 199.4 | 197.9 | 216.1 | 3.3 | 2.6 | -2.0 | 2.0 | 32.0 | plateau |
| unloaded | working_set | yes | 108 | 586.0 | 607.6 | 599.3 | 515.9 | 645.9 | 63.8 | 50.9 | 53.0 | 9.9 | 32.0 | retention |
| unloaded | private | no | 108 | 2970.0 | 3279.3 | 3232.9 | 2840.6 | 3388.6 | 444.4 | 354.1 | 378.3 | 35.5 | 64.7 | growth |
| unloaded | rss_anon | yes | 108 | 511.5 | 531.5 | 523.3 | 440.4 | 569.8 | 62.2 | 49.6 | 51.9 | 9.5 | 32.0 | retention |
| loaded | managed_heap | yes | 108 | 293.5 | 293.8 | 278.1 | 276.7 | 294.8 | 3.3 | 2.6 | -2.0 | 2.0 | 32.0 | plateau |
| loaded | working_set | yes | 108 | 2834.9 | 2841.4 | 2856.3 | 2729.8 | 2967.3 | 76.5 | 60.9 | 59.2 | 3.6 | 57.1 | plateau |
| loaded | private | no | 108 | 3546.2 | 3887.5 | 3843.5 | 3433.7 | 4036.4 | 456.3 | 363.6 | 408.1 | 20.1 | 76.9 | growth |
| loaded | rss_anon | yes | 108 | 2760.5 | 2765.3 | 2780.6 | 2654.1 | 2891.3 | 74.8 | 59.6 | 58.1 | 3.2 | 55.6 | plateau |

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
| git_commit | 9c972ca9e46d1614d819ecd497de352cd32cd759 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38882433616 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38882433616 |
| load before (loadavg, MemAvailable MB) | [2.25,0.71,0.25], 14116 |
| load after (loadavg, MemAvailable MB) | [6.75,6.62,6.35], 14247 |
