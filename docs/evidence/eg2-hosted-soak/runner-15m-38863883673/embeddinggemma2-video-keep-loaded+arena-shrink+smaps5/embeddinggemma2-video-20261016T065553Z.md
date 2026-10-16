# G4 soak: embeddinggemma-2/video-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** FAIL: unloaded working_set grows 199 MB (1012 MB/h); unloaded rss_anon grows 197 MB (1003 MB/h); loaded working_set grows 199 MB (1012 MB/h); loaded rss_anon grows 197 MB (1003 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 905 s |
| Load/unload cycles | 39 |
| Keep loaded (diagnostic) | yes: one load, no unloads; the `unloaded` series is a second sample with the model loaded |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 117; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 0, 3847 |
| Unload ms (median, max) | 0.0, 0.0 |
| Request p50 ms (median over cycles) | 11413.4 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 39 | 0 | 25.60 | 82.43 | 87.61 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 35 | 121.0 | 136.0 | 121.0 | 119.4 | 136.9 | 29.6 | 5.8 | 3.3 | 10.7 | 32.0 | plateau |
| unloaded | working_set | yes | 35 | 2551.5 | 2722.6 | 2704.5 | 2551.5 | 2773.1 | 1011.9 | 199.0 | 121.8 | 66.7 | 54.1 | growth |
| unloaded | private | no | 35 | 2831.5 | 3062.3 | 2996.8 | 2830.3 | 3080.5 | 1135.5 | 223.2 | 154.9 | 73.3 | 59.9 | growth |
| unloaded | rss_anon | yes | 35 | 2479.1 | 2648.3 | 2630.6 | 2479.0 | 2698.8 | 1002.7 | 197.1 | 120.8 | 65.9 | 52.6 | growth |
| loaded | managed_heap | yes | 35 | 121.0 | 136.0 | 121.0 | 119.4 | 136.9 | 29.5 | 5.8 | 3.3 | 10.7 | 32.0 | plateau |
| loaded | working_set | yes | 35 | 2551.5 | 2722.6 | 2704.5 | 2551.5 | 2773.1 | 1012.0 | 199.0 | 121.8 | 66.7 | 54.1 | growth |
| loaded | private | no | 35 | 2831.5 | 3062.3 | 2996.8 | 2830.3 | 3080.5 | 1135.5 | 223.2 | 154.9 | 73.3 | 59.9 | growth |
| loaded | rss_anon | yes | 35 | 2479.1 | 2648.3 | 2630.6 | 2479.0 | 2698.8 | 1002.7 | 197.1 | 120.8 | 65.9 | 52.6 | growth |

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
| git_commit | 8acc5069fce1a0c89865ce3b766fd3bfa71f1521 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant keep-loaded\u002Barena-shrink\u002Bsmaps5 --keep-loaded --arena-shrink --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant keep-loaded\u002Barena-shrink\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [4.43,1.33,0.47], 14190 |
| load after (loadavg, MemAvailable MB) | [6.87,6.63,4.49], 14246 |
