# G4 soak: embeddinggemma-2/video-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** FAIL: unloaded working_set grows 195 MB (961 MB/h); unloaded rss_anon grows 193 MB (949 MB/h); loaded working_set grows 205 MB (1008 MB/h); loaded rss_anon grows 203 MB (997 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 919 s |
| Load/unload cycles | 40 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 120; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2365, 2993 |
| Unload ms (median, max) | 132.6, 137.1 |
| Request p50 ms (median over cycles) | 10237.3 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 40 | 0 | 21.50 | 64.52 | 76.21 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 36 | 214.5 | 198.4 | 202.5 | 198.4 | 215.9 | -16.8 | -3.4 | -6.2 | -7.3 | 32.0 | plateau |
| unloaded | working_set | yes | 36 | 495.9 | 639.3 | 573.6 | 473.2 | 684.4 | 961.0 | 195.3 | 87.9 | 89.6 | 32.0 | growth |
| unloaded | private | no | 36 | 2387.6 | 2638.4 | 2566.5 | 2387.6 | 2651.9 | 1111.3 | 225.9 | 170.2 | 93.0 | 51.3 | growth |
| unloaded | rss_anon | yes | 36 | 422.7 | 563.8 | 498.8 | 399.8 | 608.9 | 949.4 | 193.0 | 86.4 | 88.9 | 32.0 | growth |
| loaded | managed_heap | yes | 36 | 293.2 | 277.1 | 281.2 | 277.1 | 294.6 | -16.8 | -3.4 | -6.2 | -7.3 | 32.0 | plateau |
| loaded | working_set | yes | 36 | 2629.0 | 2787.9 | 2692.3 | 2611.4 | 2821.1 | 1007.9 | 204.9 | 68.4 | 99.0 | 53.8 | growth |
| loaded | private | no | 36 | 2988.8 | 3230.4 | 3158.2 | 2988.8 | 3243.9 | 1141.6 | 232.0 | 172.9 | 86.6 | 63.2 | growth |
| loaded | rss_anon | yes | 36 | 2555.8 | 2712.4 | 2617.7 | 2537.7 | 2745.9 | 997.5 | 202.8 | 66.9 | 98.2 | 52.4 | growth |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V74 80-Core Processor |
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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant arena-shrink\u002Bsmaps5 --arena-shrink --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant arena-shrink\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [3.46,0.85,0.29], 14223 |
| load after (loadavg, MemAvailable MB) | [6.6,6.15,4.16], 14204 |
