# G4 soak: embeddinggemma-2/video-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 909 s |
| Load/unload cycles | 62 |
| Keep loaded (diagnostic) | yes: one load, no unloads; the `unloaded` series is a second sample with the model loaded |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 186; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 0, 3497 |
| Unload ms (median, max) | 0.0, 0.0 |
| Request p50 ms (median over cycles) | 7204.6 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 62 | 0 | 4.92 | 53.37 | 70.70 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 55 | 152.9 | 153.1 | 152.2 | 136.3 | 153.7 | 5.2 | 1.1 | 1.3 | -2.9 | 32.0 | plateau |
| unloaded | working_set | yes | 55 | 2844.7 | 2861.9 | 2927.7 | 2789.7 | 3049.2 | 223.2 | 44.9 | 107.0 | -125.7 | 58.6 | plateau |
| unloaded | private | no | 55 | 3075.0 | 3147.0 | 3154.6 | 3030.2 | 3278.0 | 373.1 | 75.0 | 112.5 | -72.4 | 63.1 | plateau |
| unloaded | rss_anon | yes | 55 | 2771.6 | 2787.4 | 2854.2 | 2715.5 | 2975.1 | 216.9 | 43.6 | 106.1 | -126.2 | 57.1 | plateau |
| loaded | managed_heap | yes | 55 | 152.9 | 153.1 | 152.2 | 136.3 | 153.7 | 5.2 | 1.1 | 1.3 | -2.9 | 32.0 | plateau |
| loaded | working_set | yes | 55 | 2844.7 | 2861.9 | 2927.7 | 2789.7 | 3049.2 | 223.6 | 45.0 | 107.1 | -125.7 | 58.6 | plateau |
| loaded | private | no | 55 | 3075.0 | 3147.0 | 3154.6 | 3030.2 | 3278.0 | 373.4 | 75.1 | 112.5 | -72.4 | 63.1 | plateau |
| loaded | rss_anon | yes | 55 | 2771.6 | 2787.4 | 2854.2 | 2715.5 | 2975.1 | 217.3 | 43.7 | 106.2 | -126.2 | 57.1 | plateau |

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
| git_commit | 8acc5069fce1a0c89865ce3b766fd3bfa71f1521 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant keep-loaded\u002Bno-arena\u002Bsmaps5 --keep-loaded --no-cpu-arena --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant keep-loaded\u002Bno-arena\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [2.2,0.77,0.28], 13857 |
| load after (loadavg, MemAvailable MB) | [3.13,3.16,2.19], 14237 |
