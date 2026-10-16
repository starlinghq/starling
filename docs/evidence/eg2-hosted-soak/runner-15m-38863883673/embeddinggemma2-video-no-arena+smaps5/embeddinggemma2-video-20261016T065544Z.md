# G4 soak: embeddinggemma-2/video-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 918 s |
| Load/unload cycles | 30 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 90; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2982, 3939 |
| Unload ms (median, max) | 170.6, 178.4 |
| Request p50 ms (median over cycles) | 12875.9 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 30 | 0 | 18.70 | 64.81 | 83.76 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 28 | 198.9 | 199.4 | 198.9 | 198.2 | 215.0 | -47.1 | -9.6 | -1.7 | 2.6 | 32.0 | plateau |
| unloaded | working_set | yes | 28 | 516.7 | 568.4 | 532.4 | 492.1 | 577.3 | 139.3 | 28.3 | -4.6 | 83.2 | 32.0 | plateau |
| unloaded | private | no | 28 | 2493.0 | 2721.8 | 2712.0 | 2493.0 | 2819.6 | 782.8 | 159.0 | 208.3 | -3.7 | 54.2 | plateau |
| unloaded | rss_anon | yes | 28 | 444.3 | 493.4 | 457.7 | 417.8 | 502.6 | 128.6 | 26.1 | -6.3 | 82.7 | 32.0 | plateau |
| loaded | managed_heap | yes | 28 | 277.6 | 278.1 | 277.6 | 276.9 | 293.7 | -47.1 | -9.6 | -1.7 | 2.6 | 32.0 | plateau |
| loaded | working_set | yes | 28 | 2757.9 | 2825.6 | 2830.8 | 2757.9 | 2882.3 | 153.5 | 31.2 | 59.3 | 45.0 | 56.6 | plateau |
| loaded | private | no | 28 | 3085.1 | 3357.6 | 3326.4 | 3085.1 | 3451.4 | 883.7 | 179.5 | 246.5 | -7.0 | 66.5 | plateau |
| loaded | rss_anon | yes | 28 | 2685.6 | 2750.9 | 2756.0 | 2684.6 | 2807.6 | 143.1 | 29.1 | 57.3 | 44.3 | 55.1 | plateau |

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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant no-arena\u002Bsmaps5 --no-cpu-arena --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant no-arena\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [1.59,0.58,0.22], 14111 |
| load after (loadavg, MemAvailable MB) | [6.34,6.32,4.29], 14293 |
