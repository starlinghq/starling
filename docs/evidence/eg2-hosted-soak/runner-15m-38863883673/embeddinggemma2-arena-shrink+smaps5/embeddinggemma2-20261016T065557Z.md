# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 905 s |
| Load/unload cycles | 149 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 8046; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1737, 2292 |
| Unload ms (median, max) | 93.5, 123.7 |
| Request p50 ms (median over cycles) | 84.5 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 894 | 0 | 454.27 | 643.34 | 890.88 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 120 | 223.2 | 255.9 | 255.9 | 223.2 | 289.0 | -74.5 | -14.9 | 11.6 | 11.7 | 32.0 | plateau |
| unloaded | working_set | yes | 120 | 489.8 | 551.2 | 536.8 | 489.8 | 570.2 | 89.0 | 17.8 | 13.0 | 19.2 | 32.0 | plateau |
| unloaded | private | no | 120 | 1712.5 | 1764.9 | 1750.7 | 1706.8 | 1822.4 | -59.7 | -11.9 | 6.1 | 16.3 | 35.0 | plateau |
| unloaded | rss_anon | yes | 120 | 415.1 | 476.2 | 462.2 | 415.1 | 495.2 | 83.6 | 16.7 | 12.3 | 18.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 120 | 301.9 | 334.6 | 334.6 | 301.9 | 367.7 | -74.0 | -14.8 | 11.6 | 8.9 | 32.0 | plateau |
| loaded | working_set | yes | 120 | 1910.0 | 1959.4 | 1954.4 | 1910.0 | 1995.2 | 56.9 | 11.4 | 7.5 | 16.3 | 39.1 | plateau |
| loaded | private | no | 120 | 2282.3 | 2325.1 | 2316.4 | 2258.2 | 2386.1 | -123.7 | -24.7 | -7.3 | 4.5 | 46.3 | plateau |
| loaded | rss_anon | yes | 120 | 1836.5 | 1884.4 | 1879.6 | 1836.5 | 1920.5 | 51.0 | 10.2 | 6.6 | 15.9 | 37.6 | plateau |

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
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant arena-shrink\u002Bsmaps5 --arena-shrink --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant arena-shrink\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [2.3,0.77,0.28], 14276 |
| load after (loadavg, MemAvailable MB) | [5.69,5.06,3.37], 14148 |
