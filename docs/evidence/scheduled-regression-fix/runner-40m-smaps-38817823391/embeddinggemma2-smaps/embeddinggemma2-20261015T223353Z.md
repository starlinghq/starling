# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (2400 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 2400 s / 2403 s |
| Load/unload cycles | 433 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 23382; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1093, 1872 |
| Unload ms (median, max) | 19.1, 47.7 |
| Request p50 ms (median over cycles) | 88.1 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 2598 | 0 | 495.02 | 680.33 | 1101.13 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 348 | 223.2 | 256.0 | 256.9 | 223.2 | 290.7 | -10.1 | -5.4 | 4.1 | -12.0 | 32.0 | plateau |
| unloaded | working_set | yes | 348 | 570.5 | 574.5 | 565.5 | 501.9 | 642.6 | -54.1 | -28.9 | -41.9 | -17.0 | 32.0 | plateau |
| unloaded | private | no | 348 | 1785.8 | 1994.6 | 1904.4 | 1774.2 | 1995.5 | 38.6 | 20.6 | 29.0 | 14.9 | 38.1 | plateau |
| unloaded | rss_anon | yes | 348 | 496.3 | 499.3 | 490.6 | 427.3 | 568.3 | -55.6 | -29.6 | -42.5 | -17.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 348 | 301.9 | 334.7 | 335.6 | 301.9 | 369.4 | -10.1 | -5.4 | 4.1 | -12.0 | 32.0 | plateau |
| loaded | working_set | yes | 348 | 2258.9 | 2258.8 | 2262.5 | 2202.3 | 2595.9 | -66.6 | -35.5 | 17.4 | -40.2 | 45.3 | plateau |
| loaded | private | no | 348 | 2866.0 | 3058.7 | 2985.0 | 2850.0 | 3531.4 | 6.5 | 3.5 | 98.4 | 1.5 | 59.7 | plateau |
| loaded | rss_anon | yes | 348 | 2184.7 | 2183.7 | 2187.7 | 2127.1 | 2521.4 | -68.1 | -36.3 | 16.8 | -40.6 | 43.8 | plateau |

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
| git_commit | e2537d98316c62ae60011d94ea09eaba443e0510 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 40m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38817823391, variant smaps --smaps-every 25 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38817823391, variant smaps |
| load before (loadavg, MemAvailable MB) | [1.61,0.51,0.18], 14167 |
| load after (loadavg, MemAvailable MB) | [5.67,5.68,5.44], 14253 |
