# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (2400 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 139 MB; loaded private (address space, report-only) grows 173 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 2400 s / 2402 s |
| Load/unload cycles | 527 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 28458; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 730, 1198 |
| Unload ms (median, max) | 10.3, 14.1 |
| Request p50 ms (median over cycles) | 75.6 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3162 | 0 | 416.60 | 571.16 | 894.11 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 424 | 288.5 | 256.0 | 256.9 | 223.2 | 290.7 | 9.8 | 5.2 | 15.8 | 9.4 | 32.0 | plateau |
| unloaded | working_set | yes | 424 | 1522.7 | 1587.9 | 1576.2 | 1413.8 | 1602.3 | 246.0 | 131.1 | 84.5 | -2.6 | 32.0 | plateau |
| unloaded | private | no | 424 | 1864.3 | 1898.4 | 1887.7 | 1766.2 | 1947.2 | 260.0 | 138.6 | 101.1 | 19.1 | 37.8 | growth |
| unloaded | rss_anon | yes | 424 | 1448.4 | 1512.6 | 1501.1 | 1339.1 | 1527.0 | 244.6 | 130.4 | 84.0 | -2.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 424 | 367.2 | 334.7 | 335.7 | 302.0 | 369.4 | 9.8 | 5.2 | 15.8 | 9.4 | 32.0 | plateau |
| loaded | working_set | yes | 424 | 2308.9 | 2374.1 | 2376.1 | 2235.8 | 2690.5 | 282.6 | 150.6 | 57.9 | 9.9 | 47.5 | plateau |
| loaded | private | no | 424 | 2904.3 | 2938.4 | 2936.8 | 2806.2 | 3499.4 | 323.6 | 172.5 | 57.0 | 41.7 | 58.7 | growth |
| loaded | rss_anon | yes | 424 | 2234.6 | 2298.8 | 2301.0 | 2161.0 | 2615.2 | 281.2 | 149.9 | 57.4 | 9.5 | 46.0 | plateau |

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
| git_commit | e2537d98316c62ae60011d94ea09eaba443e0510 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 40m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38817823391, variant no-heap-trim\u002Bsmaps --no-heap-trim --smaps-every 25 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38817823391, variant no-heap-trim\u002Bsmaps |
| load before (loadavg, MemAvailable MB) | [1.56,0.7,0.27], 14412 |
| load after (loadavg, MemAvailable MB) | [6.11,5.94,5.57], 13329 |
