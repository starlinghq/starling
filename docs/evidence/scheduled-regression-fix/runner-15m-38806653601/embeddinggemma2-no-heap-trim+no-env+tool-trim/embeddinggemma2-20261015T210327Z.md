# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 903 s |
| Load/unload cycles | 158 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 8532; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1142, 1910 |
| Unload ms (median, max) | 13.0, 25.5 |
| Request p50 ms (median over cycles) | 90.5 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 948 | 0 | 508.37 | 682.29 | 996.95 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 127 | 255.8 | 224.3 | 256.0 | 223.1 | 289.5 | 71.7 | 14.4 | -0.1 | 10.8 | 32.0 | plateau |
| unloaded | working_set | yes | 127 | 1422.1 | 1444.2 | 1419.2 | 1297.1 | 1475.3 | -246.3 | -49.4 | -36.1 | 19.1 | 32.0 | plateau |
| unloaded | private | no | 127 | 1764.6 | 1768.2 | 1771.5 | 1716.9 | 1834.2 | 28.3 | 5.7 | -25.4 | 21.8 | 35.4 | plateau |
| unloaded | rss_anon | yes | 127 | 1348.6 | 1369.6 | 1345.0 | 1222.7 | 1401.4 | -251.7 | -50.5 | -37.0 | 18.8 | 32.0 | plateau |
| loaded | managed_heap | yes | 127 | 334.5 | 303.0 | 334.7 | 301.9 | 368.3 | 71.7 | 14.4 | -0.1 | 10.8 | 32.0 | plateau |
| loaded | working_set | yes | 127 | 2550.2 | 2517.0 | 2207.7 | 2102.6 | 2550.2 | -203.3 | -40.8 | -31.1 | 34.3 | 44.2 | plateau |
| loaded | private | no | 127 | 3332.6 | 3322.5 | 2821.0 | 2758.9 | 3386.2 | 45.1 | 9.0 | -14.5 | 52.5 | 56.4 | plateau |
| loaded | rss_anon | yes | 127 | 2476.7 | 2442.4 | 2133.3 | 2028.0 | 2476.7 | -208.7 | -41.8 | -31.9 | 34.0 | 42.7 | plateau |
| trimmed | managed_heap | yes | 127 | 255.9 | 224.3 | 256.0 | 223.2 | 289.6 | 71.7 | 14.4 | -0.1 | 10.8 | 32.0 | plateau |
| trimmed | working_set | yes | 127 | 523.5 | 509.6 | 523.3 | 413.4 | 570.8 | -248.2 | -49.8 | -30.5 | 11.1 | 32.0 | plateau |
| trimmed | private | no | 127 | 1736.4 | 1736.1 | 1742.8 | 1689.6 | 1810.0 | 25.7 | 5.1 | -25.9 | 20.1 | 34.9 | plateau |
| trimmed | rss_anon | yes | 127 | 450.0 | 435.0 | 448.6 | 338.9 | 497.0 | -253.7 | -50.9 | -31.4 | 10.9 | 32.0 | plateau |

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
| git_commit | ff58b212568b4d484c6f849b75b0094a2e597636 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601, variant no-heap-trim\u002Bno-env\u002Btool-trim --no-heap-trim --malloc-trim |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601, variant no-heap-trim\u002Bno-env\u002Btool-trim |
| load before (loadavg, MemAvailable MB) | [2.02,0.75,0.27], 14281 |
| load after (loadavg, MemAvailable MB) | [5.16,5.07,3.45], 14363 |
