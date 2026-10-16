# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 905 s |
| Load/unload cycles | 118 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 6372; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2876, 3000 |
| Unload ms (median, max) | 107.9, 134.2 |
| Request p50 ms (median over cycles) | 93.2 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 708 | 0 | 528.53 | 735.49 | 1108.96 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 95 | 255.8 | 223.2 | 255.8 | 223.1 | 288.7 | 20.5 | 4.1 | 27.0 | 3.2 | 32.0 | plateau |
| unloaded | working_set | yes | 95 | 348.3 | 353.8 | 355.1 | 340.2 | 383.5 | 47.3 | 9.4 | 20.8 | 0.3 | 32.0 | plateau |
| unloaded | private | no | 95 | 659.5 | 647.6 | 649.0 | 601.2 | 760.0 | 35.3 | 7.0 | 53.6 | 6.1 | 32.0 | plateau |
| unloaded | rss_anon | yes | 95 | 275.2 | 279.2 | 280.9 | 267.0 | 309.3 | 40.1 | 8.0 | 19.8 | -0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 95 | 334.5 | 301.9 | 334.6 | 301.9 | 367.4 | 30.0 | 6.0 | 27.0 | 6.6 | 32.0 | plateau |
| loaded | working_set | yes | 95 | 1711.9 | 1717.7 | 1719.3 | 1708.8 | 2044.3 | -199.3 | -39.7 | 63.8 | -41.7 | 34.4 | plateau |
| loaded | private | no | 95 | 2269.7 | 2257.9 | 2274.8 | 2223.6 | 2907.0 | -373.5 | -74.4 | 131.0 | -61.9 | 45.5 | plateau |
| loaded | rss_anon | yes | 95 | 1638.7 | 1643.1 | 1644.7 | 1635.6 | 1970.1 | -206.6 | -41.2 | 62.7 | -42.1 | 32.9 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"true","DOTNET_MULTILEVEL_LOOKUP":"0","DOTNET_NOLOGO":"true","DOTNET_ROOT_X64":"/usr/share/dotnet","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | 9a4d2ce14573a1e1000bedb5de865a622db469de |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant mmap64k\u002Bsmaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant mmap64k\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [1.62,0.55,0.2], 14260 |
| load after (loadavg, MemAvailable MB) | [4.15,4.35,2.91], 14185 |
