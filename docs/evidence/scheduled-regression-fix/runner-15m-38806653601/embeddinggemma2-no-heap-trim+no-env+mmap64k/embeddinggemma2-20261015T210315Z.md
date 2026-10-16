# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 905 s |
| Load/unload cycles | 141 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 7614; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1850, 1975 |
| Unload ms (median, max) | 38.3, 46.4 |
| Request p50 ms (median over cycles) | 90.6 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 846 | 0 | 508.07 | 696.61 | 881.13 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 113 | 255.8 | 288.9 | 256.0 | 223.1 | 288.9 | -51.0 | -10.2 | 11.7 | 14.0 | 32.0 | plateau |
| unloaded | working_set | yes | 113 | 373.1 | 385.3 | 378.6 | 371.0 | 411.6 | -107.4 | -21.5 | 5.5 | 1.3 | 32.0 | plateau |
| unloaded | private | no | 113 | 662.1 | 683.6 | 662.2 | 613.7 | 723.0 | -186.8 | -37.3 | 4.4 | 17.4 | 32.0 | plateau |
| unloaded | rss_anon | yes | 113 | 300.0 | 310.7 | 304.2 | 297.8 | 337.9 | -113.3 | -22.6 | 4.6 | 1.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 113 | 334.5 | 367.7 | 334.7 | 301.9 | 367.7 | -51.0 | -10.2 | 11.7 | 14.0 | 32.0 | plateau |
| loaded | working_set | yes | 113 | 1725.3 | 1739.1 | 1734.3 | 1723.0 | 2061.4 | -315.3 | -63.0 | -71.3 | 5.7 | 34.7 | plateau |
| loaded | private | no | 113 | 2272.3 | 2309.9 | 2283.5 | 2224.3 | 2860.2 | -594.8 | -118.9 | -135.7 | 20.0 | 45.7 | plateau |
| loaded | rss_anon | yes | 113 | 1652.2 | 1664.5 | 1659.8 | 1649.9 | 1987.3 | -321.3 | -64.2 | -72.3 | 5.4 | 33.2 | plateau |

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
| git_commit | ff58b212568b4d484c6f849b75b0094a2e597636 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601, variant no-heap-trim\u002Bno-env\u002Bmmap64k --no-heap-trim |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601, variant no-heap-trim\u002Bno-env\u002Bmmap64k |
| load before (loadavg, MemAvailable MB) | [3.76,1.12,0.39], 14129 |
| load after (loadavg, MemAvailable MB) | [4.3,4.75,3.32], 14361 |
