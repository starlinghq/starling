# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 903 s |
| Load/unload cycles | 132 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 7128; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2043, 2889 |
| Unload ms (median, max) | 121.6, 147.5 |
| Request p50 ms (median over cycles) | 91.6 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 792 | 0 | 517.49 | 722.14 | 1067.06 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 106 | 223.2 | 223.7 | 256.1 | 223.2 | 289.0 | 29.3 | 5.9 | 10.1 | -9.5 | 32.0 | plateau |
| unloaded | working_set | yes | 106 | 489.6 | 502.0 | 531.2 | 478.4 | 568.2 | 16.2 | 3.2 | 47.6 | -45.2 | 32.0 | plateau |
| unloaded | private | no | 106 | 1683.4 | 1756.8 | 1820.2 | 1683.4 | 1872.9 | 193.8 | 38.7 | 92.3 | -60.5 | 36.4 | plateau |
| unloaded | rss_anon | yes | 106 | 416.5 | 427.3 | 456.8 | 403.4 | 493.7 | 10.2 | 2.0 | 46.9 | -45.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 106 | 301.9 | 302.5 | 334.8 | 301.9 | 367.7 | 29.3 | 5.9 | 10.1 | -9.5 | 32.0 | plateau |
| loaded | working_set | yes | 106 | 2189.5 | 2181.2 | 2220.5 | 2180.3 | 2521.7 | -66.8 | -13.3 | 35.1 | -27.1 | 44.4 | plateau |
| loaded | private | no | 106 | 2779.2 | 2822.0 | 2901.2 | 2779.2 | 3452.9 | 57.0 | 11.4 | 93.0 | -35.1 | 58.0 | plateau |
| loaded | rss_anon | yes | 106 | 2116.4 | 2106.6 | 2146.1 | 2106.6 | 2447.6 | -73.5 | -14.7 | 34.1 | -27.4 | 42.9 | plateau |

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
| git_commit | 9a4d2ce14573a1e1000bedb5de865a622db469de |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [1.71,0.55,0.19], 14210 |
| load after (loadavg, MemAvailable MB) | [5.28,4.49,2.98], 14098 |
