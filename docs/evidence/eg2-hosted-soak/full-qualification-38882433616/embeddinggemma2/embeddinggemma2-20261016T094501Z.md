# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded working_set retention 39 MB between loads, reused by the next load; unloaded private (address space, report-only) grows 45 MB; unloaded rss_anon retention 39 MB between loads, reused by the next load; loaded private (address space, report-only) grows 44 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 902 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 48708; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1311, 1707 |
| Unload ms (median, max) | 107.3, 156.2 |
| Request p50 ms (median over cycles) | 46.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 5412 | 0 | 304.54 | 383.33 | 502.18 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 721 | 255.8 | 292.9 | 288.6 | 223.4 | 292.9 | 3.6 | 2.9 | 6.8 | 10.5 | 32.0 | plateau |
| unloaded | working_set | yes | 721 | 429.2 | 516.9 | 461.2 | 428.7 | 516.9 | 49.0 | 39.2 | 30.1 | 36.9 | 32.0 | retention |
| unloaded | private | no | 721 | 1629.5 | 1738.0 | 1670.6 | 1622.5 | 1738.0 | 55.8 | 44.7 | 36.9 | 45.2 | 33.4 | growth |
| unloaded | rss_anon | yes | 721 | 354.6 | 441.7 | 385.9 | 354.0 | 441.7 | 48.2 | 38.6 | 29.6 | 36.9 | 32.0 | retention |
| loaded | managed_heap | yes | 721 | 334.6 | 371.6 | 367.3 | 302.1 | 371.6 | 3.6 | 2.9 | 6.8 | 10.5 | 32.0 | plateau |
| loaded | working_set | yes | 721 | 1860.0 | 1930.7 | 1877.7 | 1834.3 | 1933.7 | 45.4 | 36.3 | 25.9 | 38.6 | 37.6 | plateau |
| loaded | private | no | 721 | 2173.7 | 2274.2 | 2209.0 | 2160.6 | 2279.4 | 55.4 | 44.3 | 31.3 | 47.0 | 44.2 | growth |
| loaded | rss_anon | yes | 721 | 1785.3 | 1855.4 | 1802.5 | 1759.4 | 1858.5 | 44.6 | 35.7 | 25.4 | 38.5 | 36.0 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V45 96-Core Processor |
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
| git_commit | 9c972ca9e46d1614d819ecd497de352cd32cd759 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38882433616 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38882433616 |
| load before (loadavg, MemAvailable MB) | [3.48,1.06,0.37], 14247 |
| load after (loadavg, MemAvailable MB) | [2,2.18,2.18], 13943 |
