# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** FAIL: unloaded working_set grows 63 MB (79 MB/h); unloaded rss_anon grows 62 MB (78 MB/h); loaded working_set grows 71 MB (88 MB/h); loaded rss_anon grows 70 MB (88 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 512 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 27648; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2087, 2871 |
| Unload ms (median, max) | 106.9, 136.2 |
| Request p50 ms (median over cycles) | 90.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3072 | 0 | 592.75 | 810.58 | 1135.52 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 410 | 256.4 | 288.8 | 256.1 | 223.3 | 323.2 | 3.1 | 2.4 | 2.9 | -8.2 | 32.0 | plateau |
| unloaded | working_set | yes | 410 | 422.6 | 483.1 | 464.7 | 421.8 | 489.3 | 78.8 | 63.0 | 45.3 | 16.1 | 32.0 | growth |
| unloaded | private | no | 410 | 1629.4 | 1721.2 | 1670.7 | 1598.8 | 1766.7 | 82.4 | 65.9 | 50.4 | 4.7 | 33.4 | plateau |
| unloaded | rss_anon | yes | 410 | 348.0 | 408.0 | 389.6 | 347.2 | 414.2 | 78.1 | 62.5 | 44.8 | 16.1 | 32.0 | growth |
| loaded | managed_heap | yes | 410 | 335.1 | 367.5 | 334.9 | 302.0 | 401.9 | 3.1 | 2.4 | 2.9 | -8.2 | 32.0 | plateau |
| loaded | working_set | yes | 410 | 1838.8 | 1891.3 | 1883.7 | 1825.2 | 1919.4 | 88.4 | 70.7 | 47.8 | 23.0 | 37.7 | growth |
| loaded | private | no | 410 | 2181.5 | 2273.3 | 2233.8 | 2152.1 | 2350.8 | 85.3 | 68.3 | 54.3 | 4.0 | 44.7 | plateau |
| loaded | rss_anon | yes | 410 | 1764.3 | 1816.2 | 1808.6 | 1750.6 | 1844.3 | 87.6 | 70.1 | 47.3 | 23.0 | 36.2 | growth |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 7763 64-Core Processor |
| logical_cpus | 4 |
| ram_total_mb | 15994 |
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
| git_commit | f43a648a795ed6cda366007a61fd7b4158a93082 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [1.7,0.52,0.18], 14102 |
| load after (loadavg, MemAvailable MB) | [4.74,4.98,4.97], 14262 |
