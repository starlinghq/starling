# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3607 s |
| Load/unload cycles | 479 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 25866; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2011, 2681 |
| Unload ms (median, max) | 118.8, 129.2 |
| Request p50 ms (median over cycles) | 99.5 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 2874 | 0 | 672.73 | 902.48 | 1364.80 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 385 | 256.3 | 223.5 | 256.1 | 223.3 | 290.7 | -2.1 | -1.7 | 11.6 | -16.3 | 32.0 | plateau |
| unloaded | working_set | yes | 385 | 385.8 | 391.3 | 400.6 | 381.8 | 430.7 | 11.0 | 8.8 | 11.5 | 0.1 | 32.0 | plateau |
| unloaded | private | no | 385 | 1589.8 | 1608.2 | 1604.8 | 1557.2 | 1678.9 | 1.0 | 0.8 | 23.2 | -21.3 | 32.1 | plateau |
| unloaded | rss_anon | yes | 385 | 311.1 | 316.1 | 325.4 | 307.1 | 355.8 | 10.3 | 8.2 | 10.9 | 0.1 | 32.0 | plateau |
| loaded | managed_heap | yes | 385 | 335.0 | 302.2 | 334.8 | 302.0 | 369.4 | -2.1 | -1.7 | 11.9 | -16.9 | 32.0 | plateau |
| loaded | working_set | yes | 385 | 1801.1 | 1820.9 | 1811.1 | 1777.7 | 1850.7 | 12.5 | 10.0 | 11.4 | 0.2 | 36.2 | plateau |
| loaded | private | no | 385 | 2161.9 | 2183.2 | 2163.7 | 2110.6 | 2231.1 | -1.9 | -1.5 | 18.6 | -23.7 | 43.3 | plateau |
| loaded | rss_anon | yes | 385 | 1726.5 | 1745.7 | 1736.0 | 1702.8 | 1775.8 | 11.7 | 9.4 | 10.8 | 0.2 | 34.7 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [3.77,1.09,0.38], 14269 |
| load after (loadavg, MemAvailable MB) | [5.34,5.37,5.16], 14343 |
