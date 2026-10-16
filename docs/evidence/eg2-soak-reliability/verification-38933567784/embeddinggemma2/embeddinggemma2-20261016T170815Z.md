# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 507 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 27378; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2069, 2912 |
| Unload ms (median, max) | 107.4, 118.3 |
| Request p50 ms (median over cycles) | 92.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3042 | 0 | 594.79 | 815.80 | 1154.53 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 407 | 256.4 | 225.5 | 256.1 | 223.2 | 290.8 | -5.0 | -4.0 | -16.9 | -20.5 | 32.0 | plateau |
| unloaded | working_set | yes | 407 | 385.5 | 405.4 | 405.5 | 381.1 | 440.0 | 1.9 | 1.6 | 3.6 | -2.4 | 32.0 | plateau |
| unloaded | private | no | 407 | 1593.7 | 1586.1 | 1619.3 | 1562.0 | 1699.3 | 0.6 | 0.5 | -8.3 | -23.4 | 32.4 | plateau |
| unloaded | rss_anon | yes | 407 | 311.1 | 330.3 | 330.5 | 306.7 | 365.3 | 1.1 | 0.9 | 3.1 | -2.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 407 | 335.1 | 304.2 | 334.8 | 302.0 | 369.5 | -5.0 | -4.0 | -16.9 | -20.5 | 32.0 | plateau |
| loaded | working_set | yes | 407 | 1806.8 | 1833.1 | 1819.5 | 1781.8 | 1859.5 | 3.3 | 2.7 | 4.1 | -1.1 | 36.4 | plateau |
| loaded | private | no | 407 | 2159.5 | 2150.2 | 2175.6 | 2110.2 | 2251.5 | 0.1 | 0.1 | -10.6 | -21.8 | 43.5 | plateau |
| loaded | rss_anon | yes | 407 | 1732.3 | 1758.0 | 1744.5 | 1707.3 | 1784.9 | 2.5 | 2.0 | 3.6 | -1.2 | 34.9 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784 |
| load before (loadavg, MemAvailable MB) | [1.64,0.48,0.17], 14139 |
| load after (loadavg, MemAvailable MB) | [5.76,5.43,5.06], 14313 |
