# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3605 s |
| Load/unload cycles | 505 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 27270; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2074, 2936 |
| Unload ms (median, max) | 109.3, 116.9 |
| Request p50 ms (median over cycles) | 94.0 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3030 | 0 | 594.52 | 824.09 | 1232.03 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 405 | 288.9 | 290.7 | 256.1 | 223.2 | 290.7 | -12.7 | -10.1 | -17.6 | 1.8 | 32.0 | plateau |
| unloaded | working_set | yes | 405 | 410.8 | 402.2 | 396.8 | 379.8 | 421.4 | -13.9 | -11.1 | -29.5 | -2.6 | 32.0 | plateau |
| unloaded | private | no | 405 | 1652.0 | 1641.8 | 1606.0 | 1554.2 | 1692.7 | -26.6 | -21.3 | -49.2 | -5.7 | 32.1 | plateau |
| unloaded | rss_anon | yes | 405 | 336.2 | 327.0 | 321.7 | 304.6 | 346.8 | -14.7 | -11.8 | -30.1 | -2.6 | 32.0 | plateau |
| loaded | managed_heap | yes | 405 | 367.6 | 369.4 | 334.9 | 302.0 | 369.4 | -12.6 | -10.1 | -17.6 | 1.8 | 32.0 | plateau |
| loaded | working_set | yes | 405 | 1819.8 | 1807.7 | 1804.7 | 1769.3 | 1839.6 | -10.3 | -8.3 | -29.9 | -0.7 | 36.1 | plateau |
| loaded | private | no | 405 | 2211.9 | 2214.0 | 2169.6 | 2114.5 | 2272.8 | -27.5 | -22.0 | -50.8 | -5.7 | 43.4 | plateau |
| loaded | rss_anon | yes | 405 | 1745.3 | 1732.5 | 1729.7 | 1694.5 | 1765.1 | -11.1 | -8.9 | -30.4 | -0.7 | 34.6 | plateau |

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
| git_commit | 03570797591f1b66f32804bcdb8207aa58feb119 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578 |
| load before (loadavg, MemAvailable MB) | [2.81,0.81,0.28], 14100 |
| load after (loadavg, MemAvailable MB) | [5.63,5.31,5.05], 14284 |
