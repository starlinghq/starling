# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3600 s |
| Load/unload cycles | 512 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 27648; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2060, 3030 |
| Unload ms (median, max) | 107.5, 113.6 |
| Request p50 ms (median over cycles) | 91.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3072 | 0 | 587.21 | 799.73 | 1191.04 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 411 | 256.4 | 223.5 | 256.8 | 223.3 | 290.8 | -4.7 | -3.8 | -0.3 | 1.4 | 32.0 | plateau |
| unloaded | working_set | yes | 411 | 425.5 | 389.8 | 403.7 | 389.8 | 425.9 | -7.0 | -5.6 | -17.2 | -2.9 | 32.0 | plateau |
| unloaded | private | no | 411 | 1619.6 | 1601.0 | 1618.5 | 1558.7 | 1683.4 | 10.7 | 8.5 | -9.9 | -3.4 | 32.4 | plateau |
| unloaded | rss_anon | yes | 411 | 351.0 | 314.7 | 328.7 | 314.7 | 351.4 | -7.8 | -6.3 | -17.7 | -2.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 411 | 335.1 | 302.2 | 335.6 | 302.0 | 369.5 | -4.7 | -3.8 | -0.3 | 1.4 | 32.0 | plateau |
| loaded | working_set | yes | 411 | 1834.6 | 1803.5 | 1812.1 | 1787.9 | 1847.7 | -10.9 | -8.7 | -18.8 | -2.4 | 36.2 | plateau |
| loaded | private | no | 411 | 2191.7 | 2164.3 | 2175.4 | 2118.5 | 2246.1 | -12.5 | -10.0 | -20.1 | -4.0 | 43.5 | plateau |
| loaded | rss_anon | yes | 411 | 1760.1 | 1728.4 | 1737.0 | 1712.8 | 1773.3 | -11.7 | -9.3 | -19.3 | -2.4 | 34.7 | plateau |

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
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821 |
| load before (loadavg, MemAvailable MB) | [3.3,0.93,0.32], 14011 |
| load after (loadavg, MemAvailable MB) | [5.5,5,4.9], 14204 |
