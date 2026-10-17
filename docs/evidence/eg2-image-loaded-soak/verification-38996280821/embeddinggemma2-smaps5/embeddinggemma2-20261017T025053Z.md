# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3607 s |
| Load/unload cycles | 495 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 26730; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2119, 2950 |
| Unload ms (median, max) | 108.0, 121.5 |
| Request p50 ms (median over cycles) | 95.9 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 2970 | 0 | 602.70 | 833.08 | 1242.39 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 395 | 224.2 | 224.3 | 256.9 | 224.0 | 291.2 | 25.7 | 20.6 | 15.4 | -5.0 | 32.0 | plateau |
| unloaded | working_set | yes | 395 | 402.8 | 413.6 | 401.6 | 382.5 | 415.1 | 22.4 | 17.9 | 6.3 | 11.4 | 32.0 | plateau |
| unloaded | private | no | 395 | 1587.1 | 1632.0 | 1623.8 | 1571.3 | 1672.3 | 44.2 | 35.4 | 21.5 | 4.5 | 32.5 | plateau |
| unloaded | rss_anon | yes | 395 | 328.0 | 338.2 | 326.7 | 307.6 | 339.7 | 21.7 | 17.3 | 5.8 | 11.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 395 | 302.9 | 303.0 | 335.6 | 302.0 | 370.0 | 25.5 | 20.4 | 15.5 | -4.8 | 32.0 | plateau |
| loaded | working_set | yes | 395 | 1810.8 | 1819.9 | 1810.7 | 1779.1 | 1835.5 | 25.5 | 20.4 | 8.1 | 9.3 | 36.2 | plateau |
| loaded | private | no | 395 | 2155.2 | 2188.5 | 2179.7 | 2120.1 | 2238.4 | 42.4 | 33.9 | 22.2 | 0.9 | 43.6 | plateau |
| loaded | rss_anon | yes | 395 | 1736.0 | 1744.5 | 1735.3 | 1704.1 | 1760.1 | 24.7 | 19.8 | 7.7 | 9.3 | 34.7 | plateau |

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
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38996280821, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [1.75,0.56,0.2], 14218 |
| load after (loadavg, MemAvailable MB) | [5.16,5.14,4.93], 14032 |
