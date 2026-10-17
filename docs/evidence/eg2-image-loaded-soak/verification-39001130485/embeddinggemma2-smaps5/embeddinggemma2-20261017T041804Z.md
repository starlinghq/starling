# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3602 s |
| Load/unload cycles | 484 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 26136; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1998, 2623 |
| Unload ms (median, max) | 119.1, 127.3 |
| Request p50 ms (median over cycles) | 98.6 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 2904 | 0 | 655.95 | 893.01 | 1283.10 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 388 | 256.7 | 291.2 | 257.4 | 224.0 | 291.2 | 5.7 | 4.5 | 4.9 | 3.3 | 32.0 | plateau |
| unloaded | working_set | yes | 388 | 390.6 | 417.4 | 409.0 | 386.2 | 435.8 | 1.3 | 1.0 | 11.6 | 0.2 | 32.0 | plateau |
| unloaded | private | no | 388 | 1607.0 | 1678.0 | 1639.6 | 1570.8 | 1710.2 | 13.9 | 11.1 | 29.4 | 1.1 | 32.8 | plateau |
| unloaded | rss_anon | yes | 388 | 315.8 | 342.0 | 333.6 | 311.3 | 360.8 | 0.5 | 0.4 | 11.1 | 0.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 388 | 335.4 | 369.9 | 336.1 | 302.0 | 370.0 | 5.7 | 4.5 | 4.1 | 1.9 | 32.0 | plateau |
| loaded | working_set | yes | 388 | 1821.4 | 1833.6 | 1834.4 | 1804.0 | 1878.0 | 1.9 | 1.5 | 15.1 | 0.4 | 36.7 | plateau |
| loaded | private | no | 388 | 2190.3 | 2230.2 | 2198.3 | 2133.5 | 2293.7 | 7.0 | 5.6 | 22.3 | 0.5 | 44.0 | plateau |
| loaded | rss_anon | yes | 388 | 1746.6 | 1758.2 | 1759.1 | 1729.2 | 1802.9 | 1.1 | 0.9 | 14.6 | 0.4 | 35.2 | plateau |

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
| git_commit | 03570797591f1b66f32804bcdb8207aa58feb119 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [5.48,1.46,0.5], 14291 |
| load after (loadavg, MemAvailable MB) | [5.4,5.5,5.23], 14273 |
