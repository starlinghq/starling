# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 509 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 27486; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2054, 2950 |
| Unload ms (median, max) | 106.4, 118.1 |
| Request p50 ms (median over cycles) | 92.3 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3054 | 0 | 593.68 | 811.69 | 1207.29 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 409 | 223.7 | 258.1 | 256.3 | 223.2 | 290.8 | 14.3 | 11.5 | 16.7 | 6.2 | 32.0 | plateau |
| unloaded | working_set | yes | 409 | 428.1 | 416.1 | 409.7 | 397.6 | 431.0 | -2.3 | -1.8 | -9.9 | -1.3 | 32.0 | plateau |
| unloaded | private | no | 409 | 1609.2 | 1614.7 | 1610.4 | 1569.8 | 1678.5 | 13.2 | 10.5 | 6.2 | 13.5 | 32.2 | plateau |
| unloaded | rss_anon | yes | 409 | 353.6 | 340.9 | 334.6 | 323.0 | 356.5 | -3.1 | -2.5 | -10.5 | -1.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 409 | 302.4 | 336.9 | 335.0 | 302.0 | 369.5 | 14.3 | 11.5 | 16.7 | 6.2 | 32.0 | plateau |
| loaded | working_set | yes | 409 | 1839.0 | 1821.7 | 1820.2 | 1788.7 | 1848.2 | -4.3 | -3.5 | -11.3 | 1.9 | 36.4 | plateau |
| loaded | private | no | 409 | 2181.3 | 2166.8 | 2172.1 | 2112.9 | 2275.8 | 11.9 | 9.5 | 3.0 | 18.4 | 43.4 | plateau |
| loaded | rss_anon | yes | 409 | 1764.4 | 1746.5 | 1745.1 | 1713.5 | 1773.6 | -5.1 | -4.1 | -11.8 | 1.9 | 34.9 | plateau |

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
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39001130485 |
| load before (loadavg, MemAvailable MB) | [3.32,0.96,0.33], 14236 |
| load after (loadavg, MemAvailable MB) | [4.99,5.16,5.05], 14300 |
