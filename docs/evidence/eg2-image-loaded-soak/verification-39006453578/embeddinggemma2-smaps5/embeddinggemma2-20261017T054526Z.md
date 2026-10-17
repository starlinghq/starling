# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3605 s |
| Load/unload cycles | 603 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 32562; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1632, 2159 |
| Unload ms (median, max) | 91.3, 97.0 |
| Request p50 ms (median over cycles) | 80.9 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3618 | 0 | 495.37 | 666.63 | 943.94 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 483 | 256.9 | 257.0 | 256.9 | 224.0 | 291.3 | 0.8 | 0.6 | 11.4 | 13.6 | 32.0 | plateau |
| unloaded | working_set | yes | 483 | 395.6 | 408.6 | 404.2 | 389.7 | 415.1 | 4.3 | 3.5 | 10.7 | -0.4 | 32.0 | plateau |
| unloaded | private | no | 483 | 1653.6 | 1667.0 | 1631.5 | 1583.1 | 1686.5 | 7.3 | 5.8 | 20.7 | 17.8 | 32.6 | plateau |
| unloaded | rss_anon | yes | 483 | 320.6 | 333.0 | 328.6 | 314.2 | 340.0 | 3.6 | 2.9 | 10.2 | -0.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 483 | 335.6 | 368.3 | 335.7 | 302.7 | 370.0 | 1.5 | 1.2 | 11.7 | 12.3 | 32.0 | plateau |
| loaded | working_set | yes | 483 | 1806.4 | 1821.3 | 1818.9 | 1780.2 | 1835.1 | 8.6 | 6.9 | 7.9 | 1.9 | 36.4 | plateau |
| loaded | private | no | 483 | 2205.7 | 2219.5 | 2185.9 | 2127.1 | 2250.6 | 9.9 | 7.9 | 17.0 | 19.9 | 43.7 | plateau |
| loaded | rss_anon | yes | 483 | 1731.5 | 1745.7 | 1743.5 | 1704.7 | 1760.1 | 7.9 | 6.3 | 7.4 | 1.9 | 34.9 | plateau |

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
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [3.8,1.04,0.4], 14143 |
| load after (loadavg, MemAvailable MB) | [6.12,5.34,5.11], 14015 |
