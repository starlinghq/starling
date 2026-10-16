# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 637 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 34398; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1092, 1975 |
| Unload ms (median, max) | 11.4, 18.6 |
| Request p50 ms (median over cycles) | 90.6 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3822 | 0 | 510.57 | 695.30 | 1094.12 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 512 | 288.9 | 258.1 | 258.0 | 223.3 | 290.8 | -3.8 | -3.1 | -3.4 | -13.4 | 32.0 | plateau |
| unloaded | working_set | yes | 512 | 1558.6 | 1543.5 | 1554.4 | 1471.3 | 1575.7 | -7.1 | -5.7 | -20.5 | -17.2 | 32.0 | plateau |
| unloaded | private | no | 512 | 1880.5 | 1838.7 | 1859.7 | 1806.2 | 1913.2 | -18.8 | -15.0 | -19.9 | -18.6 | 37.2 | plateau |
| unloaded | rss_anon | yes | 512 | 1484.3 | 1468.6 | 1479.6 | 1396.4 | 1501.4 | -7.9 | -6.3 | -21.1 | -17.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 512 | 367.7 | 336.9 | 336.7 | 302.0 | 369.5 | -3.8 | -3.1 | -3.4 | -13.4 | 32.0 | plateau |
| loaded | working_set | yes | 512 | 2354.7 | 2338.6 | 2346.4 | 2270.9 | 2672.9 | 19.9 | 15.9 | 8.8 | -4.1 | 46.9 | plateau |
| loaded | private | no | 512 | 2932.1 | 2889.3 | 2921.2 | 2850.1 | 3481.4 | 23.9 | 19.1 | 32.2 | 4.7 | 58.4 | plateau |
| loaded | rss_anon | yes | 512 | 2280.4 | 2263.7 | 2272.0 | 2196.0 | 2598.0 | 19.2 | 15.3 | 8.3 | -4.2 | 45.4 | plateau |

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
| git_commit | aeaf1b07cad8b65cf657bb3e136ee3d2124ef35d |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38809646407, variant no-heap-trim\u002Bno-env --no-heap-trim |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38809646407, variant no-heap-trim\u002Bno-env |
| load before (loadavg, MemAvailable MB) | [4.68,1.29,0.44], 14292 |
| load after (loadavg, MemAvailable MB) | [5.72,5.49,5.61], 13414 |
