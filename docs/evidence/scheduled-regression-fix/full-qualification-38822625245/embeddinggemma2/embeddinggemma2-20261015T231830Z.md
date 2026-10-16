# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3602 s |
| Load/unload cycles | 974 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 52596; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1277, 1655 |
| Unload ms (median, max) | 112.0, 198.6 |
| Request p50 ms (median over cycles) | 46.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 5844 | 0 | 223.66 | 320.17 | 424.84 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 779 | 255.8 | 292.9 | 288.6 | 223.4 | 321.1 | 1.1 | 0.9 | -3.7 | 9.9 | 32.0 | plateau |
| unloaded | working_set | yes | 779 | 547.6 | 577.0 | 568.2 | 520.2 | 583.9 | 29.0 | 23.2 | 28.6 | 12.4 | 32.0 | plateau |
| unloaded | private | no | 779 | 1757.8 | 1814.2 | 1803.5 | 1705.6 | 1854.1 | 21.9 | 17.5 | 18.8 | 20.3 | 36.1 | plateau |
| unloaded | rss_anon | yes | 779 | 473.1 | 501.8 | 493.2 | 445.6 | 508.7 | 28.2 | 22.6 | 28.1 | 12.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 779 | 334.6 | 371.7 | 367.3 | 302.1 | 399.8 | 1.0 | 0.8 | -3.0 | 10.0 | 32.0 | plateau |
| loaded | working_set | yes | 779 | 2234.2 | 2263.4 | 2256.5 | 2198.3 | 2299.7 | 24.8 | 19.8 | 27.5 | 13.4 | 45.1 | plateau |
| loaded | private | no | 779 | 2805.9 | 2866.4 | 2855.7 | 2753.7 | 2936.6 | 20.8 | 16.6 | 20.8 | 18.8 | 57.1 | plateau |
| loaded | rss_anon | yes | 779 | 2159.6 | 2188.2 | 2181.5 | 2123.7 | 2224.9 | 24.0 | 19.2 | 26.9 | 13.4 | 43.6 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V45 96-Core Processor |
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
| git_commit | ba816c33548410a5920014b5fb010799fd6a4f3a |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38822625245 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38822625245 |
| load before (loadavg, MemAvailable MB) | [1.62,0.48,0.17], 14209 |
| load after (loadavg, MemAvailable MB) | [2.09,2.2,2.17], 13920 |
