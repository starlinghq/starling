# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 541 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 29214; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1789, 2978 |
| Unload ms (median, max) | 63.0, 92.1 |
| Request p50 ms (median over cycles) | 96.0 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3246 | 0 | 524.46 | 740.67 | 1101.97 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 434 | 256.3 | 225.5 | 256.3 | 223.2 | 321.2 | 7.8 | 6.2 | 14.7 | -6.7 | 32.0 | plateau |
| unloaded | working_set | yes | 434 | 1494.7 | 1581.7 | 1585.7 | 1401.2 | 1656.2 | 158.2 | 126.4 | 217.5 | 4.5 | 32.0 | plateau |
| unloaded | private | no | 434 | 1796.6 | 1968.3 | 2002.2 | 1736.1 | 2093.6 | 315.2 | 251.8 | 299.1 | -11.7 | 40.0 | plateau |
| unloaded | rss_anon | yes | 434 | 1420.0 | 1506.5 | 1510.5 | 1326.6 | 1581.2 | 157.5 | 125.8 | 217.0 | 4.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 434 | 335.0 | 304.2 | 335.0 | 301.9 | 400.0 | 7.8 | 6.2 | 14.7 | -6.7 | 32.0 | plateau |
| loaded | working_set | yes | 434 | 2280.2 | 2371.2 | 2374.8 | 2186.8 | 2718.4 | 121.5 | 97.1 | 167.8 | -29.5 | 47.5 | plateau |
| loaded | private | no | 434 | 2836.6 | 3012.3 | 3047.9 | 2776.1 | 3595.7 | 248.1 | 198.3 | 216.3 | -76.8 | 61.0 | plateau |
| loaded | rss_anon | yes | 434 | 2205.6 | 2296.1 | 2299.6 | 2112.2 | 2643.4 | 120.8 | 96.5 | 167.3 | -29.5 | 46.0 | plateau |

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
| git_commit | ba816c33548410a5920014b5fb010799fd6a4f3a |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38822628830, variant no-heap-trim\u002Bsmaps --no-heap-trim --smaps-every 25 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38822628830, variant no-heap-trim\u002Bsmaps |
| load before (loadavg, MemAvailable MB) | [2.99,0.83,0.28], 14054 |
| load after (loadavg, MemAvailable MB) | [5.55,5.63,5.43], 12937 |
