# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: loaded private (address space, report-only) grows 197 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3602 s |
| Load/unload cycles | 529 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 28566; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2017, 2904 |
| Unload ms (median, max) | 121.4, 144.9 |
| Request p50 ms (median over cycles) | 91.5 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3174 | 0 | 513.52 | 716.18 | 1010.97 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 424 | 288.9 | 225.5 | 256.1 | 223.2 | 290.7 | 8.8 | 7.1 | 4.2 | -2.5 | 32.0 | plateau |
| unloaded | working_set | yes | 424 | 644.7 | 606.6 | 595.7 | 505.5 | 644.7 | -15.0 | -12.0 | -70.5 | -13.3 | 32.0 | plateau |
| unloaded | private | no | 424 | 1901.6 | 1896.0 | 1918.1 | 1798.3 | 1980.6 | 135.5 | 108.5 | 88.6 | 4.6 | 38.4 | plateau |
| unloaded | rss_anon | yes | 424 | 569.8 | 531.2 | 520.6 | 430.4 | 569.9 | -15.7 | -12.5 | -71.0 | -13.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 424 | 367.6 | 304.2 | 334.8 | 301.9 | 369.5 | 8.8 | 7.1 | 4.2 | -2.5 | 32.0 | plateau |
| loaded | working_set | yes | 424 | 2322.1 | 2289.1 | 2279.3 | 2178.6 | 2598.9 | 46.2 | 37.0 | -24.2 | 32.5 | 45.6 | plateau |
| loaded | private | no | 424 | 2965.7 | 2960.1 | 2988.1 | 2866.4 | 3553.4 | 245.8 | 196.7 | 171.0 | 80.5 | 59.8 | growth |
| loaded | rss_anon | yes | 424 | 2247.2 | 2213.6 | 2204.0 | 2103.4 | 2523.5 | 45.5 | 36.4 | -24.8 | 32.5 | 44.1 | plateau |

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
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38822628830, variant smaps --smaps-every 25 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38822628830, variant smaps |
| load before (loadavg, MemAvailable MB) | [3.25,1.01,0.35], 14103 |
| load after (loadavg, MemAvailable MB) | [5.23,5.1,5.01], 14237 |
