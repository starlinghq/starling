# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 640 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 34560; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1161, 1899 |
| Unload ms (median, max) | 19.3, 50.0 |
| Request p50 ms (median over cycles) | 88.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3840 | 0 | 499.37 | 679.16 | 1041.58 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 512 | 255.9 | 223.6 | 256.4 | 223.2 | 290.8 | -7.8 | -6.2 | -13.0 | 18.7 | 32.0 | plateau |
| unloaded | working_set | yes | 512 | 487.1 | 562.7 | 568.1 | 449.2 | 670.4 | 28.7 | 23.0 | 110.0 | -7.4 | 32.0 | plateau |
| unloaded | private | no | 512 | 1747.0 | 1893.4 | 1919.7 | 1714.4 | 1973.9 | 201.0 | 160.7 | 207.5 | 18.5 | 38.4 | retention |
| unloaded | rss_anon | yes | 512 | 412.5 | 487.5 | 492.9 | 374.6 | 595.5 | 28.0 | 22.4 | 109.4 | -7.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 512 | 334.6 | 302.3 | 335.1 | 301.9 | 369.6 | -7.8 | -6.2 | -13.0 | 18.7 | 32.0 | plateau |
| loaded | working_set | yes | 512 | 2453.4 | 2278.3 | 2279.3 | 2145.9 | 2644.6 | 72.4 | 57.9 | 136.6 | -37.8 | 45.6 | plateau |
| loaded | private | no | 512 | 3323.2 | 2957.5 | 2989.3 | 2779.9 | 3564.1 | 237.9 | 190.3 | 240.5 | -14.7 | 59.8 | plateau |
| loaded | rss_anon | yes | 512 | 2378.8 | 2203.1 | 2204.1 | 2071.3 | 2569.7 | 71.7 | 57.3 | 136.1 | -37.8 | 44.1 | plateau |

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
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38822628830, variant thp-always\u002Bsmaps --smaps-every 25 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38822628830, variant thp-always\u002Bsmaps |
| load before (loadavg, MemAvailable MB) | [3.26,1.05,0.37], 14241 |
| load after (loadavg, MemAvailable MB) | [5.97,5.85,5.61], 14270 |
