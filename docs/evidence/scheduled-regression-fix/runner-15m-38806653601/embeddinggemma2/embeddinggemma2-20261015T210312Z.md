# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 902 s |
| Load/unload cycles | 155 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 8370; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1188, 1913 |
| Unload ms (median, max) | 21.1, 50.4 |
| Request p50 ms (median over cycles) | 92.6 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 930 | 0 | 516.26 | 698.04 | 895.09 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 125 | 223.4 | 256.4 | 256.3 | 223.2 | 289.0 | 34.6 | 6.9 | -4.2 | 21.6 | 32.0 | plateau |
| unloaded | working_set | yes | 125 | 585.0 | 562.7 | 573.5 | 467.6 | 608.1 | -71.5 | -14.3 | 4.8 | -23.7 | 32.0 | plateau |
| unloaded | private | no | 125 | 1789.3 | 1797.4 | 1817.1 | 1752.8 | 1863.9 | -20.7 | -4.1 | 18.0 | -1.7 | 36.3 | plateau |
| unloaded | rss_anon | yes | 125 | 511.6 | 488.2 | 499.6 | 394.0 | 534.6 | -77.1 | -15.4 | 3.9 | -24.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 125 | 302.1 | 335.1 | 335.0 | 301.9 | 367.7 | 34.6 | 6.9 | -4.2 | 21.6 | 32.0 | plateau |
| loaded | working_set | yes | 125 | 2296.0 | 2247.7 | 2259.4 | 2179.8 | 2566.6 | -208.4 | -41.8 | -69.9 | 32.7 | 45.2 | plateau |
| loaded | private | no | 125 | 2871.4 | 2861.5 | 2885.5 | 2817.0 | 3426.7 | -209.5 | -42.0 | -101.6 | 91.9 | 57.7 | plateau |
| loaded | rss_anon | yes | 125 | 2222.6 | 2173.2 | 2185.3 | 2106.1 | 2493.1 | -214.0 | -42.9 | -70.8 | 32.4 | 43.7 | plateau |

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
| git_commit | ff58b212568b4d484c6f849b75b0094a2e597636 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601 |
| load before (loadavg, MemAvailable MB) | [1.71,0.59,0.21], 14367 |
| load after (loadavg, MemAvailable MB) | [6.12,5.65,3.75], 14313 |
