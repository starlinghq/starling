# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 77 MB; loaded private (address space, report-only) grows 77 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 902 s |
| Load/unload cycles | 296 |
| Keep loaded (diagnostic) | yes: one load, no unloads; the `unloaded` series is a second sample with the model loaded |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 15984; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 0, 2337 |
| Unload ms (median, max) | 0.0, 0.0 |
| Request p50 ms (median over cycles) | 63.4 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 1776 | 0 | 335.67 | 436.68 | 495.10 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 239 | 209.7 | 209.8 | 209.7 | 144.5 | 210.8 | -146.7 | -29.4 | -12.0 | 21.2 | 32.0 | plateau |
| unloaded | working_set | yes | 239 | 1788.7 | 1865.2 | 1862.3 | 1788.7 | 1884.1 | 290.0 | 58.0 | 98.2 | -16.5 | 37.2 | plateau |
| unloaded | private | no | 239 | 2124.7 | 2224.8 | 2189.8 | 2101.3 | 2224.8 | 385.2 | 77.1 | 102.2 | 22.8 | 43.8 | growth |
| unloaded | rss_anon | yes | 239 | 1715.9 | 1791.5 | 1789.1 | 1715.9 | 1810.6 | 285.9 | 57.2 | 97.8 | -16.8 | 35.8 | plateau |
| loaded | managed_heap | yes | 239 | 209.7 | 209.8 | 209.7 | 144.5 | 210.8 | -146.7 | -29.4 | -12.0 | 21.2 | 32.0 | plateau |
| loaded | working_set | yes | 239 | 1788.7 | 1865.2 | 1862.3 | 1788.7 | 1884.1 | 290.0 | 58.0 | 98.2 | -16.5 | 37.2 | plateau |
| loaded | private | no | 239 | 2124.7 | 2224.8 | 2189.8 | 2101.3 | 2224.8 | 385.2 | 77.1 | 102.2 | 22.8 | 43.8 | growth |
| loaded | rss_anon | yes | 239 | 1715.9 | 1791.5 | 1789.1 | 1715.9 | 1810.6 | 285.9 | 57.2 | 97.8 | -16.8 | 35.8 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | INTEL(R) XEON(R) PLATINUM 8573C |
| logical_cpus | 4 |
| ram_total_mb | 15989 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.17.0-1022-azure |
| runtime | .NET 10.0.13 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI/0.17.1 |
| gpu_identity | n/a |
| cpu_governor | performance |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"true","DOTNET_MULTILEVEL_LOOKUP":"0","DOTNET_NOLOGO":"true","DOTNET_ROOT_X64":"/usr/share/dotnet","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1"} |
| git_commit | 8acc5069fce1a0c89865ce3b766fd3bfa71f1521 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant keep-loaded\u002Barena-shrink\u002Bsmaps5 --keep-loaded --arena-shrink --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant keep-loaded\u002Barena-shrink\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [1.94,0.8,0.3], 13980 |
| load after (loadavg, MemAvailable MB) | [2.98,2.84,1.96], 14390 |
