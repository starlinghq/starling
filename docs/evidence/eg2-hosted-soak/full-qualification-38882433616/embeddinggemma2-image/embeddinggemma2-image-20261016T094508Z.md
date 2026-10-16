# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 557 MB; loaded private (address space, report-only) grows 576 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3610 s |
| Load/unload cycles | 87 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 348; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2989, 3866 |
| Unload ms (median, max) | 176.3, 193.2 |
| Request p50 ms (median over cycles) | 18093.3 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 87 | 0 | 9.93 | 52.67 | 164.30 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 72 | 181.8 | 181.7 | 181.3 | 180.1 | 182.6 | 0.9 | 0.7 | 0.2 | 0.5 | 32.0 | plateau |
| unloaded | working_set | yes | 72 | 459.1 | 502.1 | 538.4 | 459.1 | 642.4 | 84.1 | 66.7 | 55.8 | -4.5 | 32.0 | plateau |
| unloaded | private | no | 72 | 2766.4 | 3149.0 | 2956.5 | 2643.7 | 3311.9 | 701.8 | 556.5 | 338.0 | 230.6 | 59.1 | growth |
| unloaded | rss_anon | yes | 72 | 385.0 | 426.0 | 462.9 | 385.0 | 566.4 | 81.8 | 64.8 | 54.6 | -5.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 72 | 260.5 | 260.4 | 260.0 | 258.8 | 261.4 | 0.9 | 0.7 | 0.2 | 0.5 | 32.0 | plateau |
| loaded | working_set | yes | 72 | 2804.2 | 2863.2 | 2783.5 | 2651.3 | 3013.4 | 230.2 | 182.5 | 22.2 | -8.8 | 55.7 | plateau |
| loaded | private | no | 72 | 3399.5 | 3749.1 | 3586.4 | 3242.8 | 3912.1 | 726.1 | 575.9 | 344.6 | 218.0 | 71.7 | growth |
| loaded | rss_anon | yes | 72 | 2730.1 | 2787.2 | 2708.5 | 2576.2 | 2937.4 | 227.9 | 180.7 | 21.0 | -9.4 | 54.2 | plateau |

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
| git_commit | 9c972ca9e46d1614d819ecd497de352cd32cd759 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38882433616 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38882433616 |
| load before (loadavg, MemAvailable MB) | [1.99,0.63,0.22], 14234 |
| load after (loadavg, MemAvailable MB) | [6.84,6.53,6.47], 14043 |
