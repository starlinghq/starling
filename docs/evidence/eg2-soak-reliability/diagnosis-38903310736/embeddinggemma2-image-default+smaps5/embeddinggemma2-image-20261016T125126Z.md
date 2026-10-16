# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 397 MB; loaded private (address space, report-only) grows 384 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3618 s |
| Load/unload cycles | 87 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 348; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3002, 3808 |
| Unload ms (median, max) | 176.5, 190.1 |
| Request p50 ms (median over cycles) | 18203.1 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 87 | 0 | 10.60 | 46.96 | 63.18 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 72 | 180.8 | 181.2 | 181.2 | 180.1 | 182.6 | 0.5 | 0.4 | 0.2 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 72 | 473.4 | 467.8 | 491.4 | 377.4 | 561.4 | -4.1 | -3.2 | 56.7 | -46.7 | 32.0 | plateau |
| unloaded | private | no | 72 | 2724.9 | 3065.0 | 2870.1 | 2646.4 | 3120.0 | 499.8 | 396.7 | 73.0 | 244.5 | 57.4 | growth |
| unloaded | rss_anon | yes | 72 | 398.9 | 391.5 | 415.9 | 302.6 | 485.8 | -6.1 | -4.9 | 55.5 | -47.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 72 | 259.5 | 259.9 | 259.9 | 258.8 | 261.4 | 0.5 | 0.4 | 0.2 | 0.4 | 32.0 | plateau |
| loaded | working_set | yes | 72 | 2881.0 | 2722.4 | 2717.8 | 2604.8 | 2895.1 | -34.3 | -27.2 | -124.3 | 32.9 | 54.4 | plateau |
| loaded | private | no | 72 | 3325.1 | 3645.6 | 3468.2 | 3240.7 | 3746.7 | 484.4 | 384.5 | 68.6 | 231.0 | 69.4 | growth |
| loaded | rss_anon | yes | 72 | 2806.5 | 2645.9 | 2642.2 | 2529.9 | 2820.6 | -36.3 | -28.8 | -125.5 | 32.2 | 52.8 | plateau |

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
| git_commit | f43a648a795ed6cda366007a61fd7b4158a93082 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant default\u002Bsmaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant default\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [4.15,1.27,0.44], 13942 |
| load after (loadavg, MemAvailable MB) | [6.99,6.53,6.27], 14115 |
