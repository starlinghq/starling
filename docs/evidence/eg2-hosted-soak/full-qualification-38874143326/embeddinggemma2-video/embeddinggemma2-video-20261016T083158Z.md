# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 413 MB; loaded private (address space, report-only) grows 404 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3612 s |
| Load/unload cycles | 130 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 390; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2968, 3917 |
| Unload ms (median, max) | 170.0, 182.9 |
| Request p50 ms (median over cycles) | 12954.7 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 130 | 0 | 14.96 | 54.52 | 69.18 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 108 | 215.2 | 200.0 | 199.9 | 197.9 | 216.6 | 4.2 | 3.4 | -0.3 | 1.3 | 32.0 | plateau |
| unloaded | working_set | yes | 108 | 574.2 | 571.4 | 561.1 | 479.9 | 635.7 | 52.6 | 42.2 | 20.0 | 3.9 | 32.0 | plateau |
| unloaded | private | no | 108 | 2833.1 | 3300.3 | 3097.2 | 2815.1 | 3319.5 | 514.2 | 412.5 | 243.7 | 197.0 | 61.9 | growth |
| unloaded | rss_anon | yes | 108 | 499.7 | 495.3 | 485.3 | 404.7 | 559.6 | 50.8 | 40.8 | 18.8 | 3.6 | 32.0 | plateau |
| loaded | managed_heap | yes | 108 | 294.0 | 278.8 | 278.6 | 276.7 | 295.3 | 4.2 | 3.4 | -0.3 | 1.3 | 32.0 | plateau |
| loaded | working_set | yes | 108 | 2831.9 | 2894.3 | 2829.8 | 2748.8 | 2927.8 | 42.0 | 33.7 | 9.2 | 20.1 | 56.6 | plateau |
| loaded | private | no | 108 | 3425.2 | 3932.1 | 3700.8 | 3425.2 | 3932.1 | 503.1 | 403.6 | 235.5 | 198.9 | 74.0 | growth |
| loaded | rss_anon | yes | 108 | 2757.4 | 2818.2 | 2754.0 | 2673.7 | 2851.5 | 40.3 | 32.4 | 8.1 | 19.7 | 55.1 | plateau |

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
| git_commit | 6df29602d9f84e48e7ce5a1a6c8b8f96c4fbf50c |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38874143326 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38874143326 |
| load before (loadavg, MemAvailable MB) | [3.35,1.21,0.44], 14163 |
| load after (loadavg, MemAvailable MB) | [5.85,6.33,6.22], 14256 |
