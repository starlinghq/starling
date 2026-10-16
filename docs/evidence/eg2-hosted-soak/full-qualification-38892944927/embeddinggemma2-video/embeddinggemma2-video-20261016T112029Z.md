# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 248 MB; loaded private (address space, report-only) grows 253 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3624 s |
| Load/unload cycles | 130 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 390; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2959, 3981 |
| Unload ms (median, max) | 167.6, 177.5 |
| Request p50 ms (median over cycles) | 13013.6 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 130 | 0 | 17.15 | 57.97 | 88.57 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 108 | 215.0 | 198.8 | 199.8 | 197.7 | 216.4 | 1.7 | 1.3 | -4.7 | 4.9 | 32.0 | plateau |
| unloaded | working_set | yes | 108 | 533.2 | 550.3 | 563.7 | 514.8 | 629.3 | -25.4 | -20.4 | 0.7 | 20.1 | 32.0 | plateau |
| unloaded | private | no | 108 | 2816.7 | 3140.8 | 3050.7 | 2816.7 | 3241.5 | 308.3 | 247.5 | 169.7 | 198.3 | 61.0 | growth |
| unloaded | rss_anon | yes | 108 | 458.9 | 474.3 | 488.3 | 439.1 | 554.0 | -27.2 | -21.8 | -0.5 | 19.7 | 32.0 | plateau |
| loaded | managed_heap | yes | 108 | 293.7 | 277.5 | 278.5 | 276.4 | 295.1 | 1.7 | 1.3 | -4.7 | 4.9 | 32.0 | plateau |
| loaded | working_set | yes | 108 | 2802.4 | 2804.3 | 2804.6 | 2723.0 | 2963.5 | -68.8 | -55.2 | -64.7 | 6.2 | 56.1 | plateau |
| loaded | private | no | 108 | 3392.8 | 3768.0 | 3657.9 | 3392.8 | 3853.4 | 315.1 | 252.9 | 165.1 | 183.5 | 73.2 | growth |
| loaded | rss_anon | yes | 108 | 2728.0 | 2728.3 | 2728.9 | 2647.3 | 2888.4 | -70.4 | -56.5 | -65.8 | 5.8 | 54.6 | plateau |

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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38892944927 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38892944927 |
| load before (loadavg, MemAvailable MB) | [3.61,0.92,0.31], 14070 |
| load after (loadavg, MemAvailable MB) | [6.69,6.42,6.29], 13950 |
