# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3613 s |
| Load/unload cycles | 131 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 393; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2955, 3834 |
| Unload ms (median, max) | 170.3, 176.7 |
| Request p50 ms (median over cycles) | 12807.8 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 131 | 0 | 17.42 | 59.33 | 89.36 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 109 | 198.4 | 215.9 | 199.7 | 197.7 | 216.6 | -0.6 | -0.4 | 0.4 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 109 | 576.7 | 515.7 | 522.3 | 487.4 | 576.7 | -38.0 | -30.5 | -10.4 | -17.6 | 32.0 | plateau |
| unloaded | private | no | 109 | 2599.5 | 2555.9 | 2558.8 | 2510.5 | 2627.2 | -26.6 | -21.3 | 0.7 | -23.3 | 51.2 | plateau |
| unloaded | rss_anon | yes | 109 | 502.1 | 439.6 | 446.4 | 411.2 | 502.1 | -39.8 | -31.9 | -11.6 | -17.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 109 | 277.1 | 294.6 | 278.4 | 276.5 | 295.3 | -0.6 | -0.4 | 0.4 | 0.4 | 32.0 | plateau |
| loaded | working_set | yes | 109 | 2829.9 | 2811.4 | 2806.3 | 2747.9 | 2890.8 | -17.8 | -14.3 | -4.2 | -9.5 | 56.1 | plateau |
| loaded | private | no | 109 | 3191.7 | 3159.9 | 3165.2 | 3087.2 | 3253.9 | -19.7 | -15.8 | 1.6 | -27.2 | 63.3 | plateau |
| loaded | rss_anon | yes | 109 | 2755.1 | 2735.2 | 2730.5 | 2671.7 | 2814.6 | -19.5 | -15.6 | -5.4 | -9.8 | 54.6 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [5.49,1.52,0.52], 14265 |
| load after (loadavg, MemAvailable MB) | [6.1,6.34,6.27], 14353 |
