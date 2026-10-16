# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3605 s |
| Load/unload cycles | 293 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 879; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1773, 2278 |
| Unload ms (median, max) | 151.8, 211.2 |
| Request p50 ms (median over cycles) | 5420.1 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 293 | 0 | 17.31 | 51.91 | 67.58 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 237 | 199.4 | 201.6 | 200.5 | 198.7 | 232.2 | 0.9 | 0.7 | -5.3 | 0.6 | 32.0 | plateau |
| unloaded | working_set | yes | 237 | 492.5 | 531.0 | 505.5 | 474.0 | 566.2 | -7.7 | -6.2 | -15.0 | 13.6 | 32.0 | plateau |
| unloaded | private | no | 237 | 2529.0 | 2597.4 | 2568.6 | 2524.4 | 2613.4 | 13.6 | 10.9 | 20.5 | 6.3 | 51.4 | plateau |
| unloaded | rss_anon | yes | 237 | 417.5 | 454.7 | 429.8 | 397.7 | 490.8 | -9.0 | -7.2 | -15.8 | 13.6 | 32.0 | plateau |
| loaded | managed_heap | yes | 237 | 278.1 | 280.3 | 279.2 | 277.5 | 311.0 | 0.9 | 0.7 | -5.3 | 0.6 | 32.0 | plateau |
| loaded | working_set | yes | 237 | 2797.8 | 2824.5 | 2804.4 | 2746.5 | 2864.5 | -8.4 | -6.7 | -0.0 | 0.4 | 56.1 | plateau |
| loaded | private | no | 237 | 3096.8 | 3125.5 | 3105.6 | 3050.7 | 3172.6 | -0.4 | -0.4 | 9.5 | -3.1 | 62.1 | plateau |
| loaded | rss_anon | yes | 237 | 2722.8 | 2748.2 | 2728.1 | 2670.2 | 2789.1 | -9.7 | -7.8 | -0.9 | 0.4 | 54.6 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V45 96-Core Processor |
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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084 |
| load before (loadavg, MemAvailable MB) | [2.03,0.71,0.26], 14208 |
| load after (loadavg, MemAvailable MB) | [2.62,2.8,2.7], 14106 |
