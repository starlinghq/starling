# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3602 s |
| Load/unload cycles | 192 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 576; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2903, 3608 |
| Unload ms (median, max) | 99.2, 105.7 |
| Request p50 ms (median over cycles) | 8318.9 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 192 | 0 | 6.03 | 49.92 | 74.66 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 158 | 200.4 | 198.8 | 200.0 | 198.2 | 216.5 | -1.4 | -1.1 | 3.3 | 1.7 | 32.0 | plateau |
| unloaded | working_set | yes | 158 | 455.0 | 493.4 | 500.6 | 449.3 | 525.0 | 63.9 | 50.9 | 49.5 | 3.2 | 32.0 | plateau |
| unloaded | private | no | 158 | 2601.9 | 2612.3 | 2622.4 | 2558.4 | 2656.1 | 49.9 | 39.7 | 42.1 | -0.9 | 52.4 | plateau |
| unloaded | rss_anon | yes | 158 | 380.1 | 417.4 | 424.5 | 374.1 | 449.0 | 62.6 | 49.9 | 48.7 | 3.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 158 | 279.1 | 277.5 | 278.8 | 276.9 | 295.3 | -1.4 | -1.1 | 3.3 | 1.7 | 32.0 | plateau |
| loaded | working_set | yes | 158 | 2835.1 | 2906.4 | 2877.1 | 2790.5 | 2938.4 | 60.3 | 48.0 | 40.9 | -9.8 | 57.5 | plateau |
| loaded | private | no | 158 | 3148.8 | 3180.2 | 3172.4 | 3091.8 | 3219.3 | 38.4 | 30.6 | 30.1 | -15.1 | 63.4 | plateau |
| loaded | rss_anon | yes | 158 | 2760.2 | 2830.4 | 2801.5 | 2715.3 | 2862.3 | 59.0 | 47.0 | 40.1 | -10.0 | 56.0 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558 |
| load before (loadavg, MemAvailable MB) | [1.91,0.67,0.24], 14036 |
| load after (loadavg, MemAvailable MB) | [3.13,2.92,2.79], 14245 |
