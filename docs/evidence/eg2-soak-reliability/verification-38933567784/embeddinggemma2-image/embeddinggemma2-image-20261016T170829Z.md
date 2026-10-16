# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3605 s |
| Load/unload cycles | 89 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 356; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2917, 3784 |
| Unload ms (median, max) | 176.7, 185.0 |
| Request p50 ms (median over cycles) | 17700.1 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 89 | 0 | 12.72 | 67.13 | 201.33 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 74 | 181.3 | 181.2 | 181.2 | 179.3 | 182.6 | 0.6 | 0.5 | 0.3 | 0.5 | 32.0 | plateau |
| unloaded | working_set | yes | 74 | 432.6 | 463.3 | 450.9 | 395.0 | 471.9 | 21.5 | 17.1 | 22.6 | 17.6 | 32.0 | plateau |
| unloaded | private | no | 74 | 2530.0 | 2581.5 | 2570.2 | 2484.3 | 2605.8 | 64.5 | 51.4 | 48.4 | 27.9 | 51.4 | plateau |
| unloaded | rss_anon | yes | 74 | 358.4 | 387.2 | 375.7 | 319.8 | 395.9 | 19.3 | 15.3 | 21.5 | 16.8 | 32.0 | plateau |
| loaded | managed_heap | yes | 74 | 260.0 | 259.9 | 259.9 | 258.8 | 261.4 | 0.6 | 0.4 | 0.3 | 0.5 | 32.0 | plateau |
| loaded | working_set | yes | 74 | 2758.2 | 2847.2 | 2832.3 | 2740.2 | 2885.1 | 68.4 | 54.4 | 79.5 | -3.2 | 56.6 | plateau |
| loaded | private | no | 74 | 3106.2 | 3227.6 | 3204.0 | 3106.2 | 3248.5 | 101.2 | 80.6 | 90.8 | 7.3 | 64.1 | plateau |
| loaded | rss_anon | yes | 74 | 2684.0 | 2771.1 | 2757.1 | 2664.2 | 2809.9 | 66.1 | 52.6 | 78.5 | -4.0 | 55.1 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784 |
| load before (loadavg, MemAvailable MB) | [4.05,1.28,0.45], 14014 |
| load after (loadavg, MemAvailable MB) | [6.82,6.56,6.39], 14286 |
