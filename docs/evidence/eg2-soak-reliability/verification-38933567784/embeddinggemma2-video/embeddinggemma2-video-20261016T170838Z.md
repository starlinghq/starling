# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3620 s |
| Load/unload cycles | 152 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 456; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2355, 3020 |
| Unload ms (median, max) | 141.4, 146.2 |
| Request p50 ms (median over cycles) | 11246.3 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 152 | 0 | 9.69 | 54.12 | 77.53 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 125 | 199.2 | 199.8 | 199.8 | 198.2 | 216.9 | -2.8 | -2.2 | -3.0 | 1.2 | 32.0 | plateau |
| unloaded | working_set | yes | 125 | 540.1 | 551.8 | 527.3 | 460.7 | 558.7 | 9.3 | 7.5 | 9.8 | 2.1 | 32.0 | plateau |
| unloaded | private | no | 125 | 2500.7 | 2513.2 | 2507.6 | 2438.5 | 2575.6 | 12.7 | 10.1 | 14.9 | 9.1 | 50.2 | plateau |
| unloaded | rss_anon | yes | 125 | 465.4 | 475.6 | 451.2 | 384.9 | 482.5 | 8.0 | 6.4 | 9.0 | 2.1 | 32.0 | plateau |
| loaded | managed_heap | yes | 125 | 277.9 | 278.5 | 278.5 | 276.9 | 295.6 | -2.8 | -2.2 | -3.0 | 1.2 | 32.0 | plateau |
| loaded | working_set | yes | 125 | 2785.4 | 2818.4 | 2783.6 | 2682.4 | 2839.9 | 22.5 | 18.0 | 12.0 | 11.9 | 55.7 | plateau |
| loaded | private | no | 125 | 3107.8 | 3180.2 | 3131.6 | 3030.6 | 3181.9 | 25.5 | 20.3 | 16.6 | 14.8 | 62.6 | plateau |
| loaded | rss_anon | yes | 125 | 2710.7 | 2742.2 | 2707.4 | 2607.1 | 2763.7 | 21.1 | 16.8 | 11.0 | 11.8 | 54.1 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V74 80-Core Processor |
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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784 |
| load before (loadavg, MemAvailable MB) | [2.07,0.88,0.35], 14237 |
| load after (loadavg, MemAvailable MB) | [7.09,7.05,6.6], 14346 |
