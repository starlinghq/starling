# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded working_set retention 41 MB between loads, reused by the next load; unloaded rss_anon retention 40 MB between loads, reused by the next load)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3602 s |
| Load/unload cycles | 178 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 534; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3154, 3683 |
| Unload ms (median, max) | 106.4, 131.5 |
| Request p50 ms (median over cycles) | 8955.0 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 178 | 0 | 5.54 | 60.48 | 84.00 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 146 | 199.2 | 199.6 | 199.9 | 198.2 | 216.6 | 5.0 | 4.0 | 1.0 | -0.5 | 32.0 | plateau |
| unloaded | working_set | yes | 146 | 480.1 | 542.3 | 529.3 | 450.1 | 585.5 | 51.9 | 41.4 | 33.0 | 51.1 | 32.0 | retention |
| unloaded | private | no | 146 | 2613.7 | 2663.6 | 2662.8 | 2583.1 | 2712.7 | 41.7 | 33.3 | 37.9 | 42.7 | 53.3 | plateau |
| unloaded | rss_anon | yes | 146 | 404.8 | 465.9 | 453.2 | 374.6 | 509.1 | 50.5 | 40.4 | 32.1 | 51.0 | 32.0 | retention |
| loaded | managed_heap | yes | 146 | 277.9 | 278.3 | 278.7 | 276.9 | 295.3 | 5.0 | 4.0 | 1.0 | -0.5 | 32.0 | plateau |
| loaded | working_set | yes | 146 | 2887.6 | 2946.0 | 2917.2 | 2832.0 | 2994.7 | 28.9 | 23.1 | 37.3 | 52.6 | 58.3 | plateau |
| loaded | private | no | 146 | 3197.5 | 3231.4 | 3217.6 | 3130.7 | 3310.8 | 14.0 | 11.2 | 34.5 | 48.4 | 64.4 | plateau |
| loaded | rss_anon | yes | 146 | 2812.4 | 2869.6 | 2840.8 | 2755.6 | 2918.5 | 27.6 | 22.0 | 36.4 | 52.5 | 56.8 | plateau |

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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [3.22,1.37,0.52], 14084 |
| load after (loadavg, MemAvailable MB) | [2.7,2.73,2.71], 14059 |
