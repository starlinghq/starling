# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 279 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 837; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1930, 2564 |
| Unload ms (median, max) | 119.5, 187.0 |
| Request p50 ms (median over cycles) | 5617.6 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 279 | 0 | 12.07 | 56.60 | 72.40 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 228 | 199.7 | 217.4 | 200.2 | 198.2 | 232.3 | 2.6 | 2.0 | 3.7 | 1.6 | 32.0 | plateau |
| unloaded | working_set | yes | 228 | 493.8 | 561.3 | 520.1 | 443.9 | 587.3 | -2.8 | -2.2 | -28.6 | -64.9 | 32.0 | plateau |
| unloaded | private | no | 228 | 2454.9 | 2533.1 | 2505.7 | 2411.8 | 2594.1 | 2.1 | 1.7 | -27.2 | -61.7 | 50.1 | plateau |
| unloaded | rss_anon | yes | 228 | 418.4 | 485.0 | 444.0 | 368.0 | 511.5 | -3.7 | -3.0 | -29.3 | -65.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 228 | 278.4 | 296.2 | 279.0 | 276.9 | 311.0 | 2.6 | 2.0 | 3.7 | 1.6 | 32.0 | plateau |
| loaded | working_set | yes | 228 | 2187.5 | 2254.6 | 2216.6 | 2167.0 | 2308.8 | -21.1 | -16.9 | -16.3 | -66.2 | 44.3 | plateau |
| loaded | private | no | 228 | 2982.9 | 3045.1 | 3028.1 | 2956.6 | 3122.1 | -21.7 | -17.4 | -13.4 | -63.0 | 60.6 | plateau |
| loaded | rss_anon | yes | 228 | 2112.0 | 2178.3 | 2140.5 | 2090.7 | 2232.5 | -22.0 | -17.6 | -16.9 | -66.2 | 42.8 | plateau |

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
| git_commit | d5798090b13dddada74fe920a81bd346f7dfd974 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141 |
| load before (loadavg, MemAvailable MB) | [2.9,0.78,0.27], 14099 |
| load after (loadavg, MemAvailable MB) | [2.91,2.8,2.73], 14088 |
