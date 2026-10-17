# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3614 s |
| Load/unload cycles | 179 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 537; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3087, 3763 |
| Unload ms (median, max) | 80.8, 86.3 |
| Request p50 ms (median over cycles) | 8783.6 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 179 | 0 | 6.53 | 26.10 | 62.05 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 148 | 200.6 | 199.6 | 200.3 | 198.4 | 216.6 | -0.6 | -0.5 | 1.9 | -1.3 | 32.0 | plateau |
| unloaded | working_set | yes | 148 | 519.7 | 499.2 | 500.6 | 471.3 | 559.1 | -0.9 | -0.7 | -30.0 | 10.8 | 32.0 | plateau |
| unloaded | private | no | 148 | 2527.0 | 2478.3 | 2472.2 | 2426.5 | 2561.3 | -13.9 | -11.1 | -37.9 | 0.3 | 49.4 | plateau |
| unloaded | rss_anon | yes | 148 | 444.5 | 422.8 | 425.0 | 395.2 | 482.7 | -2.3 | -1.8 | -31.0 | 10.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 148 | 279.3 | 278.3 | 278.8 | 277.2 | 295.3 | -0.6 | -0.5 | 1.9 | -1.3 | 32.0 | plateau |
| loaded | working_set | yes | 148 | 2212.8 | 2192.2 | 2193.6 | 2164.3 | 2252.1 | -0.3 | -0.3 | -30.4 | 11.1 | 43.9 | plateau |
| loaded | private | no | 148 | 3039.0 | 2990.3 | 3002.8 | 2952.2 | 3090.6 | -22.5 | -18.1 | -38.8 | -9.9 | 60.1 | plateau |
| loaded | rss_anon | yes | 148 | 2137.6 | 2115.8 | 2118.1 | 2088.2 | 2175.8 | -1.7 | -1.3 | -31.3 | 10.8 | 42.4 | plateau |

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
| git_commit | d5798090b13dddada74fe920a81bd346f7dfd974 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [2.68,0.9,0.33], 14062 |
| load after (loadavg, MemAvailable MB) | [2.75,2.87,2.79], 14201 |
