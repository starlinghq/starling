# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded working_set retention 43 MB between loads, reused by the next load; unloaded rss_anon retention 42 MB between loads, reused by the next load)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3607 s |
| Load/unload cycles | 87 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 348; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3093, 4012 |
| Unload ms (median, max) | 135.4, 148.2 |
| Request p50 ms (median over cycles) | 18077.3 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 87 | 0 | 17.94 | 59.87 | 84.11 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 72 | 182.0 | 181.0 | 181.6 | 180.1 | 182.6 | 0.3 | 0.2 | -0.7 | -0.0 | 32.0 | plateau |
| unloaded | working_set | yes | 72 | 425.0 | 446.7 | 447.0 | 392.9 | 544.8 | 54.5 | 43.3 | 14.8 | 25.7 | 32.0 | retention |
| unloaded | private | no | 72 | 2396.0 | 2429.9 | 2444.2 | 2360.6 | 2548.2 | 67.9 | 53.9 | 20.0 | 41.7 | 48.9 | retention |
| unloaded | rss_anon | yes | 72 | 350.5 | 370.2 | 370.9 | 317.1 | 468.4 | 52.4 | 41.6 | 13.5 | 25.1 | 32.0 | retention |
| loaded | managed_heap | yes | 72 | 260.5 | 259.7 | 260.0 | 258.8 | 261.4 | 0.2 | 0.2 | -0.8 | -0.1 | 32.0 | plateau |
| loaded | working_set | yes | 72 | 2118.3 | 2140.2 | 2140.6 | 2086.6 | 2238.4 | 49.2 | 39.0 | 19.1 | 20.0 | 42.8 | plateau |
| loaded | private | no | 72 | 2988.0 | 3062.0 | 3043.5 | 2988.0 | 3141.6 | 72.7 | 57.7 | 36.6 | 38.0 | 60.9 | plateau |
| loaded | rss_anon | yes | 72 | 2043.8 | 2063.8 | 2064.4 | 2010.8 | 2162.0 | 47.0 | 37.3 | 17.8 | 19.2 | 41.3 | plateau |

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
| git_commit | d5798090b13dddada74fe920a81bd346f7dfd974 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant default\u002Bsmaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant default\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [2.48,0.7,0.24], 14201 |
| load after (loadavg, MemAvailable MB) | [7.22,6.69,6.37], 14021 |
