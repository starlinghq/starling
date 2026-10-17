# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** FAIL: unloaded working_set grows 62 MB (77 MB/h); unloaded rss_anon grows 61 MB (76 MB/h); loaded working_set grows 71 MB (88 MB/h); loaded rss_anon grows 69 MB (87 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3618 s |
| Load/unload cycles | 129 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 387; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3150, 4100 |
| Unload ms (median, max) | 131.2, 142.9 |
| Request p50 ms (median over cycles) | 12982.7 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 129 | 0 | 14.41 | 51.26 | 64.07 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 107 | 199.2 | 199.0 | 199.7 | 198.5 | 217.4 | 1.4 | 1.1 | 4.4 | -4.7 | 32.0 | plateau |
| unloaded | working_set | yes | 107 | 522.9 | 603.6 | 566.5 | 490.3 | 613.6 | 77.3 | 61.9 | 55.9 | 30.2 | 32.0 | growth |
| unloaded | private | no | 107 | 2501.2 | 2606.8 | 2555.5 | 2463.7 | 2641.3 | 84.9 | 68.0 | 69.1 | 48.9 | 51.1 | growth |
| unloaded | rss_anon | yes | 107 | 448.1 | 527.3 | 491.1 | 414.0 | 537.3 | 75.6 | 60.5 | 54.8 | 29.9 | 32.0 | growth |
| loaded | managed_heap | yes | 107 | 277.9 | 277.8 | 278.4 | 277.0 | 296.1 | 1.4 | 1.2 | 4.4 | -4.6 | 32.0 | plateau |
| loaded | working_set | yes | 107 | 2216.5 | 2297.2 | 2262.1 | 2193.0 | 2307.2 | 88.4 | 70.8 | 54.8 | 35.9 | 45.2 | growth |
| loaded | private | no | 107 | 3093.2 | 3198.8 | 3155.0 | 3066.0 | 3217.3 | 92.4 | 74.0 | 66.4 | 51.4 | 63.1 | growth |
| loaded | rss_anon | yes | 107 | 2141.6 | 2220.9 | 2186.0 | 2118.2 | 2230.9 | 86.7 | 69.4 | 53.6 | 35.6 | 43.7 | growth |

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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant default\u002Bsmaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant default\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [3.71,1.03,0.36], 14176 |
| load after (loadavg, MemAvailable MB) | [6.82,6.52,6.24], 14243 |
