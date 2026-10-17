# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** FAIL: loaded working_set grows 60 MB (75 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=off; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3604 s |
| Load/unload cycles | 86 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 344; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3139, 4187 |
| Unload ms (median, max) | 153.3, 186.3 |
| Request p50 ms (median over cycles) | 18115.9 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 86 | 0 | 10.64 | 69.36 | 165.80 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 72 | 181.5 | 182.7 | 181.7 | 180.2 | 182.7 | 0.4 | 0.3 | 0.1 | 0.1 | 32.0 | plateau |
| unloaded | working_set | yes | 72 | 471.7 | 488.7 | 465.7 | 417.8 | 544.0 | 14.2 | 11.4 | -24.8 | 35.1 | 32.0 | plateau |
| unloaded | private | no | 72 | 2449.3 | 2553.6 | 2511.3 | 2414.0 | 2619.7 | 155.4 | 124.3 | 103.9 | 12.0 | 50.2 | plateau |
| unloaded | rss_anon | yes | 72 | 397.7 | 412.5 | 390.6 | 341.7 | 467.8 | 12.0 | 9.6 | -25.9 | 34.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 72 | 260.3 | 260.9 | 260.3 | 258.9 | 261.4 | 0.4 | 0.3 | 0.2 | 0.2 | 32.0 | plateau |
| loaded | working_set | yes | 72 | 2421.3 | 2487.2 | 2391.1 | 2331.4 | 2574.6 | 74.6 | 59.7 | 30.5 | 12.0 | 47.8 | growth |
| loaded | private | no | 72 | 3106.2 | 3182.2 | 3118.1 | 2991.8 | 3294.0 | 165.4 | 132.2 | 97.4 | 8.8 | 62.4 | plateau |
| loaded | rss_anon | yes | 72 | 2347.2 | 2410.9 | 2316.0 | 2255.3 | 2499.2 | 72.1 | 57.7 | 29.2 | 11.2 | 46.3 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant no-run-trim\u002Bsmaps5 --no-run-trim --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant no-run-trim\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [1.9,0.62,0.22], 14193 |
| load after (loadavg, MemAvailable MB) | [6.43,6.15,6.06], 14035 |
