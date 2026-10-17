# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3626 s |
| Load/unload cycles | 102 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 408; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2568, 3278 |
| Unload ms (median, max) | 118.4, 127.6 |
| Request p50 ms (median over cycles) | 15613.9 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 102 | 0 | 18.07 | 60.00 | 164.07 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 84 | 180.0 | 182.2 | 181.3 | 180.0 | 182.7 | 1.1 | 0.9 | 0.5 | 0.5 | 32.0 | plateau |
| unloaded | working_set | yes | 84 | 445.5 | 458.3 | 448.5 | 389.9 | 545.9 | -23.5 | -18.7 | -42.2 | -31.1 | 32.0 | plateau |
| unloaded | private | no | 84 | 2486.7 | 2582.2 | 2525.6 | 2469.4 | 2611.2 | 20.7 | 16.5 | -10.4 | -30.3 | 50.5 | plateau |
| unloaded | rss_anon | yes | 84 | 370.9 | 382.1 | 372.6 | 314.3 | 471.1 | -25.0 | -19.9 | -42.8 | -31.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 84 | 258.8 | 260.9 | 260.0 | 258.8 | 261.4 | 1.1 | 0.9 | 0.5 | 0.5 | 32.0 | plateau |
| loaded | working_set | yes | 84 | 2146.3 | 2152.0 | 2143.4 | 2083.8 | 2239.8 | -20.8 | -16.6 | -37.9 | -32.0 | 42.9 | plateau |
| loaded | private | no | 84 | 3069.9 | 3158.2 | 3113.9 | 3064.1 | 3234.1 | 34.2 | 27.3 | 11.4 | -27.0 | 62.3 | plateau |
| loaded | rss_anon | yes | 84 | 2071.7 | 2075.8 | 2067.7 | 2008.2 | 2164.9 | -22.6 | -18.0 | -38.8 | -32.0 | 41.4 | plateau |

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
| git_commit | d5798090b13dddada74fe920a81bd346f7dfd974 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141 |
| load before (loadavg, MemAvailable MB) | [2.37,0.84,0.3], 14309 |
| load after (loadavg, MemAvailable MB) | [5.89,6.16,6.13], 14031 |
