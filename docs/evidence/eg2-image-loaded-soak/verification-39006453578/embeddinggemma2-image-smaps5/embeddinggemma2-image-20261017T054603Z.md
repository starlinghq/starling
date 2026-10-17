# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3616 s |
| Load/unload cycles | 88 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 352; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3084, 3980 |
| Unload ms (median, max) | 132.8, 142.0 |
| Request p50 ms (median over cycles) | 17961.4 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 88 | 0 | 18.45 | 54.56 | 85.39 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 73 | 159.6 | 160.2 | 159.7 | 159.1 | 160.7 | 0.7 | 0.6 | 0.3 | 0.5 | 32.0 | plateau |
| unloaded | working_set | yes | 73 | 386.8 | 382.4 | 388.8 | 378.3 | 398.6 | 14.9 | 11.9 | 6.7 | 0.9 | 32.0 | plateau |
| unloaded | private | no | 73 | 2431.0 | 2488.3 | 2475.9 | 2389.6 | 2517.9 | 82.1 | 65.5 | 68.5 | -2.5 | 49.5 | plateau |
| unloaded | rss_anon | yes | 73 | 312.6 | 306.2 | 313.0 | 303.6 | 322.4 | 12.6 | 10.1 | 5.3 | 0.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 73 | 282.8 | 277.4 | 282.8 | 276.1 | 284.1 | 0.7 | 0.6 | 0.1 | -1.4 | 32.0 | plateau |
| loaded | working_set | yes | 73 | 2092.8 | 2090.5 | 2096.2 | 2085.4 | 2132.1 | 14.9 | 11.9 | 12.1 | -5.3 | 41.9 | plateau |
| loaded | private | no | 73 | 3035.6 | 3096.2 | 3087.2 | 3008.8 | 3131.4 | 87.1 | 69.5 | 78.1 | -6.2 | 61.7 | plateau |
| loaded | rss_anon | yes | 73 | 2018.6 | 2014.3 | 2021.0 | 2010.8 | 2056.5 | 12.7 | 10.1 | 10.8 | -6.0 | 40.4 | plateau |

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
| git_commit | 03570797591f1b66f32804bcdb8207aa58feb119 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [3.31,1,0.35], 14200 |
| load after (loadavg, MemAvailable MB) | [6.53,6.41,6.31], 14031 |
