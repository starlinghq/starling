# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 3633 s |
| Load/unload cycles | 88 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 352; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3143, 4139 |
| Unload ms (median, max) | 136.4, 145.6 |
| Request p50 ms (median over cycles) | 18039.3 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 88 | 0 | 17.03 | 47.73 | 136.60 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 73 | 159.4 | 160.2 | 159.4 | 159.1 | 160.2 | 0.7 | 0.6 | 0.2 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 73 | 385.1 | 396.1 | 393.2 | 379.1 | 432.7 | -5.9 | -4.7 | 1.3 | -5.7 | 32.0 | plateau |
| unloaded | private | no | 73 | 2409.2 | 2480.9 | 2481.7 | 2402.8 | 2528.2 | 86.3 | 69.0 | 84.6 | -12.1 | 49.6 | plateau |
| unloaded | rss_anon | yes | 73 | 311.0 | 320.3 | 318.0 | 304.9 | 358.1 | -7.8 | -6.2 | 0.3 | -6.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 73 | 283.0 | 283.9 | 277.4 | 276.3 | 283.9 | -1.5 | -1.2 | -1.5 | -1.0 | 32.0 | plateau |
| loaded | working_set | yes | 73 | 2092.7 | 2136.0 | 2100.6 | 2087.0 | 2139.7 | 1.3 | 1.0 | 2.0 | 3.5 | 42.0 | plateau |
| loaded | private | no | 73 | 3024.3 | 3120.6 | 3094.3 | 3010.2 | 3129.5 | 90.5 | 72.4 | 84.2 | -11.9 | 61.9 | plateau |
| loaded | rss_anon | yes | 73 | 2018.5 | 2060.2 | 2025.4 | 2012.8 | 2065.2 | -0.7 | -0.6 | 0.9 | 2.9 | 40.5 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 39006453578 |
| load before (loadavg, MemAvailable MB) | [2.85,0.94,0.34], 14211 |
| load after (loadavg, MemAvailable MB) | [6.41,6.18,6.3], 14384 |
