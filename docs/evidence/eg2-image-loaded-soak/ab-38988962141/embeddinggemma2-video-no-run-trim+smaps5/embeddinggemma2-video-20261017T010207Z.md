# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=off; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3604 s |
| Load/unload cycles | 146 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 438; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2654, 3292 |
| Unload ms (median, max) | 134.4, 146.1 |
| Request p50 ms (median over cycles) | 11547.2 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 146 | 0 | 9.34 | 57.26 | 68.03 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 120 | 199.7 | 199.8 | 200.1 | 198.2 | 216.9 | 2.8 | 2.2 | 6.4 | 4.9 | 32.0 | plateau |
| unloaded | working_set | yes | 120 | 550.5 | 522.5 | 535.5 | 478.8 | 558.7 | -27.4 | -21.8 | -31.0 | -22.7 | 32.0 | plateau |
| unloaded | private | no | 120 | 2627.0 | 2615.1 | 2643.6 | 2587.4 | 2684.3 | -31.7 | -25.2 | -27.8 | -15.6 | 52.9 | plateau |
| unloaded | rss_anon | yes | 120 | 475.5 | 446.0 | 459.2 | 402.3 | 482.4 | -28.8 | -22.9 | -31.9 | -22.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 120 | 278.4 | 278.0 | 278.7 | 276.9 | 295.6 | 2.8 | 2.2 | 6.5 | 4.9 | 32.0 | plateau |
| loaded | working_set | yes | 120 | 2542.6 | 2489.5 | 2536.1 | 2444.8 | 2602.6 | -64.8 | -51.5 | -31.0 | -45.9 | 50.7 | plateau |
| loaded | private | no | 120 | 3262.9 | 3208.2 | 3259.9 | 3163.5 | 3318.2 | -60.8 | -48.3 | -32.5 | -37.1 | 65.2 | plateau |
| loaded | rss_anon | yes | 120 | 2467.6 | 2413.0 | 2460.4 | 2368.3 | 2526.3 | -66.3 | -52.6 | -31.9 | -46.1 | 49.2 | plateau |

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
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant no-run-trim\u002Bsmaps5 --no-run-trim --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38988962141, variant no-run-trim\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [3.17,1.14,0.43], 14280 |
| load after (loadavg, MemAvailable MB) | [6.89,6.55,6.29], 14343 |
