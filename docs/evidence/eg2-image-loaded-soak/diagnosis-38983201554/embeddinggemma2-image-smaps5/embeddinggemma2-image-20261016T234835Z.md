# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3620 s |
| Load/unload cycles | 88 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 352; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2927, 3909 |
| Unload ms (median, max) | 177.7, 210.5 |
| Request p50 ms (median over cycles) | 18000.1 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 88 | 0 | 15.33 | 73.24 | 188.76 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 73 | 182.0 | 182.2 | 181.6 | 180.1 | 182.6 | 0.7 | 0.5 | -0.1 | 0.6 | 32.0 | plateau |
| unloaded | working_set | yes | 73 | 426.7 | 385.3 | 431.0 | 374.2 | 499.8 | -6.8 | -5.5 | 1.8 | -36.8 | 32.0 | plateau |
| unloaded | private | no | 73 | 2407.5 | 2529.7 | 2506.2 | 2403.3 | 2622.3 | 113.8 | 90.9 | 31.7 | -4.1 | 50.1 | plateau |
| unloaded | rss_anon | yes | 73 | 352.1 | 309.1 | 354.8 | 298.6 | 424.2 | -8.8 | -7.1 | 0.6 | -37.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 73 | 260.0 | 260.9 | 260.0 | 258.8 | 261.4 | 0.7 | 0.6 | -0.0 | 0.6 | 32.0 | plateau |
| loaded | working_set | yes | 73 | 2668.0 | 2725.6 | 2741.3 | 2651.7 | 2829.7 | 99.0 | 79.1 | 39.5 | -14.2 | 54.8 | plateau |
| loaded | private | no | 73 | 3023.7 | 3121.9 | 3105.9 | 3019.3 | 3222.4 | 117.8 | 94.1 | 52.4 | -6.5 | 62.1 | plateau |
| loaded | rss_anon | yes | 73 | 2593.5 | 2649.3 | 2665.1 | 2576.2 | 2753.5 | 97.0 | 77.5 | 38.3 | -14.8 | 53.3 | plateau |

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
| git_commit | a5a0f374012118a3ba0a764018448dac111797a9 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38983201554, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38983201554, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [3.89,1.28,0.46], 14070 |
| load after (loadavg, MemAvailable MB) | [6.69,6.38,6.19], 14387 |
