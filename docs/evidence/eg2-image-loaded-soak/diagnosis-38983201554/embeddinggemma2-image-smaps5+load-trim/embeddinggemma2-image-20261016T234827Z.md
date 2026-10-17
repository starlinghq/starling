# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded working_set retention 34 MB between loads, reused by the next load; unloaded rss_anon retention 33 MB between loads, reused by the next load)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3626 s |
| Load/unload cycles | 89 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 356; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2933, 3790 |
| Unload ms (median, max) | 154.6, 184.8 |
| Request p50 ms (median over cycles) | 17833.9 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 89 | 0 | 11.22 | 57.23 | 217.97 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 74 | 182.0 | 181.2 | 181.6 | 180.1 | 182.6 | 0.2 | 0.1 | 0.1 | 0.2 | 32.0 | plateau |
| unloaded | working_set | yes | 74 | 429.1 | 460.6 | 434.9 | 401.1 | 479.9 | 43.1 | 34.5 | 40.1 | 29.6 | 32.0 | retention |
| unloaded | private | no | 74 | 2557.4 | 2561.9 | 2527.9 | 2469.2 | 2603.9 | 40.1 | 32.1 | 17.2 | 33.1 | 50.6 | plateau |
| unloaded | rss_anon | yes | 74 | 354.5 | 384.3 | 359.8 | 325.7 | 404.2 | 41.1 | 32.9 | 39.0 | 29.0 | 32.0 | retention |
| loaded | managed_heap | yes | 74 | 260.5 | 259.9 | 260.3 | 258.8 | 261.4 | 0.2 | 0.1 | 0.1 | 0.2 | 32.0 | plateau |
| loaded | working_set | yes | 74 | 2448.9 | 2438.4 | 2443.5 | 2366.2 | 2542.1 | 38.2 | 30.6 | 36.1 | -1.2 | 48.9 | plateau |
| loaded | private | no | 74 | 3149.4 | 3138.1 | 3144.2 | 3066.4 | 3224.0 | 53.5 | 42.8 | 37.5 | 8.5 | 62.9 | plateau |
| loaded | rss_anon | yes | 74 | 2374.5 | 2362.2 | 2367.8 | 2290.0 | 2466.4 | 36.3 | 29.0 | 35.1 | -1.8 | 47.4 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38983201554, variant smaps5\u002Bload-trim --smaps-every 5 --trim-after-load |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38983201554, variant smaps5\u002Bload-trim |
| load before (loadavg, MemAvailable MB) | [2.96,0.92,0.32], 14226 |
| load after (loadavg, MemAvailable MB) | [6.19,6.44,6.39], 14311 |
