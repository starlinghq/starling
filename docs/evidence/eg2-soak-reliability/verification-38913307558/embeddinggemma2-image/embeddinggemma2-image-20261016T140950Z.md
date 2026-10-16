# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded working_set retention 44 MB between loads, reused by the next load; unloaded rss_anon retention 43 MB between loads, reused by the next load)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3615 s |
| Load/unload cycles | 88 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 352; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2974, 3890 |
| Unload ms (median, max) | 178.8, 189.8 |
| Request p50 ms (median over cycles) | 17964.3 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 88 | 0 | 7.73 | 55.56 | 178.15 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 73 | 181.3 | 182.2 | 181.3 | 179.1 | 182.6 | 0.3 | 0.3 | 0.1 | 0.2 | 32.0 | plateau |
| unloaded | working_set | yes | 73 | 453.3 | 416.9 | 442.3 | 384.0 | 507.8 | 55.7 | 44.3 | 72.8 | 14.1 | 32.0 | retention |
| unloaded | private | no | 73 | 2520.0 | 2483.7 | 2501.4 | 2432.6 | 2603.5 | 57.5 | 45.7 | 57.3 | -6.4 | 50.0 | plateau |
| unloaded | rss_anon | yes | 73 | 379.1 | 341.0 | 366.4 | 309.4 | 431.9 | 53.8 | 42.8 | 71.7 | 13.5 | 32.0 | retention |
| loaded | managed_heap | yes | 73 | 260.0 | 260.9 | 260.0 | 258.8 | 261.4 | 0.3 | 0.3 | 0.2 | 0.2 | 32.0 | plateau |
| loaded | working_set | yes | 73 | 2793.9 | 2700.6 | 2770.6 | 2655.0 | 2882.4 | 58.0 | 46.1 | 49.8 | -27.7 | 55.4 | plateau |
| loaded | private | no | 73 | 3151.8 | 3075.8 | 3138.1 | 3032.8 | 3254.4 | 70.0 | 55.7 | 60.1 | -23.7 | 62.8 | plateau |
| loaded | rss_anon | yes | 73 | 2719.8 | 2624.7 | 2695.0 | 2580.7 | 2806.6 | 56.0 | 44.5 | 48.8 | -28.3 | 53.9 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558 |
| load before (loadavg, MemAvailable MB) | [2.17,0.84,0.31], 14181 |
| load after (loadavg, MemAvailable MB) | [6.88,6.82,6.61], 13924 |
