# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3621 s |
| Load/unload cycles | 103 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 412; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2380, 3074 |
| Unload ms (median, max) | 153.8, 166.6 |
| Request p50 ms (median over cycles) | 15492.5 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 103 | 0 | 9.08 | 106.94 | 158.60 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 85 | 181.0 | 181.2 | 181.2 | 180.1 | 182.7 | 0.4 | 0.3 | 0.3 | 0.2 | 32.0 | plateau |
| unloaded | working_set | yes | 85 | 446.9 | 448.8 | 453.6 | 414.9 | 479.4 | 17.9 | 14.3 | 12.4 | -4.0 | 32.0 | plateau |
| unloaded | private | no | 85 | 2530.5 | 2538.2 | 2553.0 | 2504.2 | 2608.8 | 20.8 | 16.7 | 30.5 | -14.9 | 51.1 | plateau |
| unloaded | rss_anon | yes | 85 | 372.3 | 372.4 | 377.3 | 338.6 | 403.0 | 16.0 | 12.8 | 11.5 | -4.1 | 32.0 | plateau |
| loaded | managed_heap | yes | 85 | 259.8 | 259.9 | 259.9 | 258.8 | 261.4 | 0.4 | 0.3 | 0.3 | 0.2 | 32.0 | plateau |
| loaded | working_set | yes | 85 | 2783.7 | 2770.0 | 2790.7 | 2740.3 | 2897.5 | 21.1 | 16.9 | 10.9 | -9.2 | 55.8 | plateau |
| loaded | private | no | 85 | 3138.6 | 3114.3 | 3153.2 | 3100.6 | 3251.9 | 16.0 | 12.8 | 19.6 | -18.3 | 63.1 | plateau |
| loaded | rss_anon | yes | 85 | 2709.1 | 2693.6 | 2715.1 | 2665.3 | 2821.2 | 18.9 | 15.1 | 9.7 | -9.3 | 54.3 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38933567784, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [3.52,1.18,0.42], 14242 |
| load after (loadavg, MemAvailable MB) | [6.56,6.63,6.38], 14190 |
