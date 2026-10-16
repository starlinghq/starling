# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3617 s |
| Load/unload cycles | 174 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 696; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1855, 2538 |
| Unload ms (median, max) | 164.9, 285.3 |
| Request p50 ms (median over cycles) | 9012.0 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 174 | 0 | 45.83 | 110.95 | 142.76 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 143 | 192.1 | 192.8 | 191.1 | 180.1 | 200.5 | -0.3 | -0.3 | -4.5 | 0.9 | 32.0 | plateau |
| unloaded | working_set | yes | 143 | 493.8 | 454.4 | 450.5 | 400.2 | 535.7 | 4.6 | 3.7 | -8.4 | -9.0 | 32.0 | plateau |
| unloaded | private | no | 143 | 2558.0 | 2510.0 | 2498.4 | 2435.2 | 2589.1 | -4.5 | -3.6 | -18.4 | -6.2 | 50.0 | plateau |
| unloaded | rss_anon | yes | 143 | 418.7 | 378.3 | 374.6 | 324.4 | 460.6 | 3.4 | 2.7 | -9.3 | -9.1 | 32.0 | plateau |
| loaded | managed_heap | yes | 143 | 270.8 | 271.5 | 269.9 | 258.8 | 279.2 | -0.3 | -0.3 | -4.5 | 0.9 | 32.0 | plateau |
| loaded | working_set | yes | 143 | 2784.6 | 2811.8 | 2748.8 | 2678.5 | 2856.6 | 13.4 | 10.8 | -32.7 | 6.0 | 55.0 | plateau |
| loaded | private | no | 143 | 3092.3 | 3079.7 | 3040.2 | 2979.8 | 3136.1 | -1.4 | -1.1 | -37.3 | 7.3 | 60.8 | plateau |
| loaded | rss_anon | yes | 143 | 2709.5 | 2735.8 | 2672.7 | 2602.7 | 2780.5 | 12.3 | 9.8 | -33.6 | 5.9 | 53.5 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V45 96-Core Processor |
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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084 |
| load before (loadavg, MemAvailable MB) | [2.91,0.94,0.33], 14178 |
| load after (loadavg, MemAvailable MB) | [3.09,2.9,2.86], 14161 |
