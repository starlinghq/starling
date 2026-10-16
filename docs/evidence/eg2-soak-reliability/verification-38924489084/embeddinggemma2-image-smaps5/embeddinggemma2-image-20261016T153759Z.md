# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3619 s |
| Load/unload cycles | 172 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 688; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1923, 2500 |
| Unload ms (median, max) | 165.2, 222.3 |
| Request p50 ms (median over cycles) | 9076.9 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 172 | 0 | 43.75 | 111.84 | 143.09 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 140 | 181.3 | 182.7 | 190.1 | 180.6 | 193.0 | 1.8 | 1.4 | -1.2 | -1.0 | 32.0 | plateau |
| unloaded | working_set | yes | 140 | 485.2 | 478.3 | 469.9 | 440.9 | 500.9 | -13.7 | -10.9 | -34.6 | 14.4 | 32.0 | plateau |
| unloaded | private | no | 140 | 2512.6 | 2508.8 | 2501.6 | 2472.8 | 2572.0 | -14.4 | -11.5 | -31.5 | 9.9 | 50.0 | plateau |
| unloaded | rss_anon | yes | 140 | 410.2 | 402.4 | 394.1 | 365.0 | 425.4 | -14.8 | -11.8 | -35.6 | 14.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 140 | 260.0 | 261.4 | 268.9 | 259.3 | 271.7 | 1.7 | 1.4 | -1.2 | -1.0 | 32.0 | plateau |
| loaded | working_set | yes | 140 | 2748.7 | 2745.9 | 2741.1 | 2692.2 | 2814.0 | -20.7 | -16.6 | -37.0 | 18.9 | 54.8 | plateau |
| loaded | private | no | 140 | 3040.7 | 3036.9 | 3034.7 | 2984.9 | 3111.8 | -17.5 | -14.0 | -30.5 | 10.9 | 60.7 | plateau |
| loaded | rss_anon | yes | 140 | 2673.7 | 2670.0 | 2665.3 | 2616.4 | 2738.2 | -21.9 | -17.5 | -37.9 | 18.9 | 53.3 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [1.59,0.54,0.2], 14126 |
| load after (loadavg, MemAvailable MB) | [2.81,2.82,2.82], 13618 |
