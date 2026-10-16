# G4 soak: embeddinggemma-2/video-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** FAIL: loaded working_set grows 121 MB (600 MB/h); loaded rss_anon grows 119 MB (589 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 914 s |
| Load/unload cycles | 38 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 114; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3379, 3507 |
| Unload ms (median, max) | 186.4, 209.7 |
| Request p50 ms (median over cycles) | 10108.3 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 38 | 0 | 11.98 | 44.12 | 65.83 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 34 | 205.0 | 214.8 | 199.6 | 198.2 | 215.8 | 16.2 | 3.3 | 2.1 | 11.7 | 32.0 | plateau |
| unloaded | working_set | yes | 34 | 431.0 | 477.4 | 484.2 | 416.9 | 504.7 | 346.2 | 69.7 | 63.8 | -15.6 | 32.0 | plateau |
| unloaded | private | no | 34 | 676.2 | 706.8 | 714.4 | 661.4 | 749.4 | 313.8 | 63.2 | 59.3 | -10.4 | 32.0 | plateau |
| unloaded | rss_anon | yes | 34 | 357.7 | 401.9 | 408.7 | 343.2 | 430.1 | 336.0 | 67.6 | 62.4 | -16.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 34 | 283.7 | 293.5 | 278.3 | 277.0 | 294.5 | 16.2 | 3.3 | 2.1 | 11.7 | 32.0 | plateau |
| loaded | working_set | yes | 34 | 3323.1 | 3439.1 | 3519.4 | 3223.4 | 3706.5 | 599.9 | 120.7 | 208.0 | 25.6 | 70.4 | growth |
| loaded | private | no | 34 | 4789.1 | 4836.0 | 4835.4 | 4775.3 | 4894.3 | 342.3 | 68.9 | 66.1 | -9.9 | 96.7 | plateau |
| loaded | rss_anon | yes | 34 | 3249.9 | 3363.6 | 3444.7 | 3149.8 | 3631.8 | 588.9 | 118.5 | 206.4 | 24.9 | 68.9 | growth |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"true","DOTNET_MULTILEVEL_LOOKUP":"0","DOTNET_NOLOGO":"true","DOTNET_ROOT_X64":"/usr/share/dotnet","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | 9a4d2ce14573a1e1000bedb5de865a622db469de |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant mmap64k\u002Bsmaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant mmap64k\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [3.64,1.22,0.44], 14172 |
| load after (loadavg, MemAvailable MB) | [6.61,5.97,3.97], 14360 |
