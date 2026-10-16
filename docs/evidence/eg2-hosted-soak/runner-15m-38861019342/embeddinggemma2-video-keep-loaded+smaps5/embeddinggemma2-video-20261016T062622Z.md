# G4 soak: embeddinggemma-2/video-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** FAIL: unloaded working_set grows 447 MB (2240 MB/h); unloaded rss_anon grows 446 MB (2234 MB/h); loaded working_set grows 447 MB (2241 MB/h); loaded rss_anon grows 446 MB (2233 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 913 s |
| Load/unload cycles | 45 |
| Keep loaded (diagnostic) | yes: one load, no unloads; the `unloaded` series is a second sample with the model loaded |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 135; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 0, 3118 |
| Unload ms (median, max) | 0.0, 0.0 |
| Request p50 ms (median over cycles) | 10048.5 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 45 | 0 | 10.10 | 56.32 | 71.83 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 40 | 136.1 | 120.2 | 123.8 | 119.4 | 137.3 | 21.6 | 4.3 | 1.1 | 1.6 | 32.0 | plateau |
| unloaded | working_set | yes | 40 | 3855.3 | 4291.2 | 4107.7 | 3853.6 | 4295.0 | 2240.5 | 447.2 | 332.4 | 262.5 | 82.2 | growth |
| unloaded | private | no | 40 | 4963.3 | 4983.9 | 4970.9 | 4941.7 | 4988.2 | 74.6 | 14.9 | -1.9 | 22.8 | 99.4 | plateau |
| unloaded | rss_anon | yes | 40 | 3782.4 | 4216.7 | 4033.8 | 3780.5 | 4220.6 | 2233.5 | 445.8 | 331.4 | 262.0 | 80.7 | growth |
| loaded | managed_heap | yes | 40 | 136.1 | 120.2 | 123.8 | 119.4 | 137.3 | 21.6 | 4.3 | 1.1 | 1.6 | 32.0 | plateau |
| loaded | working_set | yes | 40 | 3855.3 | 4291.2 | 4107.7 | 3853.6 | 4295.0 | 2240.5 | 447.2 | 332.3 | 262.5 | 82.2 | growth |
| loaded | private | no | 40 | 4963.3 | 4983.9 | 4970.9 | 4941.7 | 4988.2 | 74.5 | 14.9 | -2.0 | 22.8 | 99.4 | plateau |
| loaded | rss_anon | yes | 40 | 3782.4 | 4216.7 | 4033.8 | 3780.5 | 4220.6 | 2233.4 | 445.8 | 331.4 | 262.1 | 80.7 | growth |

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
| git_commit | 9a4d2ce14573a1e1000bedb5de865a622db469de |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant keep-loaded\u002Bsmaps5 --keep-loaded --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant keep-loaded\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [1.9,0.69,0.27], 14238 |
| load after (loadavg, MemAvailable MB) | [7.13,7.03,4.73], 14661 |
