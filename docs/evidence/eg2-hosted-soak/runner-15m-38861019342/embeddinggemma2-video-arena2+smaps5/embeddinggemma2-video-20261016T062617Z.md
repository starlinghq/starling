# G4 soak: embeddinggemma-2/video-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 902 s |
| Load/unload cycles | 34 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 102; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2986, 3817 |
| Unload ms (median, max) | 239.9, 256.2 |
| Request p50 ms (median over cycles) | 11399.1 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 34 | 0 | 9.85 | 42.27 | 50.14 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 31 | 214.5 | 214.5 | 199.6 | 198.1 | 215.3 | -19.6 | -3.9 | 0.5 | 4.6 | 32.0 | plateau |
| unloaded | working_set | yes | 31 | 445.7 | 522.0 | 533.0 | 445.7 | 557.4 | 84.5 | 16.8 | 32.4 | -5.8 | 32.0 | plateau |
| unloaded | private | no | 31 | 2274.4 | 2387.6 | 2404.5 | 2274.4 | 2453.4 | 376.1 | 74.9 | 78.6 | 14.0 | 48.1 | retention |
| unloaded | rss_anon | yes | 31 | 372.9 | 446.8 | 458.8 | 372.9 | 483.9 | 72.8 | 14.5 | 30.9 | -6.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 31 | 293.2 | 293.2 | 278.4 | 276.9 | 294.0 | -19.6 | -3.9 | 0.5 | 4.6 | 32.0 | plateau |
| loaded | working_set | yes | 31 | 3821.3 | 3886.5 | 3865.0 | 3664.6 | 4028.0 | -21.0 | -4.2 | -57.0 | 133.5 | 77.3 | plateau |
| loaded | private | no | 31 | 5106.6 | 5211.8 | 5234.5 | 5106.6 | 5260.6 | 174.9 | 34.8 | 62.6 | -12.9 | 104.7 | plateau |
| loaded | rss_anon | yes | 31 | 3748.7 | 3811.3 | 3791.5 | 3590.3 | 3954.3 | -31.9 | -6.4 | -58.6 | 132.6 | 75.8 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"true","DOTNET_MULTILEVEL_LOOKUP":"0","DOTNET_NOLOGO":"true","DOTNET_ROOT_X64":"/usr/share/dotnet","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1","MALLOC_ARENA_MAX":"2"} |
| git_commit | 9a4d2ce14573a1e1000bedb5de865a622db469de |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant arena2\u002Bsmaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant arena2\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [1.64,0.59,0.21], 14212 |
| load after (loadavg, MemAvailable MB) | [6.72,6.22,4.15], 14375 |
