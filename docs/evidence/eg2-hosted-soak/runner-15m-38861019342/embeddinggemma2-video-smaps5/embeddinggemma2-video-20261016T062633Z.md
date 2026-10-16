# G4 soak: embeddinggemma-2/video-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 915 s |
| Load/unload cycles | 51 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 153; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2970, 3626 |
| Unload ms (median, max) | 132.1, 142.8 |
| Request p50 ms (median over cycles) | 7344.1 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 51 | 0 | 7.88 | 28.93 | 41.49 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 44 | 199.1 | 199.7 | 199.5 | 198.6 | 215.5 | -7.7 | -1.5 | 1.7 | 4.1 | 32.0 | plateau |
| unloaded | working_set | yes | 44 | 538.7 | 576.1 | 532.7 | 494.4 | 576.1 | 49.6 | 9.9 | -13.6 | 6.0 | 32.0 | plateau |
| unloaded | private | no | 44 | 2504.7 | 2612.6 | 2570.2 | 2504.7 | 2619.3 | 224.5 | 44.8 | 65.7 | -26.3 | 51.4 | plateau |
| unloaded | rss_anon | yes | 44 | 465.5 | 500.7 | 457.3 | 419.1 | 500.7 | 39.7 | 7.9 | -15.1 | 5.6 | 32.0 | plateau |
| loaded | managed_heap | yes | 44 | 277.9 | 278.4 | 278.2 | 277.3 | 294.2 | -7.7 | -1.5 | 1.7 | 4.1 | 32.0 | plateau |
| loaded | working_set | yes | 44 | 3970.1 | 4102.1 | 4076.7 | 3532.5 | 4373.0 | 672.3 | 134.1 | 168.5 | -93.6 | 81.5 | plateau |
| loaded | private | no | 44 | 5263.0 | 5364.6 | 5349.1 | 5263.0 | 5399.8 | 247.3 | 49.3 | 69.8 | -23.5 | 107.0 | plateau |
| loaded | rss_anon | yes | 44 | 3896.9 | 4026.6 | 4001.5 | 3457.3 | 4297.9 | 662.2 | 132.1 | 167.1 | -94.1 | 80.0 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | INTEL(R) XEON(R) PLATINUM 8573C |
| logical_cpus | 4 |
| ram_total_mb | 15989 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.17.0-1022-azure |
| runtime | .NET 10.0.13 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI/0.17.1 |
| gpu_identity | n/a |
| cpu_governor | performance |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"true","DOTNET_MULTILEVEL_LOOKUP":"0","DOTNET_NOLOGO":"true","DOTNET_ROOT_X64":"/usr/share/dotnet","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1"} |
| git_commit | 9a4d2ce14573a1e1000bedb5de865a622db469de |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [3.97,1.54,0.57], 14067 |
| load after (loadavg, MemAvailable MB) | [3.1,2.9,2.03], 14268 |
