# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 634 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 34236; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1138, 1906 |
| Unload ms (median, max) | 19.4, 49.0 |
| Request p50 ms (median over cycles) | 92.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3804 | 0 | 497.13 | 692.37 | 946.20 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 509 | 289.0 | 290.8 | 258.1 | 223.3 | 290.8 | -9.5 | -7.6 | 8.2 | -13.3 | 32.0 | plateau |
| unloaded | working_set | yes | 509 | 470.0 | 582.6 | 578.6 | 466.4 | 620.0 | 56.4 | 45.1 | 84.5 | 3.9 | 32.0 | plateau |
| unloaded | private | no | 509 | 1792.5 | 1963.8 | 1933.7 | 1742.7 | 1987.6 | 209.3 | 167.5 | 232.1 | -18.9 | 38.7 | plateau |
| unloaded | rss_anon | yes | 509 | 395.5 | 507.6 | 503.6 | 391.9 | 545.3 | 55.6 | 44.5 | 83.9 | 3.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 509 | 367.7 | 369.5 | 336.8 | 302.0 | 369.5 | -9.5 | -7.6 | 8.2 | -13.3 | 32.0 | plateau |
| loaded | working_set | yes | 509 | 2446.6 | 2263.7 | 2270.3 | 2151.6 | 2617.0 | 42.7 | 34.2 | 133.2 | -16.0 | 45.4 | plateau |
| loaded | private | no | 509 | 3368.6 | 3029.9 | 3000.4 | 2806.9 | 3563.2 | 176.4 | 141.2 | 305.8 | -59.9 | 60.0 | plateau |
| loaded | rss_anon | yes | 509 | 2372.2 | 2188.7 | 2195.3 | 2077.2 | 2542.3 | 41.9 | 33.5 | 132.6 | -16.0 | 43.9 | plateau |

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
| git_commit | aeaf1b07cad8b65cf657bb3e136ee3d2124ef35d |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38809646407, variant no-env |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38809646407, variant no-env |
| load before (loadavg, MemAvailable MB) | [1.71,0.85,0.32], 14411 |
| load after (loadavg, MemAvailable MB) | [6.25,5.75,5.66], 14322 |
