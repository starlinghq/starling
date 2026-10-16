# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 482 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 26028; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2222, 3105 |
| Unload ms (median, max) | 114.3, 136.1 |
| Request p50 ms (median over cycles) | 97.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 2892 | 0 | 619.52 | 853.37 | 1249.16 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 387 | 223.7 | 290.7 | 256.1 | 223.3 | 290.8 | -5.0 | -4.0 | -11.5 | 20.7 | 32.0 | plateau |
| unloaded | working_set | yes | 387 | 460.7 | 482.3 | 470.3 | 442.7 | 486.9 | 18.7 | 14.9 | -7.8 | 9.0 | 32.0 | plateau |
| unloaded | private | no | 387 | 1591.1 | 1721.6 | 1660.7 | 1561.9 | 1739.6 | 45.4 | 36.3 | 30.5 | 31.7 | 33.2 | retention |
| unloaded | rss_anon | yes | 387 | 386.0 | 407.0 | 395.0 | 367.9 | 412.2 | 17.9 | 14.3 | -8.3 | 9.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 387 | 302.4 | 369.4 | 334.8 | 302.0 | 369.5 | -4.1 | -3.3 | -11.4 | 20.4 | 32.0 | plateau |
| loaded | working_set | yes | 387 | 1877.3 | 1894.9 | 1883.0 | 1848.2 | 1913.4 | 23.8 | 19.0 | -7.5 | 15.7 | 37.7 | plateau |
| loaded | private | no | 387 | 2183.2 | 2314.3 | 2224.0 | 2155.2 | 2314.3 | 14.2 | 11.4 | -11.4 | 36.5 | 44.5 | plateau |
| loaded | rss_anon | yes | 387 | 1802.6 | 1819.5 | 1807.7 | 1773.1 | 1838.6 | 23.0 | 18.4 | -8.0 | 15.7 | 36.2 | plateau |

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
| git_commit | f43a648a795ed6cda366007a61fd7b4158a93082 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant default\u002Bsmaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant default\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [3.11,0.86,0.3], 14122 |
| load after (loadavg, MemAvailable MB) | [4.47,5.1,5.02], 14201 |
