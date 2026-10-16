# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3606 s |
| Load/unload cycles | 503 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 27162; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2109, 3007 |
| Unload ms (median, max) | 110.5, 122.5 |
| Request p50 ms (median over cycles) | 94.4 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3018 | 0 | 602.03 | 833.60 | 1169.26 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 402 | 289.0 | 288.8 | 256.3 | 223.2 | 290.8 | 18.2 | 14.6 | -1.6 | 5.2 | 32.0 | plateau |
| unloaded | working_set | yes | 402 | 468.5 | 493.6 | 493.7 | 458.4 | 502.8 | 17.0 | 13.6 | 16.7 | -2.3 | 32.0 | plateau |
| unloaded | private | no | 402 | 1669.9 | 1698.6 | 1670.0 | 1598.8 | 1720.6 | 31.7 | 25.4 | 11.8 | 3.4 | 33.4 | plateau |
| unloaded | rss_anon | yes | 402 | 393.9 | 418.5 | 418.7 | 383.8 | 427.6 | 16.2 | 13.0 | 16.2 | -2.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 402 | 367.7 | 367.5 | 335.1 | 302.0 | 369.5 | 17.7 | 14.2 | -1.2 | 5.2 | 32.0 | plateau |
| loaded | working_set | yes | 402 | 1864.9 | 1893.3 | 1899.8 | 1856.4 | 1922.7 | 22.2 | 17.8 | 22.8 | -3.3 | 38.0 | plateau |
| loaded | private | no | 402 | 2222.1 | 2250.7 | 2235.6 | 2160.9 | 2292.7 | 26.7 | 21.4 | 9.2 | 1.2 | 44.7 | plateau |
| loaded | rss_anon | yes | 402 | 1790.4 | 1818.2 | 1824.9 | 1781.9 | 1847.5 | 21.4 | 17.1 | 22.2 | -3.3 | 36.5 | plateau |

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
| git_commit | 9c972ca9e46d1614d819ecd497de352cd32cd759 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38892944927 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38892944927 |
| load before (loadavg, MemAvailable MB) | [2.49,0.7,0.24], 14085 |
| load after (loadavg, MemAvailable MB) | [5.71,5.5,5.2], 14204 |
