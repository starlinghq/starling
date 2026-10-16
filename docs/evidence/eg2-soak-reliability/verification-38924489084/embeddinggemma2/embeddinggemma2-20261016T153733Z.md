# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 500 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 27000; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2128, 3053 |
| Unload ms (median, max) | 108.6, 128.2 |
| Request p50 ms (median over cycles) | 94.9 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3000 | 0 | 597.69 | 828.07 | 1163.84 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 402 | 256.4 | 288.7 | 256.3 | 223.2 | 290.8 | 13.9 | 11.1 | 16.4 | 18.4 | 32.0 | plateau |
| unloaded | working_set | yes | 402 | 422.4 | 398.2 | 400.1 | 384.1 | 422.4 | 7.3 | 5.8 | 2.7 | 2.0 | 32.0 | plateau |
| unloaded | private | no | 402 | 1654.1 | 1651.0 | 1622.9 | 1573.2 | 1694.8 | 16.0 | 12.8 | 15.3 | 21.3 | 32.5 | plateau |
| unloaded | rss_anon | yes | 402 | 347.8 | 323.0 | 325.0 | 309.3 | 347.8 | 6.5 | 5.2 | 2.1 | 2.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 402 | 335.1 | 367.4 | 335.0 | 301.9 | 369.5 | 13.9 | 11.1 | 16.4 | 18.4 | 32.0 | plateau |
| loaded | working_set | yes | 402 | 1838.0 | 1815.6 | 1807.2 | 1783.6 | 1838.0 | 5.9 | 4.7 | 1.8 | -2.1 | 36.1 | plateau |
| loaded | private | no | 402 | 2206.2 | 2205.1 | 2175.8 | 2112.6 | 2262.9 | 8.9 | 7.1 | 8.8 | 18.5 | 43.5 | plateau |
| loaded | rss_anon | yes | 402 | 1763.5 | 1740.5 | 1732.3 | 1708.5 | 1763.5 | 5.1 | 4.1 | 1.3 | -2.2 | 34.6 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084 |
| load before (loadavg, MemAvailable MB) | [3.65,1.12,0.39], 13984 |
| load after (loadavg, MemAvailable MB) | [5.09,4.88,4.86], 14189 |
