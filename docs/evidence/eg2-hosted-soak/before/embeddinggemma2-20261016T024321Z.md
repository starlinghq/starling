# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** FAIL: unloaded working_set grows 103 MB (129 MB/h); unloaded rss_anon grows 103 MB (128 MB/h); loaded working_set grows 114 MB (142 MB/h); loaded rss_anon grows 113 MB (142 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3602 s |
| Load/unload cycles | 512 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 27648; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2146, 2938 |
| Unload ms (median, max) | 123.9, 146.9 |
| Request p50 ms (median over cycles) | 94.4 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3072 | 0 | 521.19 | 738.93 | 1101.56 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 410 | 256.4 | 256.1 | 256.9 | 223.2 | 290.7 | 7.2 | 5.8 | -13.7 | 5.7 | 32.0 | plateau |
| unloaded | working_set | yes | 410 | 482.6 | 656.4 | 579.8 | 471.1 | 669.1 | 128.9 | 103.1 | 90.6 | 22.3 | 32.0 | growth |
| unloaded | private | no | 410 | 1785.2 | 1984.7 | 1912.1 | 1750.3 | 2034.5 | 207.9 | 166.3 | 129.2 | 79.6 | 38.2 | growth |
| unloaded | rss_anon | yes | 410 | 408.1 | 581.3 | 504.7 | 396.6 | 594.0 | 128.1 | 102.5 | 90.0 | 22.3 | 32.0 | growth |
| loaded | managed_heap | yes | 410 | 335.1 | 334.9 | 335.6 | 301.9 | 369.4 | 7.2 | 5.8 | -13.7 | 5.7 | 32.0 | plateau |
| loaded | working_set | yes | 410 | 2161.7 | 2375.0 | 2294.0 | 2156.4 | 2665.3 | 142.4 | 114.0 | 147.0 | 14.1 | 45.9 | growth |
| loaded | private | no | 410 | 2849.4 | 3048.8 | 2982.4 | 2820.4 | 3621.2 | 232.6 | 186.1 | 209.8 | 68.1 | 59.6 | growth |
| loaded | rss_anon | yes | 410 | 2087.2 | 2299.9 | 2218.9 | 2081.9 | 2590.2 | 141.6 | 113.3 | 146.4 | 14.1 | 44.4 | growth |

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
| git_commit | a71c89b9f8a8aa39156b2172cdffd19f23d4e51c |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38841680114 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38841680114 |
| load before (loadavg, MemAvailable MB) | [1.61,0.7,0.26], 14224 |
| load after (loadavg, MemAvailable MB) | [4.48,4.84,4.86], 14055 |
