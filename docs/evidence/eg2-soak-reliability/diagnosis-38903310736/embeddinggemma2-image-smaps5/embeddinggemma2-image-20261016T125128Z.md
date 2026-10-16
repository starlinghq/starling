# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** FAIL: unloaded working_set grows 166 MB (208 MB/h); unloaded rss_anon grows 165 MB (206 MB/h); loaded working_set grows 78 MB (97 MB/h); loaded rss_anon grows 76 MB (95 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3628 s |
| Load/unload cycles | 89 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 356; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2923, 3842 |
| Unload ms (median, max) | 175.9, 192.4 |
| Request p50 ms (median over cycles) | 17846.3 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 89 | 0 | 9.55 | 44.85 | 74.74 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 74 | 181.3 | 181.2 | 181.2 | 180.1 | 182.6 | 0.4 | 0.3 | -0.2 | 0.2 | 32.0 | plateau |
| unloaded | working_set | yes | 74 | 453.0 | 605.2 | 514.2 | 430.0 | 642.9 | 207.6 | 166.3 | 42.6 | 103.6 | 32.0 | growth |
| unloaded | private | no | 74 | 2640.1 | 2991.6 | 2984.6 | 2612.0 | 3114.0 | 394.7 | 316.2 | 420.1 | 28.9 | 59.7 | growth |
| unloaded | rss_anon | yes | 74 | 378.6 | 529.0 | 439.0 | 355.0 | 566.8 | 205.7 | 164.8 | 41.5 | 103.0 | 32.0 | growth |
| loaded | managed_heap | yes | 74 | 260.0 | 259.9 | 259.9 | 258.8 | 261.4 | 0.4 | 0.3 | -0.2 | 0.2 | 32.0 | plateau |
| loaded | working_set | yes | 74 | 2813.5 | 2835.2 | 2812.7 | 2669.7 | 2963.5 | 97.2 | 77.8 | 79.3 | 115.6 | 56.3 | growth |
| loaded | private | no | 74 | 3313.1 | 3583.7 | 3590.6 | 3269.5 | 3739.8 | 378.4 | 303.2 | 421.1 | 20.4 | 71.8 | growth |
| loaded | rss_anon | yes | 74 | 2739.2 | 2759.1 | 2737.5 | 2594.1 | 2887.3 | 95.2 | 76.3 | 78.2 | 115.0 | 54.7 | growth |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [2.69,0.8,0.28], 14110 |
| load after (loadavg, MemAvailable MB) | [6.95,6.58,6.46], 14106 |
