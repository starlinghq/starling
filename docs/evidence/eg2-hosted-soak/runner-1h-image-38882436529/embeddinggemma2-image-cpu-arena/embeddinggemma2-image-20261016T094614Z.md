# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** FAIL: unloaded working_set grows 152 MB (191 MB/h); unloaded rss_anon grows 151 MB (189 MB/h); loaded working_set grows 205 MB (257 MB/h); loaded rss_anon grows 203 MB (255 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=on, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=on, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3609 s |
| Load/unload cycles | 125 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 500; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2348, 3042 |
| Unload ms (median, max) | 263.3, 320.8 |
| Request p50 ms (median over cycles) | 12456.3 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 125 | 0 | 9.16 | 51.44 | 68.61 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 103 | 191.6 | 191.7 | 191.2 | 180.1 | 192.9 | 2.2 | 1.8 | 2.4 | 3.2 | 32.0 | plateau |
| unloaded | working_set | yes | 103 | 531.4 | 697.3 | 673.6 | 492.6 | 716.6 | 190.9 | 152.4 | 168.7 | 27.5 | 32.0 | growth |
| unloaded | private | no | 103 | 2662.4 | 3034.6 | 2998.4 | 2597.2 | 3078.1 | 418.7 | 334.3 | 375.6 | 16.0 | 60.0 | growth |
| unloaded | rss_anon | yes | 103 | 456.8 | 620.9 | 597.5 | 418.0 | 640.6 | 189.0 | 150.9 | 167.4 | 27.1 | 32.0 | growth |
| loaded | managed_heap | yes | 103 | 270.3 | 270.4 | 269.9 | 258.9 | 271.7 | 2.2 | 1.8 | 2.4 | 3.2 | 32.0 | plateau |
| loaded | working_set | yes | 103 | 5188.6 | 5464.0 | 5115.5 | 4420.3 | 5714.4 | 256.5 | 204.9 | 161.4 | 191.7 | 102.3 | growth |
| loaded | private | no | 103 | 7906.6 | 8033.8 | 7991.5 | 6827.6 | 8326.4 | 465.5 | 371.7 | 126.8 | 103.9 | 159.8 | growth |
| loaded | rss_anon | yes | 103 | 5113.9 | 5387.6 | 5039.4 | 4345.7 | 5638.2 | 254.7 | 203.4 | 160.1 | 191.3 | 100.8 | growth |

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
| git_commit | 9c972ca9e46d1614d819ecd497de352cd32cd759 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38882436529, variant cpu-arena --cpu-arena |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38882436529, variant cpu-arena |
| load before (loadavg, MemAvailable MB) | [2.27,0.76,0.28], 14212 |
| load after (loadavg, MemAvailable MB) | [6.05,6.49,6.43], 13835 |
