# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** FAIL: unloaded working_set grows 139 MB (172 MB/h); unloaded rss_anon grows 137 MB (171 MB/h); loaded working_set grows 229 MB (284 MB/h); loaded rss_anon grows 227 MB (283 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3620 s |
| Load/unload cycles | 149 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 447; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2944, 3795 |
| Unload ms (median, max) | 240.6, 261.3 |
| Request p50 ms (median over cycles) | 11336.6 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 149 | 0 | 14.20 | 52.98 | 85.53 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 123 | 199.1 | 215.6 | 200.5 | 198.2 | 216.9 | 0.9 | 0.7 | 4.1 | -2.5 | 32.0 | plateau |
| unloaded | working_set | yes | 123 | 643.9 | 854.1 | 792.0 | 642.2 | 933.7 | 172.3 | 138.5 | 72.7 | 112.7 | 32.0 | growth |
| unloaded | private | no | 123 | 2601.9 | 2980.3 | 2904.8 | 2601.9 | 3016.6 | 353.4 | 284.0 | 314.8 | 62.7 | 58.1 | growth |
| unloaded | rss_anon | yes | 123 | 569.1 | 777.9 | 715.8 | 567.1 | 857.5 | 170.9 | 137.3 | 71.7 | 112.6 | 32.0 | growth |
| loaded | managed_heap | yes | 123 | 277.9 | 294.4 | 279.2 | 276.9 | 295.6 | 0.9 | 0.7 | 4.1 | -2.5 | 32.0 | plateau |
| loaded | working_set | yes | 123 | 3927.7 | 4239.8 | 4190.3 | 3722.4 | 4493.0 | 284.5 | 228.6 | 264.7 | 102.0 | 83.8 | growth |
| loaded | private | no | 123 | 5525.1 | 5876.4 | 5788.6 | 5443.8 | 5882.5 | 352.8 | 283.5 | 329.1 | 68.4 | 115.8 | growth |
| loaded | rss_anon | yes | 123 | 3853.0 | 4163.6 | 4114.1 | 3647.1 | 4416.8 | 283.0 | 227.5 | 263.7 | 101.9 | 82.3 | growth |

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
| git_commit | f3edb065f04abbbf817e2b10a77e4d555de7cf95 |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38848321624 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38848321624 |
| load before (loadavg, MemAvailable MB) | [2.54,0.78,0.27], 14128 |
| load after (loadavg, MemAvailable MB) | [6.57,6.45,6.44], 14100 |
