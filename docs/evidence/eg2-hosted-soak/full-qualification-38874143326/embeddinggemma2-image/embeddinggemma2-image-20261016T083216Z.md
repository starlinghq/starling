# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** FAIL: unloaded working_set grows 106 MB (133 MB/h); unloaded rss_anon grows 105 MB (132 MB/h); loaded working_set grows 148 MB (185 MB/h); loaded rss_anon grows 146 MB (184 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3614 s |
| Load/unload cycles | 113 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 452; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3059, 3691 |
| Unload ms (median, max) | 113.2, 128.3 |
| Request p50 ms (median over cycles) | 13670.6 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 113 | 0 | 9.61 | 133.80 | 154.62 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 93 | 181.0 | 180.2 | 181.1 | 180.0 | 182.7 | 0.2 | 0.2 | 0.0 | -0.2 | 32.0 | plateau |
| unloaded | working_set | yes | 93 | 502.2 | 535.3 | 520.6 | 437.9 | 660.9 | 133.0 | 106.0 | 25.5 | 29.5 | 32.0 | growth |
| unloaded | private | no | 93 | 2795.4 | 2924.0 | 2873.1 | 2721.1 | 3017.4 | 58.8 | 46.9 | 10.9 | -93.0 | 57.5 | plateau |
| unloaded | rss_anon | yes | 93 | 427.6 | 459.2 | 444.6 | 362.5 | 584.8 | 131.6 | 104.8 | 24.6 | 29.4 | 32.0 | growth |
| loaded | managed_heap | yes | 93 | 259.8 | 258.9 | 259.8 | 258.8 | 261.4 | 0.2 | 0.2 | 0.0 | -0.2 | 32.0 | plateau |
| loaded | working_set | yes | 93 | 2847.7 | 2844.2 | 2769.3 | 2604.6 | 2928.4 | 185.2 | 147.6 | 51.9 | 30.8 | 55.4 | growth |
| loaded | private | no | 93 | 3364.4 | 3460.1 | 3418.6 | 3233.1 | 3569.6 | 70.3 | 56.0 | 32.4 | -76.0 | 68.4 | plateau |
| loaded | rss_anon | yes | 93 | 2773.1 | 2768.1 | 2693.2 | 2529.7 | 2852.4 | 183.6 | 146.3 | 50.7 | 30.8 | 53.9 | growth |

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
| git_commit | 6df29602d9f84e48e7ce5a1a6c8b8f96c4fbf50c |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38874143326 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38874143326 |
| load before (loadavg, MemAvailable MB) | [2.86,1.08,0.4], 14083 |
| load after (loadavg, MemAvailable MB) | [3.41,3.1,3.03], 13878 |
