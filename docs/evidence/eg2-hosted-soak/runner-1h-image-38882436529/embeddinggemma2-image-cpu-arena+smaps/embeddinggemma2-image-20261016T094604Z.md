# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=on, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=on, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3618 s |
| Load/unload cycles | 147 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 588; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2996, 3699 |
| Unload ms (median, max) | 173.2, 198.8 |
| Request p50 ms (median over cycles) | 10015.2 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 147 | 0 | 10.44 | 54.59 | 148.25 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 120 | 191.9 | 193.0 | 191.7 | 181.6 | 193.0 | 3.3 | 2.6 | 1.2 | 0.9 | 32.0 | plateau |
| unloaded | working_set | yes | 120 | 577.6 | 610.6 | 620.8 | 540.6 | 727.4 | 29.8 | 23.8 | -18.9 | -62.7 | 32.0 | plateau |
| unloaded | private | no | 120 | 2746.8 | 2837.3 | 2822.5 | 2714.9 | 2903.0 | 154.1 | 123.2 | 100.2 | 19.3 | 56.4 | retention |
| unloaded | rss_anon | yes | 120 | 502.9 | 534.2 | 544.9 | 464.6 | 651.1 | 28.3 | 22.7 | -19.9 | -62.8 | 32.0 | plateau |
| loaded | managed_heap | yes | 120 | 270.6 | 271.7 | 270.4 | 260.3 | 271.7 | 3.3 | 2.6 | 1.2 | 0.9 | 32.0 | plateau |
| loaded | working_set | yes | 120 | 5211.5 | 5188.6 | 5137.3 | 4348.5 | 5581.9 | 46.9 | 37.5 | 124.0 | -141.3 | 102.7 | plateau |
| loaded | private | no | 120 | 7707.0 | 8013.5 | 7753.7 | 6840.0 | 8079.2 | 125.7 | 100.6 | 288.4 | -247.9 | 155.1 | plateau |
| loaded | rss_anon | yes | 120 | 5136.8 | 5112.3 | 5060.9 | 4272.1 | 5505.6 | 45.4 | 36.3 | 122.9 | -141.5 | 101.2 | plateau |

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
| git_commit | 9c972ca9e46d1614d819ecd497de352cd32cd759 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38882436529, variant cpu-arena\u002Bsmaps --cpu-arena --smaps-every 25 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38882436529, variant cpu-arena\u002Bsmaps |
| load before (loadavg, MemAvailable MB) | [2.39,1.14,0.45], 13997 |
| load after (loadavg, MemAvailable MB) | [3.01,3.07,3.03], 14178 |
