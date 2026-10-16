# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3618 s |
| Load/unload cycles | 177 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 708; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1810, 2366 |
| Unload ms (median, max) | 156.1, 205.7 |
| Request p50 ms (median over cycles) | 8882.3 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 177 | 0 | 41.78 | 106.99 | 134.83 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 145 | 181.3 | 181.2 | 191.2 | 180.1 | 193.0 | 0.2 | 0.2 | 0.5 | -2.7 | 32.0 | plateau |
| unloaded | working_set | yes | 145 | 518.8 | 496.7 | 534.4 | 420.8 | 589.7 | 21.5 | 17.3 | 35.1 | 5.9 | 32.0 | plateau |
| unloaded | private | no | 145 | 2879.8 | 2886.1 | 2949.8 | 2817.2 | 3076.4 | 101.8 | 81.8 | 48.2 | -59.4 | 59.0 | plateau |
| unloaded | rss_anon | yes | 145 | 444.0 | 420.6 | 458.3 | 345.9 | 513.7 | 20.1 | 16.2 | 33.9 | 5.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 145 | 260.0 | 259.9 | 269.9 | 258.8 | 271.7 | 0.2 | 0.2 | 0.5 | -2.7 | 32.0 | plateau |
| loaded | working_set | yes | 145 | 2699.3 | 2621.4 | 2787.7 | 2621.4 | 2915.2 | 22.4 | 18.0 | 75.7 | -95.5 | 55.8 | plateau |
| loaded | private | no | 145 | 3434.8 | 3414.3 | 3491.7 | 3372.3 | 3620.0 | 82.4 | 66.2 | 39.7 | -74.1 | 69.8 | plateau |
| loaded | rss_anon | yes | 145 | 2624.5 | 2545.3 | 2711.9 | 2545.3 | 2839.7 | 21.0 | 16.9 | 74.5 | -95.5 | 54.2 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V45 96-Core Processor |
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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38882436529, variant smaps --smaps-every 25 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38882436529, variant smaps |
| load before (loadavg, MemAvailable MB) | [2.62,0.93,0.34], 14168 |
| load after (loadavg, MemAvailable MB) | [2.82,2.93,2.9], 14066 |
