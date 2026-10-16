# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 506 MB; loaded private (address space, report-only) grows 515 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3621 s |
| Load/unload cycles | 101 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 404; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2480, 2986 |
| Unload ms (median, max) | 154.1, 167.3 |
| Request p50 ms (median over cycles) | 15803.1 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 101 | 0 | 10.95 | 114.83 | 154.63 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 84 | 181.8 | 181.2 | 181.3 | 180.0 | 182.6 | 0.8 | 0.7 | 0.2 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 84 | 435.6 | 508.8 | 499.7 | 397.6 | 568.0 | 28.8 | 23.1 | 72.2 | 19.7 | 32.0 | plateau |
| unloaded | private | no | 84 | 2742.9 | 3114.5 | 3052.9 | 2641.7 | 3279.4 | 630.3 | 506.4 | 579.9 | 147.1 | 61.1 | growth |
| unloaded | rss_anon | yes | 84 | 361.0 | 432.5 | 423.5 | 322.9 | 492.4 | 27.0 | 21.7 | 71.3 | 19.6 | 32.0 | plateau |
| loaded | managed_heap | yes | 84 | 260.5 | 259.9 | 260.0 | 258.8 | 261.4 | 0.8 | 0.7 | 0.2 | 0.4 | 32.0 | plateau |
| loaded | working_set | yes | 84 | 2650.0 | 2698.1 | 2743.4 | 2610.9 | 2919.9 | 111.8 | 89.8 | 205.8 | -0.1 | 54.9 | plateau |
| loaded | private | no | 84 | 3331.6 | 3706.6 | 3683.3 | 3217.8 | 3879.6 | 640.5 | 514.7 | 613.3 | 138.3 | 73.7 | growth |
| loaded | rss_anon | yes | 84 | 2575.5 | 2621.9 | 2667.1 | 2535.3 | 2844.3 | 109.9 | 88.3 | 204.6 | -0.2 | 53.3 | plateau |

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
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38882436529 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38882436529 |
| load before (loadavg, MemAvailable MB) | [3.41,1.02,0.37], 14230 |
| load after (loadavg, MemAvailable MB) | [6.85,6.34,6.26], 14065 |
