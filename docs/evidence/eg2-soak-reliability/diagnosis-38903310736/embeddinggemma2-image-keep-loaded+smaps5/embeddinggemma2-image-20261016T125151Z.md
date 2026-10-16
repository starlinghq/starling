# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3606 s |
| Load/unload cycles | 115 |
| Keep loaded (diagnostic) | yes: one load, no unloads; the `unloaded` series is a second sample with the model loaded |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 460; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 0, 2979 |
| Unload ms (median, max) | 0.0, 0.0 |
| Request p50 ms (median over cycles) | 15025.7 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 115 | 0 | 15.89 | 54.83 | 183.10 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 95 | 101.8 | 101.9 | 102.0 | 101.8 | 103.6 | 0.6 | 0.5 | 0.3 | 0.2 | 32.0 | plateau |
| unloaded | working_set | yes | 95 | 2958.6 | 2928.1 | 2849.8 | 2779.9 | 3000.3 | 4.4 | 3.5 | -57.6 | -16.0 | 57.0 | plateau |
| unloaded | private | no | 95 | 3234.9 | 3262.7 | 3229.3 | 3174.3 | 3336.8 | 4.7 | 3.7 | -27.6 | -8.6 | 64.6 | plateau |
| unloaded | rss_anon | yes | 95 | 2884.5 | 2853.0 | 2775.3 | 2705.1 | 2925.7 | 3.1 | 2.5 | -58.5 | -16.1 | 55.5 | plateau |
| loaded | managed_heap | yes | 95 | 101.8 | 101.9 | 102.0 | 101.8 | 112.7 | 0.7 | 0.6 | 0.3 | -0.8 | 32.0 | plateau |
| loaded | working_set | yes | 95 | 2958.6 | 2928.1 | 2849.8 | 2779.9 | 3000.3 | 4.5 | 3.6 | -57.6 | -16.7 | 57.0 | plateau |
| loaded | private | no | 95 | 3234.9 | 3262.7 | 3229.3 | 3174.3 | 3336.8 | 4.8 | 3.8 | -27.6 | -9.4 | 64.6 | plateau |
| loaded | rss_anon | yes | 95 | 2884.5 | 2853.0 | 2775.3 | 2705.1 | 2925.7 | 3.2 | 2.5 | -58.5 | -16.8 | 55.5 | plateau |

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
| git_commit | f43a648a795ed6cda366007a61fd7b4158a93082 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant keep-loaded\u002Bsmaps5 --keep-loaded --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant keep-loaded\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [2.13,0.67,0.24], 14148 |
| load after (loadavg, MemAvailable MB) | [7.66,7.24,6.99], 14204 |
