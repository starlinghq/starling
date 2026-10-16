# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3600 s |
| Load/unload cycles | 936 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 50544; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1260, 1601 |
| Unload ms (median, max) | 103.1, 168.5 |
| Request p50 ms (median over cycles) | 44.5 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 5616 | 0 | 292.85 | 374.79 | 507.29 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 745 | 288.5 | 256.3 | 288.6 | 255.9 | 325.5 | 11.3 | 9.0 | 1.0 | 1.4 | 32.0 | plateau |
| unloaded | working_set | yes | 745 | 498.4 | 503.8 | 476.9 | 455.8 | 515.3 | 18.3 | 14.6 | -10.0 | 22.2 | 32.0 | plateau |
| unloaded | private | no | 745 | 1724.5 | 1676.0 | 1686.0 | 1642.6 | 1774.4 | 28.1 | 22.5 | -10.6 | 22.1 | 33.7 | plateau |
| unloaded | rss_anon | yes | 745 | 423.4 | 428.3 | 401.4 | 380.6 | 439.8 | 17.6 | 14.0 | -10.5 | 22.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 745 | 367.2 | 335.0 | 367.3 | 334.6 | 404.3 | 12.1 | 9.7 | 0.9 | 2.5 | 32.0 | plateau |
| loaded | working_set | yes | 745 | 1919.8 | 1920.7 | 1891.6 | 1856.7 | 1942.6 | 20.8 | 16.6 | -8.2 | 22.4 | 37.8 | plateau |
| loaded | private | no | 745 | 2260.6 | 2226.2 | 2226.2 | 2182.3 | 2310.5 | 30.0 | 24.0 | -8.8 | 22.3 | 44.5 | plateau |
| loaded | rss_anon | yes | 745 | 1844.8 | 1845.2 | 1816.3 | 1781.5 | 1867.1 | 20.1 | 16.0 | -8.7 | 22.4 | 36.3 | plateau |

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
| git_commit | f43a648a795ed6cda366007a61fd7b4158a93082 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant smaps5\u002Bdefault --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant smaps5\u002Bdefault |
| load before (loadavg, MemAvailable MB) | [2.53,0.75,0.26], 14095 |
| load after (loadavg, MemAvailable MB) | [2.21,2.21,2.17], 13720 |
