# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 707 |
| Keep loaded (diagnostic) | yes: one load, no unloads; the `unloaded` series is a second sample with the model loaded |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 38178; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 0, 3071 |
| Unload ms (median, max) | 0.0, 0.0 |
| Request p50 ms (median over cycles) | 96.0 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 4242 | 0 | 629.71 | 856.63 | 1249.90 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 568 | 177.2 | 214.2 | 178.3 | 144.6 | 214.2 | -3.2 | -2.5 | -11.8 | -9.4 | 32.0 | plateau |
| unloaded | working_set | yes | 568 | 1680.3 | 1705.0 | 1690.8 | 1680.3 | 1705.0 | 22.4 | 17.9 | 7.7 | 12.6 | 33.8 | plateau |
| unloaded | private | no | 568 | 2005.3 | 2061.7 | 2022.0 | 1974.5 | 2087.9 | 18.5 | 14.8 | -5.4 | 3.3 | 40.4 | plateau |
| unloaded | rss_anon | yes | 568 | 1606.7 | 1630.6 | 1616.5 | 1606.7 | 1630.6 | 21.4 | 17.1 | 7.0 | 12.5 | 32.3 | plateau |
| loaded | managed_heap | yes | 568 | 177.2 | 214.2 | 178.3 | 144.6 | 214.2 | -3.2 | -2.5 | -11.8 | -9.4 | 32.0 | plateau |
| loaded | working_set | yes | 568 | 1680.3 | 1705.0 | 1690.8 | 1680.3 | 1705.0 | 22.4 | 17.9 | 7.7 | 12.6 | 33.8 | plateau |
| loaded | private | no | 568 | 2005.3 | 2061.7 | 2022.0 | 1974.5 | 2087.9 | 18.5 | 14.8 | -5.4 | 3.3 | 40.4 | plateau |
| loaded | rss_anon | yes | 568 | 1606.7 | 1630.6 | 1616.5 | 1606.7 | 1630.6 | 21.4 | 17.1 | 7.0 | 12.5 | 32.3 | plateau |

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
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant keep-loaded\u002Bsmaps5 --keep-loaded --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant keep-loaded\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [2.14,0.63,0.22], 14138 |
| load after (loadavg, MemAvailable MB) | [7.12,6.97,6.73], 14015 |
